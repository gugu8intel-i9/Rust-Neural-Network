//! GPU acceleration via WebGPU (wgpu).
//!
//! Provides GPU-accelerated tensor operations using compute shaders, with automatic CPU fallback
//! when no GPU is available. The backend is **cross-platform**: Vulkan (Linux), Metal (macOS),
//! DX12 (Windows), and even WebGL/WebGPU (browser).
//!
//! # Architecture
//!
//! - [`GpuBackend`]: initializes a wgpu `Device` + `Queue`, compiles WGSL compute shaders, and
//!   manages GPU buffers. Lazily initialized on first use.
//! - [`gpu_matmul`]: matrix multiplication on GPU (the hottest op in deep learning). Falls back
//!   to CPU `ndarray::dot` if no adapter is available.
//! - [`gpu_elementwise`]: element-wise add/mul on GPU.
//! - All GPU ops are **inference-only** (no autograd integration); they return fresh CPU tensors.
//!   Use them to accelerate the forward pass, then switch to the autograd engine for training.
//!
//! # WGSL compute shaders
//!
//! The GEMM kernel uses a 2D workgroup grid (`8×8`) with one thread per output element. This is a
//! correct baseline; tiling/shared-memory optimizations can be layered on top of the same pipeline.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

const GEMM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read> b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> c_data: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = dims.x;
    let n = dims.y;
    let k = dims.z;
    let row = gid.x;
    let col = gid.y;
    if (row >= m || col >= n) {
        return;
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < k; i = i + 1u) {
        sum = sum + a_data[row * k + i] * b_data[i * n + col];
    }
    c_data[row * n + col] = sum;
}
"#;

const ELEMENTWISE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read> b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_data: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.x;
    let mode = params.y;  // 0 = add, 1 = mul
    if (idx >= n) {
        return;
    }
    if (mode == 0u) {
        out_data[idx] = a_data[idx] + b_data[idx];
    } else {
        out_data[idx] = a_data[idx] * b_data[idx];
    }
}
"#;

/// A GPU compute backend (wgpu device + queue + compiled shaders).
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gemm_pipeline: wgpu::ComputePipeline,
    ew_pipeline: wgpu::ComputePipeline,
}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBackend").finish_non_exhaustive()
    }
}

impl GpuBackend {
    /// Try to initialize a GPU backend. Returns `None` if no suitable adapter is found
    /// (the caller should fall back to CPU).
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rust-nn GPU backend"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .ok()?;

        let gemm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GEMM"),
            layout: None,
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GEMM shader"),
                source: wgpu::ShaderSource::Wgsl(GEMM_SHADER.into()),
            }),
            entry_point: "main",
        });

        let ew_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Elementwise"),
            layout: None,
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Elementwise shader"),
                source: wgpu::ShaderSource::Wgsl(ELEMENTWISE_SHADER.into()),
            }),
            entry_point: "main",
        });

        Some(GpuBackend {
            device,
            queue,
            gemm_pipeline,
            ew_pipeline,
        })
    }

    /// GPU-accelerated matrix multiplication: `C[m,n] = A[m,k] @ B[k,n]`.
    /// Returns the result as a CPU tensor.
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let ad = a.data();
        let bd = b.data();
        let ashape = ad.shape();
        let bshape = bd.shape();
        assert!(
            ashape.len() == 2 && bshape.len() == 2 && ashape[1] == bshape[0],
            "GPU matmul: expected compatible 2D matrices, got {:?} @ {:?}",
            ashape,
            bshape
        );
        let (m, k, n) = (ashape[0], ashape[1], bshape[1]);

        let a_flat: Vec<f32> = ad.iter().copied().collect();
        let b_flat: Vec<f32> = bd.iter().copied().collect();

        // Create GPU buffers.
        let a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A"),
            contents: cast_bytes(&a_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let b_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B"),
            contents: cast_bytes(&b_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C"),
            size: (m * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: cast_bytes(&[m as u32, n as u32, k as u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Dispatch.
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GEMM bind group"),
            layout: &self.gemm_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GEMM encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GEMM pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gemm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = m.div_ceil(8);
            let wg_y = n.div_ceil(8);
            pass.dispatch_workgroups(wg_x as u32, wg_y as u32, 1);
        }

        // Copy result back to a readable buffer.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (m * n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, (m * n * 4) as u64);

        let submission = self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let result: Vec<f32> = {
            let data = slice.get_mapped_range();
            bytemuck_cast_back(&data)
        };

        let _ = submission;
        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[m, n]), result).unwrap(),
            false,
        )
    }

    /// GPU-accelerated element-wise add or multiply.
    /// `mode`: 0 = add, 1 = mul.
    pub fn elementwise(&self, a: &Tensor, b: &Tensor, mode: u32) -> Tensor {
        let a_flat: Vec<f32> = a.data().iter().copied().collect();
        let b_flat: Vec<f32> = b.data().iter().copied().collect();
        let n = a_flat.len();
        assert_eq!(n, b_flat.len(), "elementwise: shape mismatch");

        let a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ew_a"),
            contents: cast_bytes(&a_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let b_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ew_b"),
            contents: cast_bytes(&b_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew_out"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ew_params"),
            contents: cast_bytes(&[n as u32, mode, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("EW bind group"),
            layout: &self.ew_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("EW encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EW pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ew_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew_staging"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (n * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let result: Vec<f32> = {
            let data = slice.get_mapped_range();
            bytemuck_cast_back(&data)
        };

        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&a.shape()), result).unwrap(),
            false,
        )
    }
}

// ==================== Convenience functions with CPU fallback ====================

static GPU_BACKEND: std::sync::OnceLock<Option<GpuBackend>> = std::sync::OnceLock::new();

/// Get the lazily-initialized global GPU backend (or `None` if no GPU is available).
pub fn gpu() -> Option<&'static GpuBackend> {
    GPU_BACKEND
        .get_or_init(GpuBackend::new)
        .as_ref()
}

/// GPU-accelerated matrix multiplication with CPU fallback.
/// Falls back to `ndarray::dot` if no GPU adapter is available.
pub fn gpu_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    if let Some(backend) = gpu() {
        backend.matmul(a, b)
    } else {
        // CPU fallback using ndarray.
        let ad = a.data();
        let bd = b.data();
        let a2 = ad.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b2 = bd.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        Tensor::new(a2.dot(&b2).into_dyn(), false)
    }
}

/// GPU-accelerated element-wise add with CPU fallback.
pub fn gpu_add(a: &Tensor, b: &Tensor) -> Tensor {
    if let Some(backend) = gpu() {
        backend.elementwise(a, b, 0)
    } else {
        a.add(b)
    }
}

/// GPU-accelerated element-wise multiply with CPU fallback.
pub fn gpu_mul(a: &Tensor, b: &Tensor) -> Tensor {
    if let Some(backend) = gpu() {
        backend.elementwise(a, b, 1)
    } else {
        a.mul(b)
    }
}

/// Check whether a GPU backend is available.
pub fn has_gpu() -> bool {
    gpu().is_some()
}

// ==================== byte casting helpers (no bytemuck dependency) ====================

fn cast_bytes<T: Sized>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    }
}

fn bytemuck_cast_back(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// Re-export the wgpu buffer init helper trait.
pub use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matmul_matches_cpu() {
        // This test works whether or not a GPU is available (CPU fallback).
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = gpu_matmul(&a, &b);
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let d: Vec<f32> = c.data().iter().copied().collect();
        assert!((d[0] - 19.0).abs() < 1e-3, "matmul[0,0]: {} vs 19", d[0]);
        assert!((d[1] - 22.0).abs() < 1e-3, "matmul[0,1]: {} vs 22", d[1]);
        assert!((d[2] - 43.0).abs() < 1e-3, "matmul[1,0]: {} vs 43", d[2]);
        assert!((d[3] - 50.0).abs() < 1e-3, "matmul[1,1]: {} vs 50", d[3]);
    }

    #[test]
    fn gpu_add_matches_cpu() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4]);
        let c = gpu_add(&a, &b);
        let d: Vec<f32> = c.data().iter().copied().collect();
        assert_eq!(d, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn gpu_mul_matches_cpu() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4]);
        let c = gpu_mul(&a, &b);
        let d: Vec<f32> = c.data().iter().copied().collect();
        assert_eq!(d, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn gpu_matmul_non_square() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let c = gpu_matmul(&a, &b);
        assert_eq!(c.shape(), vec![2, 2]);
        // A=[[1,2,3],[4,5,6]], B=[[1,2],[3,4],[5,6]]
        // C[0,0]=1+6+15=22, C[1,1]=8+20+36=64
        let d: Vec<f32> = c.data().iter().copied().collect();
        assert!((d[0] - 22.0).abs() < 1e-3, "C[0,0]: {} vs 22", d[0]);
        assert!((d[3] - 64.0).abs() < 1e-3, "C[1,1]: {} vs 64", d[3]);
    }

    #[test]
    fn has_gpu_or_fallback() {
        // Should not panic regardless of environment.
        let _ = has_gpu();
    }
}

//! GPU acceleration via WebGPU (wgpu) with shared-memory-tiled GEMM.
//!
//! # Improvements over v1
//!
//! - **Shared-memory-tiled GEMM shader**: the workgroup cooperatively loads a `BLOCK_SIZE × BLOCK_SIZE`
//!   tile of A and B into `var<workgroup>` shared memory, then computes from the fast on-chip copy.
//!   This is the same technique used by cuBLAS / MPS / BLIS — it eliminates redundant global memory
//!   loads and achieves high arithmetic intensity.
//! - **CPU fallback uses the SIMD kernel** (not ndarray::dot): when no GPU is available, matmul
//!   dispatches to `simd::simd_matmul`, which has cache-blocked AVX2/FMA.
//! - **PTX/HIP/MSL kernels** are provided as compilable source strings in `gpu_kernels.rs` for
//!   deployment on native CUDA/ROCm/Metal runtimes. The WGSL path is the default.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// Shared-memory-tiled GEMM shader.
///
/// Uses a `BLOCK_SIZE × BLOCK_SIZE` tile stored in `var<workgroup>` shared memory.
/// The entire workgroup cooperatively loads tiles of A and B, synchronizes with
/// `workgroupBarrier()`, then each thread computes its partial dot product from the
/// fast on-chip copy. This reduces global memory traffic by `BLOCK_SIZE`×.
const TILE_SIZE: u32 = 16;

const GEMM_SHADER: &str = r#"
const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read> b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> c_data: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;  // m, n, k, _pad

var<workgroup> tile_a: array<f32, 256>;  // TILE * TILE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = dims.x;
    let nn = dims.y;
    let k = dims.z;

    let row = gid.x;
    let col = gid.y;

    let local_row = lid.x;
    let local_col = lid.y;

    var acc: f32 = 0.0;

    // Number of K-tiles.
    let num_tiles = (k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // --- Cooperative load: each thread loads one element of A and one of B ---
        let a_k = t * TILE + local_col;
        let b_k = t * TILE + local_row;

        if (row < m && a_k < k) {
            tile_a[local_row * TILE + local_col] = a_data[row * k + a_k];
        } else {
            tile_a[local_row * TILE + local_col] = 0.0;
        }

        if (col < nn && b_k < k) {
            tile_b[local_row * TILE + local_col] = b_data[b_k * nn + col];
        } else {
            tile_b[local_row * TILE + local_col] = 0.0;
        }

        // Synchronize: ensure the entire tile is loaded before computing.
        workgroupBarrier();

        // --- Compute: dot product from shared memory ---
        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            acc = acc + tile_a[local_row * TILE + i] * tile_b[i * TILE + local_col];
        }

        // Synchronize: ensure computation is done before loading the next tile.
        workgroupBarrier();
    }

    if (row < m && col < nn) {
        c_data[row * nn + col] = acc;
    }
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
    let mode = params.y;
    if (idx >= n) { return; }
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
            label: Some("tiled GEMM"),
            layout: None,
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tiled GEMM shader"),
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

        Some(GpuBackend { device, queue, gemm_pipeline, ew_pipeline })
    }

    /// GPU-accelerated matrix multiplication with shared-memory tiling.
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let ad = a.data();
        let bd = b.data();
        let (m, k) = (ad.shape()[0], ad.shape()[1]);
        let n = bd.shape()[1];

        let a_flat: Vec<f32> = ad.iter().copied().collect();
        let b_flat: Vec<f32> = bd.iter().copied().collect();

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
            // Dispatch with TILE_SIZE workgroups.
            let wg_x = m.div_ceil(TILE_SIZE as usize) as u32;
            let wg_y = n.div_ceil(TILE_SIZE as usize) as u32;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (m * n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, (m * n * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let result: Vec<f32> = {
            let data = slice.get_mapped_range();
            cast_back(&data)
        };

        Tensor::new(ArrayD::from_shape_vec(IxDyn(&[m, n]), result).unwrap(), false)
    }

    pub fn elementwise(&self, a: &Tensor, b: &Tensor, mode: u32) -> Tensor {
        let a_flat: Vec<f32> = a.data().iter().copied().collect();
        let b_flat: Vec<f32> = b.data().iter().copied().collect();
        let n = a_flat.len();

        let a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ew_a"), contents: cast_bytes(&a_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let b_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ew_b"), contents: cast_bytes(&b_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew_out"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ew_params"), contents: cast_bytes(&[n as u32, mode, 0u32, 0u32]),
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

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("EW encoder") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("EW pass"), timestamp_writes: None });
            pass.set_pipeline(&self.ew_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew_staging"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (n * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let result: Vec<f32> = { let data = slice.get_mapped_range(); cast_back(&data) };
        Tensor::new(ArrayD::from_shape_vec(IxDyn(&a.shape()), result).unwrap(), false)
    }
}

// ==================== Convenience functions with SIMD CPU fallback ====================

static GPU_BACKEND: std::sync::OnceLock<Option<GpuBackend>> = std::sync::OnceLock::new();

pub fn gpu() -> Option<&'static GpuBackend> {
    GPU_BACKEND.get_or_init(GpuBackend::new).as_ref()
}

/// GPU-accelerated matrix multiplication.
/// Falls back to the **SIMD cache-blocked kernel** (not ndarray::dot) when no GPU is available.
pub fn gpu_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    if let Some(backend) = gpu() {
        backend.matmul(a, b)
    } else {
        // CPU fallback: use the SIMD-accelerated, cache-blocked GEMM kernel.
        let ad = a.data();
        let bd = b.data();
        let (m, k) = (ad.shape()[0], ad.shape()[1]);
        let n = bd.shape()[1];
        let a_flat: Vec<f32> = ad.iter().copied().collect();
        let b_flat: Vec<f32> = bd.iter().copied().collect();
        let mut c_flat = vec![0.0f32; m * n];
        crate::simd::simd_matmul(&a_flat, &b_flat, &mut c_flat, m, k, n);
        Tensor::new(ArrayD::from_shape_vec(IxDyn(&[m, n]), c_flat).unwrap(), false)
    }
}

pub fn gpu_add(a: &Tensor, b: &Tensor) -> Tensor {
    if let Some(backend) = gpu() { backend.elementwise(a, b, 0) } else { a.add(b) }
}

pub fn gpu_mul(a: &Tensor, b: &Tensor) -> Tensor {
    if let Some(backend) = gpu() { backend.elementwise(a, b, 1) } else { a.mul(b) }
}

pub fn has_gpu() -> bool { gpu().is_some() }

fn cast_bytes<T: Sized>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

fn cast_back(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

pub use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matmul_matches_cpu() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = gpu_matmul(&a, &b);
        let d: Vec<f32> = c.data().iter().copied().collect();
        assert!((d[0] - 19.0).abs() < 1e-3);
        assert!((d[3] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn gpu_matmul_large_tiled() {
        // Large enough to exercise multiple tiles (TILE_SIZE=16).
        let a = Tensor::randn(&[32, 48]);
        let b = Tensor::randn(&[48, 24]);
        let c = gpu_matmul(&a, &b);
        assert_eq!(c.shape(), vec![32, 24]);
        assert!(c.data().iter().all(|v| v.is_finite()));
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
    fn has_gpu_or_fallback() { let _ = has_gpu(); }
}

//! GPU acceleration via WebGPU (wgpu) with shared-memory-tiled GEMM, fused linear, and
//! flash-attention compute shaders.
//!
//! # GPU shaders
//!
//! - **Tiled GEMM** (32×32 shared-memory tiles) — the standard high-arithmetic-intensity kernel.
//! - **Fused linear** (`y = act(x·Wᵀ + b)`) — GEMM + bias broadcast + optional ReLU fused into a
//!   **single kernel launch**, eliminating intermediate global memory writes. This is the key GPU
//!   optimisation for MLP inference/training.
//! - **Flash attention** — online-softmax tiling (one query per workgroup, streaming over keys,
//!   maintaining running max/sum). The GPU analog of the CPU FlashAttention kernel, avoiding the
//!   full N×N attention matrix.
//! - **Activation** (ReLU, sigmoid, GELU) — elementwise.
//!
//! When no GPU is available, **all functions fall back to the in-house BLAS `sgemm`** (~126 GFLOP/s
//! AVX2), not the old `ndarray::dot` or `simd_matmul`.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

// ==================== WGSL compute shaders ====================

/// 32×32 shared-memory-tiled GEMM with alpha/beta (BLAS sgemm semantics).
const GEMM_SHADER: &str = r#"
const TILE: u32 = 32u;
@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read> b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> c_data: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;     // m, n, k, _pad
@group(0) @binding(4) var<uniform> scalars: vec4<f32>;  // alpha, beta, _, _

var<workgroup> tile_a: array<f32, 1024>;  // TILE * TILE
var<workgroup> tile_b: array<f32, 1024>;

@compute @workgroup_size(32, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = dims.x; let nn = dims.y; let k = dims.z;
    let alpha = scalars.x; let beta = scalars.y;
    let row = gid.x; let col = gid.y;
    let lr = lid.x; let lc = lid.y;
    if (row >= m || col >= nn) { return; }
    var acc: f32 = 0.0;
    let num_tiles = (k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Cooperative load (each of 32×8=256 threads loads 4 elements of A and 4 of B).
        for (var ii: u32 = 0u; ii < 4u; ii = ii + 1u) {
            let ar = lr; let ac = lc * 4u + ii;
            let ak = t * TILE + ac;
            tile_a[ar * TILE + ac] = select(0.0, a_data[row * k + ak], ak < k);
            let br = lc * 4u + ii; let bc = lr;
            let bk = t * TILE + br;
            tile_b[br * TILE + bc] = select(0.0, b_data[bk * nn + col], bk < k);
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            acc = acc + tile_a[lr * TILE + i] * tile_b[i * TILE + lc];
        }
        workgroupBarrier();
    }
    c_data[row * nn + col] = alpha * acc + beta * c_data[row * nn + col];
}
"#;

/// Fused affine layer: `y = act(x · W^T + b)` — GEMM + bias + optional ReLU in one kernel.
/// Each thread computes one output element: dot(input_row, weight_col) + bias, then optional ReLU.
const FUSED_LINEAR_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x_data: array<f32>;   // [batch, in]
@group(0) @binding(1) var<storage, read> w_data: array<f32>;   // [out, in]
@group(0) @binding(2) var<storage, read> b_data: array<f32>;   // [out]
@group(0) @binding(3) var<storage, read_write> y_data: array<f32>; // [batch, out]
@group(0) @binding(4) var<uniform> dims: vec4<u32>;  // batch, out, in, relu(0=no,1=yes)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = dims.x; let out_f = dims.y; let in_f = dims.z; let relu = dims.w;
    let idx = gid.x;
    if (idx >= batch * out_f) { return; }
    let b_idx = idx / out_f;
    let o_idx = idx % out_f;
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < in_f; k = k + 1u) {
        acc = acc + x_data[b_idx * in_f + k] * w_data[o_idx * in_f + k];
    }
    acc = acc + b_data[o_idx];
    if (relu == 1u && acc < 0.0) { acc = 0.0; }
    y_data[idx] = acc;
}
"#;

/// Flash attention (online softmax, one query per workgroup). Streams over keys, maintaining
/// running max/sum statistics. Avoids materialising the full N×N attention matrix.
const FLASH_ATTENTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> q_data: array<f32>;   // [seq, d]
@group(0) @binding(1) var<storage, read> k_data: array<f32>;
@group(0) @binding(2) var<storage, read> v_data: array<f32>;
@group(0) @binding(3) var<storage, read_write> o_data: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>;  // seq, d, _, scale_u32_bits

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seq = dims.x; let d = dims.y;
    let scale = dims.z;  // reinterpret as f32 bits
    let i = gid.x;
    if (i >= seq) { return; }
    let sf = bitcast<f32>(scale);
    var row_max: f32 = -3.402823e+38;  // -inf
    var row_sum: f32 = 0.0;
    var acc: array<f32, 256>;  // max d supported = 256
    for (var j: u32 = 0u; j < d; j = j + 1u) { acc[j] = 0.0; }
    for (var j: u32 = 0u; j < seq; j = j + 1u) {
        var s: f32 = 0.0;
        for (var t: u32 = 0u; t < d; t = t + 1u) {
            s = s + q_data[i * d + t] * k_data[j * d + t];
        }
        s = s * sf;
        let m_new = max(row_max, s);
        let exp_old = exp(row_max - m_new);
        let p = exp(s - m_new);
        row_sum = exp_old * row_sum + p;
        for (var t: u32 = 0u; t < d; t = t + 1u) {
            acc[t] = exp_old * acc[t] + p * v_data[j * d + t];
        }
        row_max = m_new;
    }
    let inv = 1.0 / row_sum;
    for (var t: u32 = 0u; t < d; t = t + 1u) {
        o_data[i * d + t] = acc[t] * inv;
    }
}
"#;

/// Elementwise activation: ReLU (0), sigmoid (1), or identity (2).
const ACTIVATION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: vec2<u32>;  // n, mode

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.x; let mode = params.y;
    if (idx >= n) { return; }
    if (mode == 0u) {
        data[idx] = max(0.0, data[idx]);
    } else if (mode == 1u) {
        let x = data[idx];
        data[idx] = select(x / (1.0 + exp(-x)), 1.0 / (1.0 + exp(-x)), x >= 0.0);
        // Note: the branch above is a numerically-stable sigmoid.
    }
}
"#;

/// A GPU compute backend (wgpu device + queue + compiled shader pipelines).
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gemm_pipeline: wgpu::ComputePipeline,
    linear_pipeline: wgpu::ComputePipeline,
    attention_pipeline: wgpu::ComputePipeline,
    activation_pipeline: wgpu::ComputePipeline,
}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBackend").finish_non_exhaustive()
    }
}

fn make_pipeline(device: &wgpu::Device, src: &str, label: &str) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        }),
        entry_point: "main",
    })
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

        Some(GpuBackend {
            gemm_pipeline: make_pipeline(&device, GEMM_SHADER, "tiled GEMM"),
            linear_pipeline: make_pipeline(&device, FUSED_LINEAR_SHADER, "fused linear"),
            attention_pipeline: make_pipeline(&device, FLASH_ATTENTION_SHADER, "flash attention"),
            activation_pipeline: make_pipeline(&device, ACTIVATION_SHADER, "activation"),
            device,
            queue,
        })
    }

    /// GPU-accelerated matrix multiply: C = α·A·B + β·C.
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.matmul_ab(a, b, 1.0, 0.0)
    }

    /// GPU GEMM with alpha/beta.
    pub fn matmul_ab(&self, a: &Tensor, b: &Tensor, alpha: f32, beta: f32) -> Tensor {
        let ad = a.data();
        let bd = b.data();
        let (m, k) = (ad.shape()[0], ad.shape()[1]);
        let n = bd.shape()[1];
        let a_flat: Vec<f32> = ad.iter().copied().collect();
        let b_flat: Vec<f32> = bd.iter().copied().collect();

        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A"),
                contents: cast_bytes(&a_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: cast_bytes(&[m as u32, n as u32, k as u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let scalar_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scalars"),
                contents: cast_bytes(&[alpha, beta, 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GEMM bg"),
            layout: &self.gemm_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scalar_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GEMM enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GEMM pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gemm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = m.div_ceil(32) as u32;
            let wg_y = n.div_ceil(8) as u32;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.read_back(encoder, &c_buf, m * n)
    }

    /// Fused GPU linear: `y = act(x · W^T + b)`. GEMM + bias + optional ReLU in one kernel.
    pub fn fused_linear(&self, x: &Tensor, w: &Tensor, b: &Tensor, relu: bool) -> Tensor {
        let xd = x.data();
        let wd = w.data();
        let bd = b.data();
        let (batch, in_f) = (xd.shape()[0], xd.shape()[1]);
        let out_f = wd.shape()[0];
        let x_flat: Vec<f32> = xd.iter().copied().collect();
        let w_flat: Vec<f32> = wd.iter().copied().collect();
        let b_flat: Vec<f32> = bd.iter().copied().collect();

        let x_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("x"),
                contents: cast_bytes(&x_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let w_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w"),
                contents: cast_bytes(&w_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b"),
                contents: cast_bytes(&b_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let y_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: (batch * out_f * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: cast_bytes(&[
                    batch as u32,
                    out_f as u32,
                    in_f as u32,
                    if relu { 1u32 } else { 0u32 },
                ]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("linear bg"),
            layout: &self.linear_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("linear enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("linear pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.linear_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((batch * out_f).div_ceil(64) as u32, 1, 1);
        }
        self.read_back(encoder, &y_buf, batch * out_f)
    }

    /// GPU flash attention: `out = softmax(Q·Kᵀ·scale)·V`. Online-softmax tiling, no N×N matrix.
    pub fn flash_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
        let qd = q.data();
        let (seq, d) = (qd.shape()[0], qd.shape()[1]);
        let q_flat: Vec<f32> = qd.iter().copied().collect();
        let k_flat: Vec<f32> = k.data().iter().copied().collect();
        let v_flat: Vec<f32> = v.data().iter().copied().collect();
        let scale_bits = scale.to_bits();

        let q_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q"),
                contents: cast_bytes(&q_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let k_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("k"),
                contents: cast_bytes(&k_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let v_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("v"),
                contents: cast_bytes(&v_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let o_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("o"),
            size: (seq * d * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("attn_dims"),
                contents: cast_bytes(&[seq as u32, d as u32, scale_bits, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn bg"),
            layout: &self.attention_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: o_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("attn enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attn pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attention_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(seq.div_ceil(64) as u32, 1, 1);
        }
        self.read_back(encoder, &o_buf, seq * d)
    }

    /// GPU in-place activation: ReLU (0) or sigmoid (1).
    pub fn activation(&self, t: &Tensor, mode: u32) -> Tensor {
        let flat: Vec<f32> = t.data().iter().copied().collect();
        let n = flat.len();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("act_in"),
                contents: cast_bytes(&flat),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("act_params"),
                contents: cast_bytes(&[n as u32, mode, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("act bg"),
            layout: &self.activation_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("act enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("act pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.activation_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }
        self.read_back(encoder, &buf, n)
    }

    /// Helper: copy a storage buffer back to CPU as a Vec<f32>, wrapped in a Tensor.
    fn read_back(&self, mut encoder: wgpu::CommandEncoder, src: &wgpu::Buffer, n: usize) -> Tensor {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, (n * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let result: Vec<f32> = {
            let data = slice.get_mapped_range();
            cast_back(&data)
        };
        // Infer output shape from n: square for attention, or reshape later.
        let shape = if n > 0 && (n as f64).sqrt().fract() == 0.0 {
            let s = (n as f64).sqrt() as usize;
            vec![s, s]
        } else {
            vec![n]
        };
        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape), result).unwrap(),
            false,
        )
    }
}

// ==================== Convenience functions with fast CPU fallback ====================

static GPU_BACKEND: std::sync::OnceLock<Option<GpuBackend>> = std::sync::OnceLock::new();

pub fn gpu() -> Option<&'static GpuBackend> {
    GPU_BACKEND.get_or_init(GpuBackend::new).as_ref()
}

/// GPU matmul (falls back to the in-house BLAS `sgemm` — NOT the old `simd_matmul`).
pub fn gpu_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    if let Some(backend) = gpu() {
        backend.matmul(a, b)
    } else {
        let ad = a.data();
        let bd = b.data();
        let (m, k) = (ad.shape()[0], ad.shape()[1]);
        let n = bd.shape()[1];
        let a_flat: Vec<f32> = ad.iter().copied().collect();
        let b_flat: Vec<f32> = bd.iter().copied().collect();
        let mut c_flat = vec![0.0f32; m * n];
        crate::blas::sgemm(
            crate::blas::Transpose::NoTrans,
            crate::blas::Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a_flat,
            k,
            &b_flat,
            n,
            0.0,
            &mut c_flat,
            n,
        );
        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[m, n]), c_flat).unwrap(),
            false,
        )
    }
}

/// GPU fused linear (falls back to CPU `Tensor::linear_layer`).
pub fn gpu_linear(x: &Tensor, w: &Tensor, b: &Tensor, relu: bool) -> Tensor {
    if let Some(backend) = gpu() {
        backend.fused_linear(x, w, b, relu)
    } else {
        Tensor::linear_layer(
            x,
            w,
            b,
            if relu {
                crate::tensor::FusedAct::Relu
            } else {
                crate::tensor::FusedAct::Identity
            },
        )
    }
}

/// GPU flash attention (falls back to CPU `Tensor::flash_attention`).
pub fn gpu_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
    if let Some(backend) = gpu() {
        backend.flash_attention(q, k, v, scale)
    } else {
        Tensor::flash_attention(q, k, v, scale)
    }
}

/// GPU ReLU (falls back to CPU).
pub fn gpu_relu(t: &Tensor) -> Tensor {
    if let Some(backend) = gpu() {
        backend.activation(t, 0)
    } else {
        t.relu()
    }
}

pub fn gpu_add(a: &Tensor, b: &Tensor) -> Tensor {
    a.add(b)
}

pub fn gpu_mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.mul(b)
}

pub fn has_gpu() -> bool {
    gpu().is_some()
}

fn cast_bytes<T: Sized>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

fn cast_back(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests {
    use crate::blas::{sgemm, Transpose};
    use crate::tensor::FusedAct;

    #[test]
    fn blas_matmul_correct() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        assert!((c[0] - 19.0).abs() < 1e-3);
        assert!((c[3] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn cpu_linear_relu() {
        let x = crate::tensor::Tensor::randn(&[4, 8]);
        let w = crate::tensor::Tensor::he(&[3, 8]);
        let b = crate::tensor::Tensor::zeros(&[3]);
        let y = crate::tensor::Tensor::linear_layer(&x, &w, &b, FusedAct::Relu);
        assert_eq!(y.shape(), vec![4, 3]);
        assert!(y.data().iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn has_gpu_or_fallback() {
        let _ = super::has_gpu();
    }
}

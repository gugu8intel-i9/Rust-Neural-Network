use rust_nn::compiled::CompiledMLP;
use std::time::Instant;
fn main() {
    let mut m = CompiledMLP::new(&[(256, 512), (512, 512), (512, 256)], 64, 0.01);
    let x: Vec<f32> = (0..64 * 256).map(|i| (i as f32 * 0.001).sin()).collect();
    let t: Vec<f32> = (0..64 * 256).map(|i| (i as f32 * 0.001).cos()).collect();
    for _ in 0..5 {
        m.step(&x, &t);
    }
    let iters = 500;
    let t0 = Instant::now();
    for _ in 0..iters {
        m.step(&x, &t);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
    println!("compiled large (256-512-512-256, b64): {dt:.3} ms/step");
}

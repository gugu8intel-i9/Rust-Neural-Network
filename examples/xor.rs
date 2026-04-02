use rust_nn::tensor::Tensor;
use rust_nn::nn::{Sequential, Linear, ReLU, Module};
use rust_nn::optim::{Optimizer, Adam};
use rust_nn::loss::{MSELoss, Loss};
use rust_nn::train::{SimpleDataLoader, Trainer};
use std::sync::Arc;

fn main() {
    // 1. Create data (XOR)
    // XOR inputs: [0,0], [0,1], [1,0], [1,1]
    let inputs = Tensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        vec![4, 2]
    );
    // XOR targets: [0], [1], [1], [0]
    let targets = Tensor::from_vec(
        vec![0.0, 1.0, 1.0, 0.0],
        vec![4, 1]
    );

    // 2. Define model
    let model = Arc::new(Sequential::new()
        .add(Linear::new(2, 8, true))
        .add(ReLU)
        .add(Linear::new(8, 1, true)));

    // 3. Setup optimizer and loss
    let params = model.parameters();
    let optimizer = Adam::new(params, 0.01);
    let loss_fn = MSELoss;

    // 4. Training loop
    let mut trainer = Trainer::new(model.clone(), optimizer, loss_fn);
    
    println!("Training XOR model...");
    for epoch in 0..200 {
        let loader = SimpleDataLoader::new(inputs.clone(), targets.clone(), 4);
        let loss = trainer.train_epoch(loader);
        if (epoch + 1) % 50 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch + 1, loss);
        }
    }

    // 5. Test
    println!("
Predictions:");
    let outputs = model.forward(&inputs);
    let out_data = outputs.data();
    println!("0 ^ 0 = {:.4}", out_data[[0, 0]]);
    println!("0 ^ 1 = {:.4}", out_data[[1, 0]]);
    println!("1 ^ 0 = {:.4}", out_data[[2, 0]]);
    println!("1 ^ 1 = {:.4}", out_data[[3, 0]]);
}

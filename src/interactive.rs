//! Interactive REPL: an exploratory environment for building, training, and inspecting models.
//!
//! # Design
//!
//! A readline-style REPL that supports interactive model construction, dataset loading, training,
//! and evaluation. Designed for the "hands-on" workflow — build a model layer by layer, load a
//! dataset, train it, and watch the loss curve in real time.
//!
//! # Commands
//!
//! ```text
//! rust-nn> help                          Show available commands
//! rust-nn> tensor randn 3 4              Create a random tensor
//! rust-nn> tensor zeros 2 2              Create a zeros tensor
//! rust-nn> model new                     Start building a new Sequential model
//! rust-nn> model add linear 784 256      Add a Linear layer
//! rust-nn> model add relu                Add a ReLU activation
//! rust-nn> model summary                 Print the model architecture
//! rust-nn> data csv path/to/file.csv     Load a CSV dataset
//! rust-nn> data hf dataset/split 1000    Load from HuggingFace Hub (1000 rows)
//! rust-nn> data synthetic 100 4 3        Generate synthetic classification data
//! rust-nn> train 10 0.01                 Train for 10 epochs at lr=0.01
//! rust-nn> predict 0                     Run inference on row 0
//! rust-nn> exit                          Quit
//! ```
//!
//! # Live training visualization
//!
//! During training, a real-time ASCII loss curve is printed after each epoch, showing the loss
//! trajectory with a sparkline.

use crate::data::{self, Dataset};
use crate::loss::{Loss, MSELoss};
use crate::nn::{Linear, Module, ReLU, Sequential};
use crate::optim::{Adam, Optimizer};
use crate::tensor::Tensor;
use std::io::{self, BufRead, Write};

/// State of the interactive session.
pub struct Session {
    pub model: Option<Sequential>,
    pub model_input_dim: Option<usize>,
    pub model_output_dim: Option<usize>,
    pub dataset: Option<Dataset>,
    pub feature_columns: Vec<String>,
    pub target_column: Option<String>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Session {
            model: None,
            model_input_dim: None,
            model_output_dim: None,
            dataset: None,
            feature_columns: Vec::new(),
            target_column: None,
        }
    }
}

/// Run the interactive REPL loop. Reads from stdin.
pub fn run_repl() {
    let mut session = Session::new();
    print_banner();

    let stdin = io::stdin();
    loop {
        print!("rust-nn> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0] {
            "help" | "?" | "h" => print_help(),
            "exit" | "quit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "tensor" => handle_tensor(&parts[1..]),
            "model" => handle_model(&parts[1..], &mut session),
            "data" => handle_data(&parts[1..], &mut session),
            "train" => handle_train(&parts[1..], &mut session),
            "predict" => handle_predict(&parts[1..], &mut session),
            "info" => handle_info(&session),
            _ => {
                println!("Unknown command: '{}'. Type 'help' for available commands.", parts[0]);
            }
        }
    }
}

fn print_banner() {
    println!(r"
  ╔═══════════════════════════════════════════╗
  ║   rust-nn Interactive Session  v0.11.0   ║
  ║   Neural Networks in Pure Rust            ║
  ╚═══════════════════════════════════════════╝
  Type 'help' for commands. 'exit' to quit.
");
}

fn print_help() {
    println!(r"
  Commands:
    tensor randn <dims...>        Create a random tensor and print it
    tensor zeros <dims...>        Create a zeros tensor and print it

    model new                     Start a new Sequential model
    model add linear <in> <out>   Add a Linear(in, out) layer with bias
    model add relu                Add a ReLU activation
    model add sigmoid             Add a Sigmoid activation
    model add tanh                Add a Tanh activation
    model summary                 Print model architecture and param count

    data csv <path>               Load a CSV file
    data tsv <path>               Load a TSV file
    data jsonl <path>             Load a JSONL file
    data hf <dataset> <split> <n> Load n rows from HuggingFace Hub
    data synthetic <n> <f> <c>    Generate synthetic classification data (n samples, f features, c classes)
    data regression <n> <f>       Generate synthetic regression data
    data columns                  List dataset columns
    data head <n>                 Show first n rows
    data split <ratio>            Set train/test split ratio
    data target <column>          Set the target/label column
    data features <col1> <col2>   Set feature columns (space-separated)

    train <epochs> <lr>           Train the model on the loaded dataset
    predict <row>                 Run inference on a specific row
    info                          Show current session state
    exit                          Quit
");
}

fn handle_tensor(args: &[&str]) {
    if args.is_empty() {
        println!("Usage: tensor randn|zeros <dims...>");
        return;
    }
    let dims: Vec<usize> = args[1..]
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    if dims.is_empty() {
        println!("Error: provide at least one dimension");
        return;
    }
    match args[0] {
        "randn" => {
            let t = Tensor::randn(&dims);
            println!("{t}");
        }
        "zeros" => {
            let t = Tensor::zeros(&dims);
            println!("{t}");
        }
        "ones" => {
            let t = Tensor::ones(&dims);
            println!("{t}");
        }
        _ => println!("Unknown tensor command: {}", args[0]),
    }
}

fn handle_model(args: &[&str], session: &mut Session) {
    if args.is_empty() {
        println!("Usage: model new|add|summary");
        return;
    }
    match args[0] {
        "new" => {
            session.model = Some(Sequential::new());
            session.model_input_dim = None;
            session.model_output_dim = None;
            println!("Started a new Sequential model.");
        }
        "add" => {
            if session.model.is_none() {
                println!("No model. Run 'model new' first.");
                return;
            }
            if args.len() < 2 {
                println!("Usage: model add linear|relu|sigmoid|tanh ...");
                return;
            }
            let model = session.model.as_mut().unwrap();
            match args[1] {
                "linear" => {
                    if args.len() < 4 {
                        println!("Usage: model add linear <in_features> <out_features>");
                        return;
                    }
                    let in_f: usize = args[2].parse().unwrap_or(0);
                    let out_f: usize = args[3].parse().unwrap_or(0);
                    if in_f == 0 || out_f == 0 {
                        println!("Error: invalid dimensions");
                        return;
                    }
                    if session.model_input_dim.is_none() {
                        session.model_input_dim = Some(in_f);
                    }
                    session.model_output_dim = Some(out_f);
                    *model = std::mem::take(model).add(Linear::new(in_f, out_f, true));
                    println!("Added Linear({in_f} -> {out_f})");
                }
                "relu" => {
                    *model = std::mem::take(model).add(ReLU);
                    println!("Added ReLU");
                }
                "sigmoid" => {
                    *model = std::mem::take(model).add(crate::nn::Sigmoid);
                    println!("Added Sigmoid");
                }
                "tanh" => {
                    *model = std::mem::take(model).add(crate::nn::Tanh);
                    println!("Added Tanh");
                }
                _ => println!("Unknown layer: {}", args[1]),
            }
        }
        "summary" => {
            if let Some(ref model) = session.model {
                let params = model.parameters();
                let total: usize = params.iter().map(|t| t.len()).sum();
                println!("\n  Model: Sequential");
                println!("  Parameters: {} tensors, {} total elements", params.len(), total);
                if let (Some(inp), Some(out)) = (session.model_input_dim, session.model_output_dim) {
                    println!("  Input dim: {inp}, Output dim: {out}");
                }
            } else {
                println!("No model loaded.");
            }
        }
        _ => println!("Unknown model command: {}", args[0]),
    }
}

fn handle_data(args: &[&str], session: &mut Session) {
    if args.is_empty() {
        println!("Usage: data csv|tsv|jsonl|hf|synthetic|regression|columns|head|features|target");
        return;
    }
    match args[0] {
        "csv" if args.len() >= 2 => {
            match data::load_csv(args[1]) {
                Ok(ds) => {
                    println!("{}", ds.summary());
                    session.dataset = Some(ds);
                }
                Err(e) => println!("Error: {e}"),
            }
        }
        "tsv" if args.len() >= 2 => {
            match data::load_tsv(args[1]) {
                Ok(ds) => {
                    println!("{}", ds.summary());
                    session.dataset = Some(ds);
                }
                Err(e) => println!("Error: {e}"),
            }
        }
        "jsonl" if args.len() >= 2 => {
            match data::load_jsonl(args[1]) {
                Ok(ds) => {
                    println!("{}", ds.summary());
                    session.dataset = Some(ds);
                }
                Err(e) => println!("Error: {e}"),
            }
        }
        "hf" if args.len() >= 4 => {
            let n: usize = args[3].parse().unwrap_or(100);
            println!("Loading from HuggingFace: {}/{} ({} rows)...", args[1], args[2], n);
            match data::load_huggingface(args[1], args[2], n, None) {
                Ok(ds) => {
                    println!("{}", ds.summary());
                    session.dataset = Some(ds);
                }
                Err(e) => println!("Error: {e}"),
            }
        }
        "synthetic" if args.len() >= 4 => {
            let n: usize = args[1].parse().unwrap_or(100);
            let f: usize = args[2].parse().unwrap_or(4);
            let c: usize = args[3].parse().unwrap_or(3);
            let ds = data::make_classification(n, f, c);
            println!("{}", ds.summary());
            session.dataset = Some(ds);
            // Auto-set features and target.
            session.feature_columns = (0..f).map(|j| format!("f{j}")).collect();
            session.target_column = Some("label".into());
        }
        "regression" if args.len() >= 3 => {
            let n: usize = args[1].parse().unwrap_or(100);
            let f: usize = args[2].parse().unwrap_or(4);
            let ds = data::make_regression(n, f);
            println!("{}", ds.summary());
            session.dataset = Some(ds);
            session.feature_columns = (0..f).map(|j| format!("f{j}")).collect();
            session.target_column = Some("target".into());
        }
        "columns" => {
            if let Some(ref ds) = session.dataset {
                let mut names: Vec<&String> = ds.columns.keys().collect();
                names.sort();
                for name in names {
                    let col = &ds.columns[name];
                    let dtype = if col.is_numeric() { "f32" } else { "str" };
                    let mark = if session.feature_columns.contains(name) { " [feature]" }
                        else if session.target_column.as_deref() == Some(name.as_str()) { " [target]" }
                        else { "" };
                    println!("  {name}: {dtype}{}", mark);
                }
            } else {
                println!("No dataset loaded.");
            }
        }
        "head" => {
            if let Some(ref ds) = session.dataset {
                let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
                let head = ds.head(n);
                println!("{}", head.summary());
            } else {
                println!("No dataset loaded.");
            }
        }
        "features" => {
            session.feature_columns = args[1..].iter().map(|s| s.to_string()).collect();
            println!("Feature columns: {:?}", session.feature_columns);
        }
        "target" if args.len() >= 2 => {
            session.target_column = Some(args[1].to_string());
            println!("Target column: {}", args[1]);
        }
        _ => println!("Unknown data command or missing arguments. Type 'help'."),
    }
}

fn handle_train(args: &[&str], session: &mut Session) {
    let epochs: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
    let lr: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.01);

    let (Some(ref model), Some(ref ds)) = (&session.model, &session.dataset) else {
        println!("Need both a model and a dataset. Use 'model new' and 'data ...' first.");
        return;
    };
    if session.feature_columns.is_empty() || session.target_column.is_none() {
        println!("Set feature and target columns first. Use 'data features ...' and 'data target ...'.");
        return;
    }

    let feat_names: Vec<&str> = session.feature_columns.iter().map(|s| s.as_str()).collect();
    let target_name = session.target_column.as_ref().unwrap();
    let inputs = ds.to_tensor(&feat_names);
    let targets = ds.to_tensor(&[target_name.as_str()]);

    let params = model.parameters();
    let mut opt = Adam::new(params, lr);
    let loss_fn = MSELoss;

    println!("\n  Training: {} epochs, lr={lr}", epochs);
    println!("  {}", "-".repeat(50));

    let mut losses: Vec<f64> = Vec::new();
    for epoch in 0..epochs {
        opt.zero_grad();
        let out = model.forward(&inputs);
        let loss = loss_fn.forward(&out, &targets);
        let loss_val = loss.data().iter().copied().next().unwrap_or(0.0) / inputs.len() as f32;
        loss.backward();
        opt.step();
        losses.push(loss_val as f64);

        let bar = sparkline(&losses);
        println!("  Epoch {:>3}: loss = {:>10.4} {}", epoch + 1, loss_val, bar);
    }
    println!("  {}", "-".repeat(50));
    println!("  Training complete. Final loss: {:.4}\n", losses.last().copied().unwrap_or(0.0));
}

fn handle_predict(args: &[&str], session: &mut Session) {
    let row: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let (Some(ref model), Some(ref ds)) = (&session.model, &session.dataset) else {
        println!("Need both a model and a dataset.");
        return;
    };
    let feat_names: Vec<&str> = session.feature_columns.iter().map(|s| s.as_str()).collect();
    let row_ds = ds.head(row + 1);
    let inputs = row_ds.to_tensor(&feat_names);
    let out = model.forward(&inputs);
    let row_out: Vec<f32> = out.data().iter().copied().skip(row * out.shape()[out.ndim() - 1]).take(out.shape()[out.ndim()-1]).collect();
    println!("  Prediction for row {row}: {:?}", row_out);
}

fn handle_info(session: &Session) {
    println!("\n  Session state:");
    println!("    Model: {}", if session.model.is_some() { "loaded" } else { "none" });
    println!("    Dataset: {}", if session.dataset.is_some() { "loaded" } else { "none" });
    if !session.feature_columns.is_empty() {
        println!("    Features: {:?}", session.feature_columns);
    }
    if let Some(ref t) = session.target_column {
        println!("    Target: {t}");
    }
    println!();
}

/// Render an ASCII sparkline of the loss curve.
fn sparkline(values: &[f64]) -> String {
    if values.is_empty() {
        return String::new();
    }
    let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-12 {
        return blocks[7].to_string().repeat(values.len());
    }
    values
        .iter()
        .map(|&v| {
            let normalized = (v - min) / range;
            let idx = (normalized * 7.0).round() as usize;
            blocks[idx.min(7)]
        })
        .collect()
}

/// Process a single command string (for programmatic use or testing).
pub fn process_command(session: &mut Session, command: &str) -> String {
    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        return String::new();
    }
    // Capture output by redirecting to a string.
    let mut output = String::new();
    match parts[0] {
        "tensor" => {
            if parts.len() >= 3 && parts[1] == "randn" {
                let dims: Vec<usize> = parts[2..].iter().filter_map(|s| s.parse().ok()).collect();
                let t = Tensor::randn(&dims);
                output = format!("{t}");
            }
        }
        "model" => {
            if parts.len() >= 2 && parts[1] == "new" {
                session.model = Some(Sequential::new());
                output = "New model created".into();
            } else if parts.len() >= 4 && parts[1] == "add" && parts[2] == "linear"
                && session.model.is_some() {
                    let in_f: usize = parts[3].parse().unwrap_or(0);
                    let out_f: usize = parts[4].parse().unwrap_or(0);
                    let model = session.model.as_mut().unwrap();
                    if session.model_input_dim.is_none() {
                        session.model_input_dim = Some(in_f);
                    }
                    session.model_output_dim = Some(out_f);
                    *model = std::mem::take(model).add(Linear::new(in_f, out_f, true));
                    output = format!("Added Linear({in_f} -> {out_f})");
                }
        }
        "data" => {
            if parts.len() >= 4 && parts[1] == "synthetic" {
                let n: usize = parts[2].parse().unwrap_or(100);
                let f: usize = parts[3].parse().unwrap_or(4);
                let c: usize = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
                let ds = data::make_classification(n, f, c);
                output = ds.summary();
                session.dataset = Some(ds);
                session.feature_columns = (0..f).map(|j| format!("f{j}")).collect();
                session.target_column = Some("label".into());
            }
        }
        "info" => {
            output = format!(
                "model: {}, dataset: {}",
                session.model.is_some(),
                session.dataset.is_some()
            );
        }
        _ => {}
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_rendering() {
        let vals = vec![10.0, 8.0, 5.0, 3.0, 1.0];
        let s = sparkline(&vals);
        assert!(!s.is_empty());
        // Highest value should map to '█' (index 7).
        assert!(s.contains('█'));
    }

    #[test]
    fn sparkline_constant() {
        let s = sparkline(&[5.0, 5.0, 5.0]);
        assert!(s.chars().all(|c| c == '█'));
    }

    #[test]
    fn process_command_model_build() {
        let mut session = Session::new();
        let out = process_command(&mut session, "model new");
        assert!(out.contains("New model"));
        let out = process_command(&mut session, "model add linear 4 2");
        assert!(out.contains("Linear(4 -> 2)"));
        assert!(session.model.is_some());
        assert_eq!(session.model_input_dim, Some(4));
        assert_eq!(session.model_output_dim, Some(2));
    }

    #[test]
    fn process_command_synthetic_data() {
        let mut session = Session::new();
        let out = process_command(&mut session, "data synthetic 50 3 2");
        assert!(out.contains("synthetic"));
        assert!(session.dataset.is_some());
        assert_eq!(session.feature_columns.len(), 3);
        assert_eq!(session.target_column.as_deref(), Some("label"));
    }

    #[test]
    fn process_command_info() {
        let mut session = Session::new();
        session.model = Some(Sequential::new());
        let out = process_command(&mut session, "info");
        assert!(out.contains("model: true"));
    }
}

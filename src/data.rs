//! Multi-source dataset loading: HuggingFace Hub, CSV/TSV/JSON/JSONL, and Kaggle.
//!
//! # Design
//!
//! - **Unified `Dataset` abstraction**: columnar storage with named columns, supporting both
//!   numeric features (for direct tensor conversion) and text features (for tokenization).
//! - **HuggingFace Hub loading** via the Datasets Server REST API (`/rows` endpoint). Streams
//!   rows in batches of 100 without downloading the entire dataset — **lazy loading** for large
//!   datasets. Caches responses locally to avoid redundant network requests.
//! - **CSV/TSV loading**: a dependency-free parser (no `csv` crate) that handles quoted fields,
//!   headers, and type inference.
//! - **JSON/JSONL loading**: a dependency-free parser for line-delimited or array JSON.
//! - **Kaggle loading**: downloads datasets via the Kaggle API (requires credentials).
//! - **Tensor conversion**: `to_tensor()` converts numeric columns into a `Tensor` for training.
//!
//! # Innovative features
//!
//! - **Lazy streaming from HuggingFace**: fetches only the rows you need, not the whole dataset.
//! - **Automatic type inference**: detects whether columns are numeric or text.
//! - **Local caching**: downloaded rows are cached in `~/.cache/rust-nn/datasets/` to avoid
//!   repeated network round-trips.
//! - **Zero heavy dependencies**: CSV/JSON parsing is built-in; only `ureq` for HTTP.

use crate::tensor::Tensor;
use crate::tokenizer::BpeTokenizer;
use std::collections::HashMap;
use std::path::PathBuf;

/// A column in a dataset — either numeric (f32) or text (String).
#[derive(Debug, Clone)]
pub enum Column {
    Float(Vec<f32>),
    Text(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Float(v) => v.len(),
            Column::Text(v) => v.len(),
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Column::Float(_))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A tabular dataset with named columns. Each column is either numeric or text.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub columns: HashMap<String, Column>,
    pub num_rows: usize,
    pub source: String,
}

impl Dataset {
    /// Create an empty dataset.
    pub fn new(source: impl Into<String>) -> Self {
        Dataset {
            columns: HashMap::new(),
            num_rows: 0,
            source: source.into(),
        }
    }

    /// Add a numeric column.
    pub fn add_float_column(&mut self, name: &str, data: Vec<f32>) {
        self.num_rows = self.num_rows.max(data.len());
        self.columns.insert(name.to_string(), Column::Float(data));
    }

    /// Add a text column.
    pub fn add_text_column(&mut self, name: &str, data: Vec<String>) {
        self.num_rows = self.num_rows.max(data.len());
        self.columns.insert(name.to_string(), Column::Text(data));
    }

    /// Get a numeric column as a slice.
    pub fn get_float(&self, name: &str) -> Option<&[f32]> {
        match self.columns.get(name) {
            Some(Column::Float(v)) => Some(v),
            _ => None,
        }
    }

    /// Get a text column as a slice.
    pub fn get_text(&self, name: &str) -> Option<&[String]> {
        match self.columns.get(name) {
            Some(Column::Text(v)) => Some(v),
            _ => None,
        }
    }

    /// Convert numeric columns into a `[num_rows, num_features]` Tensor for training.
    /// Only includes columns that are numeric. Column order follows `column_names`.
    pub fn to_tensor(&self, column_names: &[&str]) -> Tensor {
        let ncols = column_names.len();
        let nrows = self.num_rows;
        let mut data = Vec::with_capacity(nrows * ncols);
        for row in 0..nrows {
            for col_name in column_names {
                let val = self.get_float(col_name).map(|c| c.get(row).copied().unwrap_or(0.0)).unwrap_or(0.0);
                data.push(val);
            }
        }
        Tensor::from_vec(data, vec![nrows, ncols])
    }

    /// Convert text columns into token-id tensors using a tokenizer.
    pub fn to_token_tensor(&self, column_name: &str, tokenizer: &BpeTokenizer, max_len: usize) -> Tensor {
        let text = self.get_text(column_name).unwrap_or(&[]);
        let nrows = text.len();
        let mut data = Vec::with_capacity(nrows * max_len);
        for line in text {
            let ids = tokenizer.encode(line);
            for i in 0..max_len {
                data.push(ids.get(i).copied().unwrap_or(0) as f32);
            }
        }
        Tensor::from_vec(data, vec![nrows, max_len])
    }

    /// Summary of the dataset for display.
    pub fn summary(&self) -> String {
        let mut cols: Vec<&String> = self.columns.keys().collect();
        cols.sort();
        let col_info: Vec<String> = cols
            .iter()
            .map(|name| {
                let col = &self.columns[*name];
                let dtype = if col.is_numeric() { "f32" } else { "str" };
                format!("  {name}: {dtype}[{}]", col.len())
            })
            .collect();
        format!(
            "Dataset (source: {})\n  rows: {}\n  columns ({}):\n{}",
            self.source,
            self.num_rows,
            cols.len(),
            col_info.join("\n")
        )
    }

    /// Take the first `n` rows as a new dataset (head).
    pub fn head(&self, n: usize) -> Dataset {
        let mut sub = Dataset::new(self.source.clone());
        sub.num_rows = n.min(self.num_rows);
        for (name, col) in &self.columns {
            match col {
                Column::Float(v) => sub.add_float_column(name, v[..sub.num_rows].to_vec()),
                Column::Text(v) => sub.add_text_column(name, v[..sub.num_rows].to_vec()),
            }
        }
        sub
    }

    /// Split into train/test by ratio (e.g. 0.8 = 80% train, 20% test).
    pub fn train_test_split(&self, ratio: f32) -> (Dataset, Dataset) {
        let split = (self.num_rows as f32 * ratio) as usize;
        let mut train = Dataset::new(format!("{} (train)", self.source));
        let mut test = Dataset::new(format!("{} (test)", self.source));
        train.num_rows = split;
        test.num_rows = self.num_rows - split;
        for (name, col) in &self.columns {
            match col {
                Column::Float(v) => {
                    train.add_float_column(name, v[..split].to_vec());
                    test.add_float_column(name, v[split..].to_vec());
                }
                Column::Text(v) => {
                    train.add_text_column(name, v[..split].to_vec());
                    test.add_text_column(name, v[split..].to_vec());
                }
            }
        }
        (train, test)
    }
}

// ==================== CSV / TSV loading ====================

/// Load a CSV or TSV file into a Dataset. The delimiter is auto-detected (',' for CSV, '\t' for TSV).
/// The first line is treated as a header. Numeric columns are inferred; non-numeric are text.
pub fn load_csv(path: &str) -> Result<Dataset, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    parse_delimited(&content, ',')
}

/// Load a TSV file.
pub fn load_tsv(path: &str) -> Result<Dataset, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    parse_delimited(&content, '\t')
}

/// Parse delimited text into a Dataset with automatic type inference.
fn parse_delimited(content: &str, delimiter: char) -> Result<Dataset, String> {
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return Err("empty file".into());
    }

    let headers: Vec<String> = parse_csv_line(lines[0], delimiter);
    let mut columns: Vec<Vec<String>> = vec![Vec::new(); headers.len()];

    for line in &lines[1..] {
        let fields = parse_csv_line(line, delimiter);
        for (i, field) in fields.iter().enumerate() {
            if i < columns.len() {
                columns[i].push(field.clone());
            }
        }
    }

    let mut dataset = Dataset::new("csv");
    for (i, header) in headers.iter().enumerate() {
        let raw = &columns[i];
        // Try to parse as f32 — if all values parse, it's numeric.
        let floats: Option<Vec<f32>> = raw.iter().map(|s| s.trim().parse::<f32>().ok()).collect();
        if let Some(floats) = floats {
            dataset.add_float_column(header, floats);
        } else {
            dataset.add_text_column(header, raw.clone());
        }
    }
    Ok(dataset)
}

/// Parse a single CSV line, handling quoted fields with embedded commas.
fn parse_csv_line(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' if in_quotes => in_quotes = false,
            '"' => in_quotes = true,
            c if c == delimiter && !in_quotes => {
                fields.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    fields.push(current);
    fields
}

// ==================== JSON / JSONL loading ====================

/// Load a JSONL file (one JSON object per line) into a Dataset.
pub fn load_jsonl(path: &str) -> Result<Dataset, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    let mut dataset = Dataset::new("jsonl");
    let mut col_data: HashMap<String, Vec<String>> = HashMap::new();

    for line in content.lines().filter(|l| !l.trim().is_empty()) {
        let pairs = parse_simple_json_object(line);
        for (key, value) in pairs {
            col_data.entry(key).or_default().push(value);
        }
    }

    for (key, values) in col_data {
        let floats: Option<Vec<f32>> = values.iter().map(|s| s.trim().parse::<f32>().ok()).collect();
        if let Some(floats) = floats {
            dataset.add_float_column(&key, floats);
        } else {
            dataset.add_text_column(&key, values);
        }
    }
    Ok(dataset)
}

/// A minimal JSON object parser that extracts key-value string pairs.
/// Handles flat objects with string keys and scalar values (numbers or strings).
fn parse_simple_json_object(line: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let trimmed = line.trim();
    let inner = trimmed.strip_prefix('{').and_then(|s| s.strip_suffix('}'));
    let Some(inner) = inner else { return pairs };

    // Split by commas that are NOT inside quoted strings.
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_quotes => {
                current.push(ch);
                escaped = true;
            }
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        fields.push(current);
    }

    for field in fields {
        let field = field.trim();
        if let Some(colon_pos) = field.find(':') {
            let key = field[..colon_pos].trim().trim_matches('"').to_string();
            let value = field[colon_pos + 1..].trim();
            let value = value.trim_matches('"').to_string();
            pairs.push((key, value));
        }
    }
    pairs
}

// ==================== HuggingFace Hub loading ====================

const HF_BASE: &str = "https://datasets-server.huggingface.co";

/// Cache directory for downloaded dataset rows.
fn cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(format!("{home}/.cache/rust-nn/datasets"))
}

/// Load a dataset from the HuggingFace Hub via the Datasets Server `/rows` endpoint.
///
/// Streams rows in batches of 100 (the API max per request). Numeric fields are stored as `f32`,
/// text fields as `String`. Optionally provide a HuggingFace token for gated/private datasets.
///
/// # Example
/// ```ignore
/// let ds = load_huggingface("cornell-movie-review-data/rotten_tomatoes", "train", 1000, None)?;
/// println!("{}", ds.summary());
/// ```
pub fn load_huggingface(
    dataset: &str,
    split: &str,
    max_rows: usize,
    token: Option<&str>,
) -> Result<Dataset, String> {
    // Check cache first.
    let cache_key = format!("{}_{}_{}", dataset.replace('/', "_"), split, max_rows);
    let cache_path = cache_dir().join(format!("{cache_key}.json"));

    if cache_path.exists() {
        if let Ok(cached) = std::fs::read_to_string(&cache_path) {
            if let Ok(ds) = dataset_from_cache_json(&cached, dataset) {
                return Ok(ds);
            }
        }
    }

    let mut all_rows: Vec<HashMap<String, String>> = Vec::new();
    let mut offset = 0usize;
    let batch = 100usize.min(max_rows);
    let agent = ureq::Agent::new();

    while offset < max_rows {
        let length = batch.min(max_rows - offset);
        let url = format!(
            "{HF_BASE}/rows?dataset={dataset}&config=default&split={split}&offset={offset}&length={length}"
        );

        let req = agent.get(&url);
        let req = if let Some(tok) = token {
            req.set("Authorization", &format!("Bearer {tok}"))
        } else {
            req
        };

        let response = req
            .call()
            .map_err(|e| format!("HuggingFace API request failed: {e}"))?;

        let body: String = response
            .into_string()
            .map_err(|e| format!("Failed to read response: {e}"))?;

        let rows = extract_rows_from_hf_json(&body);
        if rows.is_empty() {
            break;
        }
        let fetched = rows.len();
        all_rows.extend(rows);
        offset += fetched;
        if fetched < length {
            break; // reached end of dataset
        }
    }

    // Build the Dataset from collected rows.
    let mut dataset_obj = Dataset::new(format!("huggingface:{dataset}"));
    let mut col_names: Vec<String> = Vec::new();
    if let Some(first) = all_rows.first() {
        col_names = first.keys().cloned().collect();
        col_names.sort();
    }

    for col_name in &col_names {
        let values: Vec<String> = all_rows
            .iter()
            .map(|row| row.get(col_name).cloned().unwrap_or_default())
            .collect();
        let floats: Option<Vec<f32>> = values.iter().map(|s| s.parse::<f32>().ok()).collect();
        if let Some(floats) = floats {
            dataset_obj.add_float_column(col_name, floats);
        } else {
            dataset_obj.add_text_column(col_name, values);
        }
    }

    // Cache the result.
    let _ = std::fs::create_dir_all(cache_dir());
    let _ = std::fs::write(&cache_path, dataset_to_cache_json(&dataset_obj));

    Ok(dataset_obj)
}

/// Extract rows from the HuggingFace /rows JSON response.
fn extract_rows_from_hf_json(body: &str) -> Vec<HashMap<String, String>> {
    // The JSON has a "rows" array where each element has a "row" object.
    // We parse this with a lightweight approach: find "rows" and extract key-value pairs.
    let mut result = Vec::new();

    // Find the "rows" array.
    let rows_marker = "\"rows\"";
    if let Some(rows_start) = body.find(rows_marker) {
        // Find the opening bracket of the array.
        let rest = &body[rows_start..];
        if let Some(arr_start) = rest.find('[') {
            let arr_body = &rest[arr_start + 1..];
            // Split by "row" markers.
            let row_marker = "\"row\"";
            let mut pos = 0;
            while let Some(rel) = arr_body[pos..].find(row_marker) {
                let abs = pos + rel;
                let after_marker = &arr_body[abs + row_marker.len()..];
                // Find the opening brace.
                if let Some(brace) = after_marker.find('{') {
                    let obj_start = brace;
                    // Find matching closing brace.
                    let mut depth = 0;
                    let mut end = obj_start;
                    for (i, ch) in after_marker[obj_start..].char_indices() {
                        match ch {
                            '{' => depth += 1,
                            '}' => {
                                depth -= 1;
                                if depth == 0 {
                                    end = obj_start + i;
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    let obj_str = &after_marker[obj_start + 1..end];
                    let pairs = parse_simple_json_object(&format!("{{{obj_str}}}"));
                    let row: HashMap<String, String> = pairs.into_iter().collect();
                    result.push(row);
                    pos = abs + row_marker.len() + end + 1;
                } else {
                    break;
                }
            }
        }
    }
    result
}

fn dataset_to_cache_json(ds: &Dataset) -> String {
    let mut s = String::new();
    s.push_str(&format!("{{\"source\":\"{}\",\"num_rows\":{},\"columns\":{{", ds.source, ds.num_rows));
    let mut first = true;
    let mut names: Vec<&String> = ds.columns.keys().collect();
    names.sort();
    for name in names {
        if !first { s.push(','); }
        first = false;
        s.push_str(&format!("\"{name}\":{{"));
        match &ds.columns[name] {
            Column::Float(v) => {
                s.push_str("\"type\":\"float\",\"values\":[");
                for (i, val) in v.iter().enumerate() {
                    if i > 0 { s.push(','); }
                    s.push_str(&val.to_string());
                }
                s.push(']');
            }
            Column::Text(v) => {
                s.push_str("\"type\":\"text\",\"values\":[");
                for (i, val) in v.iter().enumerate() {
                    if i > 0 { s.push(','); }
                    s.push_str(&format!("\"{}\"", val.replace('"', "\\\"")));
                }
                s.push(']');
            }
        }
        s.push('}');
    }
    s.push_str("}}");
    s
}

fn dataset_from_cache_json(json: &str, source: &str) -> Result<Dataset, String> {
    let mut ds = Dataset::new(source);
    let pairs = parse_simple_json_object(json);
    for (key, value) in pairs {
        if key == "num_rows" {
            ds.num_rows = value.parse().unwrap_or(0);
        } else if key == "columns" {
            // Parse nested columns — just store the raw for now
            let _ = value;
        } else {
            // Each column value is like {"type":"float","values":[...]}
            if value.contains("\"type\":\"float\"") {
                let nums: Vec<f32> = extract_float_array(&value);
                ds.add_float_column(&key, nums);
            } else {
                ds.add_text_column(&key, vec![value]);
            }
        }
    }
    Ok(ds)
}

fn extract_float_array(s: &str) -> Vec<f32> {
    let start = s.find('[').map(|p| p + 1).unwrap_or(0);
    let end = s.rfind(']').unwrap_or(s.len());
    let inner = &s[start..end];
    inner.split(',').filter_map(|n| n.trim().parse().ok()).collect()
}

// ==================== Kaggle loading ====================

/// Load a dataset from Kaggle. Requires Kaggle API credentials (username + key).
///
/// Set credentials via environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`, or a
/// `~/.kaggle/kaggle.json` file.
///
/// # Example
/// ```ignore
/// let ds = load_kaggle("unanimad/dataisbeautiful", "data.csv")?;
/// ```
pub fn load_kaggle(dataset: &str, file: &str) -> Result<Dataset, String> {
    let username = std::env::var("KAGGLE_USERNAME")
        .map_err(|_| "KAGGLE_USERNAME not set. Get your API key from kaggle.com -> Account -> Create New Token".to_string())?;
    let key = std::env::var("KAGGLE_KEY")
        .map_err(|_| "KAGGLE_KEY not set".to_string())?;

    let url = format!("https://www.kaggle.com/api/v1/datasets/download/{dataset}");
    let agent = ureq::Agent::new();

    // Download the zip archive.
    let response = agent
        .get(&url)
        .set("Authorization", &format!("Basic {}", base64_encode(&format!("{username}:{key}"))))
        .call()
        .map_err(|e| format!("Kaggle download failed: {e}"))?;

    let zip_bytes = response
        .into_string()
        .map_err(|e| format!("Failed to read response: {e}"))?;

    // We expect a ZIP file; for simplicity, if the file is CSV, try to parse directly.
    // Full ZIP extraction would require a dependency; we document this limitation.
    if zip_bytes.starts_with("text/csv") || file.ends_with(".csv") {
        return parse_delimited(&zip_bytes, ',');
    }
    Err(format!(
        "Kaggle dataset downloaded but needs ZIP extraction for '{file}'. \
         Extract manually and use load_csv() on the extracted file."
    ))
}

/// Minimal base64 encoder (for Kaggle auth).
fn base64_encode(input: &str) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::new();
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((n >> 18) & 63) as usize] as char);
        result.push(CHARS[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((n >> 6) & 63) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(n & 63) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ==================== Synthetic dataset generation ====================

/// Generate a synthetic classification dataset (like sklearn's make_classification).
/// `n_samples` rows, `n_features` features, `n_classes` classes.
/// Features are class-dependent Gaussians; the "label" column holds the class index.
pub fn make_classification(n_samples: usize, n_features: usize, n_classes: usize) -> Dataset {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut ds = Dataset::new("synthetic");

    let mut features: Vec<Vec<f32>> = (0..n_features).map(|_| Vec::with_capacity(n_samples)).collect();
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        labels.push(class as f32);
        for (j, feat) in features.iter_mut().enumerate() {
            let mean = class as f32 * 0.8 + j as f32 * 0.05;
            let noise = (rng.gen::<f32>() - 0.5) * 0.6;
            feat.push(mean + noise);
        }
    }

    for (j, feat) in features.iter().enumerate() {
        ds.add_float_column(&format!("f{j}"), feat.clone());
    }
    ds.add_float_column("label", labels);
    ds
}

/// Generate a synthetic regression dataset (like sklearn's make_regression).
pub fn make_regression(n_samples: usize, n_features: usize) -> Dataset {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut ds = Dataset::new("synthetic");

    // Random true weights.
    let true_w: Vec<f32> = (0..n_features).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

    let mut features: Vec<Vec<f32>> = (0..n_features).map(|_| Vec::with_capacity(n_samples)).collect();
    let mut targets = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x: Vec<f32> = (0..n_features).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let y: f32 = x.iter().zip(true_w.iter()).map(|(xi, wi)| xi * wi).sum::<f32>() + (rng.gen::<f32>() - 0.5) * 0.1;
        targets.push(y);
        for (j, &xi) in x.iter().enumerate() {
            features[j].push(xi);
        }
    }

    for (j, feat) in features.iter().enumerate() {
        ds.add_float_column(&format!("f{j}"), feat.clone());
    }
    ds.add_float_column("target", targets);
    ds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_loading_and_type_inference() {
        let csv = "name,score,grade\nAlice,95.5,A\nBob,87.2,B\nCarol,92.0,A\n";
        let ds = parse_delimited(csv, ',').unwrap();
        assert_eq!(ds.num_rows, 3);
        assert!(ds.get_float("score").is_some(), "score should be numeric");
        assert!(ds.get_text("name").is_some(), "name should be text");
        assert!(ds.get_text("grade").is_some(), "grade should be text");
        let scores = ds.get_float("score").unwrap();
        assert!((scores[0] - 95.5).abs() < 1e-3);
    }

    #[test]
    fn csv_quoted_fields() {
        let line = r#""hello, world",42,"quoted,field""#;
        let fields = parse_csv_line(line, ',');
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "hello, world");
        assert_eq!(fields[1], "42");
        assert_eq!(fields[2], "quoted,field");
    }

    #[test]
    fn jsonl_loading() {
        let jsonl = r#"{"text":"hello","label":1}
{"text":"world","label":0}
{"text":"test","label":1}"#;
        // Write to temp file.
        let path = "/tmp/test_data.jsonl";
        std::fs::write(path, jsonl).unwrap();
        let ds = load_jsonl(path).unwrap();
        assert!(ds.get_text("text").is_some());
        assert!(ds.get_float("label").is_some());
        let labels = ds.get_float("label").unwrap();
        assert_eq!(labels, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn dataset_to_tensor() {
        let mut ds = Dataset::new("test");
        ds.add_float_column("a", vec![1.0, 2.0, 3.0]);
        ds.add_float_column("b", vec![4.0, 5.0, 6.0]);
        let t = ds.to_tensor(&["a", "b"]);
        assert_eq!(t.shape(), vec![3, 2]);
        let d: Vec<f32> = t.data().iter().copied().collect();
        assert_eq!(d, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn dataset_head() {
        let mut ds = Dataset::new("test");
        ds.add_float_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let head = ds.head(3);
        assert_eq!(head.num_rows, 3);
        assert_eq!(head.get_float("x").unwrap().len(), 3);
    }

    #[test]
    fn train_test_split() {
        let mut ds = Dataset::new("test");
        ds.add_float_column("x", (0..100).map(|i| i as f32).collect());
        let (train, test) = ds.train_test_split(0.8);
        assert_eq!(train.num_rows, 80);
        assert_eq!(test.num_rows, 20);
    }

    #[test]
    fn synthetic_classification() {
        let ds = make_classification(50, 4, 3);
        assert_eq!(ds.num_rows, 50);
        assert!(ds.get_float("f0").is_some());
        assert!(ds.get_float("label").is_some());
        let labels = ds.get_float("label").unwrap();
        assert!(labels.iter().all(|&l| (0.0..3.0).contains(&l)));
    }

    #[test]
    fn synthetic_regression() {
        let ds = make_regression(30, 3);
        assert_eq!(ds.num_rows, 30);
        assert!(ds.get_float("target").is_some());
    }

    #[test]
    fn dataset_summary() {
        let mut ds = Dataset::new("test");
        ds.add_float_column("x", vec![1.0, 2.0]);
        ds.add_text_column("name", vec!["a".into(), "b".into()]);
        let s = ds.summary();
        assert!(s.contains("test"));
        assert!(s.contains("f32"));
        assert!(s.contains("str"));
    }

    #[test]
    fn base64_encoding() {
        assert_eq!(base64_encode("user:pass"), "dXNlcjpwYXNz");
        assert_eq!(base64_encode("a"), "YQ==");
        assert_eq!(base64_encode("ab"), "YWI=");
        assert_eq!(base64_encode("abc"), "YWJj");
    }

    #[test]
    fn simple_json_parser() {
        let pairs = parse_simple_json_object(r#"{"text":"hello world","label":1}"#);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "text");
        assert_eq!(pairs[0].1, "hello world");
        assert_eq!(pairs[1].0, "label");
        assert_eq!(pairs[1].1, "1");
    }
}

// ==================== Credential management ====================

/// Manages authentication tokens for HuggingFace and Kaggle.
#[derive(Debug, Clone, Default)]
pub struct Credentials {
    /// HuggingFace access token (from https://huggingface.co/settings/tokens).
    pub hf_token: Option<String>,
    /// Kaggle username.
    pub kaggle_username: Option<String>,
    /// Kaggle API key (from https://www.kaggle.com -> Account -> Create New Token).
    pub kaggle_key: Option<String>,
}

impl Credentials {
    /// Create empty credentials.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load credentials from environment variables.
    /// Checks `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` for HuggingFace,
    /// `KAGGLE_USERNAME` and `KAGGLE_KEY` for Kaggle.
    pub fn from_env() -> Self {
        let hf_token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();
        let kaggle_username = std::env::var("KAGGLE_USERNAME").ok();
        let kaggle_key = std::env::var("KAGGLE_KEY").ok();
        Credentials { hf_token, kaggle_username, kaggle_key }
    }

    /// Set HuggingFace token.
    pub fn with_hf_token(mut self, token: impl Into<String>) -> Self {
        self.hf_token = Some(token.into());
        self
    }

    /// Set Kaggle credentials.
    pub fn with_kaggle(mut self, username: impl Into<String>, key: impl Into<String>) -> Self {
        self.kaggle_username = Some(username.into());
        self.kaggle_key = Some(key.into());
        self
    }

    /// Check if HuggingFace authentication is available.
    pub fn has_hf(&self) -> bool { self.hf_token.is_some() }

    /// Check if Kaggle authentication is available.
    pub fn has_kaggle(&self) -> bool { self.kaggle_username.is_some() && self.kaggle_key.is_some() }

    /// Save credentials to `~/.rust-nn/credentials.toml` for persistence across sessions.
    pub fn save(&self) -> std::io::Result<()> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let dir = format!("{home}/.rust-nn");
        std::fs::create_dir_all(&dir)?;
        let mut content = String::new();
        if let Some(ref t) = self.hf_token { content.push_str(&format!("[huggingface]\ntoken = \"{t}\"\n")); }
        if let (Some(u), Some(k)) = (&self.kaggle_username, &self.kaggle_key) {
            content.push_str(&format!("[kaggle]\nusername = \"{u}\"\nkey = \"{k}\"\n"));
        }
        std::fs::write(format!("{dir}/credentials.toml"), content)
    }

    /// Load credentials from `~/.rust-nn/credentials.toml`.
    pub fn load() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let path = format!("{home}/.rust-nn/credentials.toml");
        let mut creds = Self::from_env(); // env vars take priority
        if let Ok(content) = std::fs::read_to_string(&path) {
            let mut section = "";
            for line in content.lines() {
                let line = line.trim();
                if line.starts_with('[') && line.ends_with(']') {
                    section = &line[1..line.len()-1];
                } else if let Some(eq) = line.find('=') {
                    let key = line[..eq].trim();
                    let val = line[eq+1..].trim().trim_matches('"');
                    match (section, key) {
                        ("huggingface", "token") => { if creds.hf_token.is_none() { creds.hf_token = Some(val.into()); } }
                        ("kaggle", "username") => { if creds.kaggle_username.is_none() { creds.kaggle_username = Some(val.into()); } }
                        ("kaggle", "key") if creds.kaggle_key.is_none() => { creds.kaggle_key = Some(val.into()); }
                        _ => {}
                    }
                }
            }
        }
        creds
    }
}

// ==================== Dataset browsing / search ====================

/// A dataset listing entry (name, description, tags).
#[derive(Debug, Clone)]
pub struct DatasetListing {
    pub id: String,
    pub description: String,
    pub downloads: u64,
    pub tags: Vec<String>,
}

/// Search for datasets on the HuggingFace Hub.
///
/// Uses the HuggingFace Hub REST API to search for datasets by keyword.
/// Returns up to `limit` results.
pub fn search_huggingface(query: &str, limit: usize, creds: &Credentials) -> Result<Vec<DatasetListing>, String> {
    let agent = ureq::Agent::new();
    let url = format!("https://huggingface.co/api/datasets?search={query}&limit={limit}");
    let mut req = agent.get(&url);
    if let Some(ref token) = creds.hf_token {
        req = req.set("Authorization", &format!("Bearer {token}"));
    }
    let response = req.call().map_err(|e| format!("HF search failed: {e}"))?;
    let body = response.into_string().map_err(|e| format!("Read error: {e}"))?;

    // Parse the JSON array response (lightweight parser).
    let mut listings = Vec::new();
    // Find each object in the array
    let trimmed = body.trim();
    let inner = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(trimmed);
    for obj_str in split_json_objects(inner) {
        let pairs = parse_simple_json_object(&obj_str);
        let mut id = String::new();
        let mut downloads = 0u64;
        let mut desc = String::new();
        for (key, val) in &pairs {
            match key.as_str() {
                "id" => id = val.clone(),
                "downloads" => downloads = val.parse().unwrap_or(0),
                "description" => desc = val.clone(),
                _ => {}
            }
        }
        if !id.is_empty() {
            listings.push(DatasetListing {
                id,
                description: desc,
                downloads,
                tags: Vec::new(),
            });
        }
        if listings.len() >= limit { break; }
    }
    Ok(listings)
}

/// Search for datasets on Kaggle.
///
/// Uses the Kaggle API to list datasets matching a search query.
/// Requires Kaggle credentials.
pub fn search_kaggle(query: &str, limit: usize, creds: &Credentials) -> Result<Vec<DatasetListing>, String> {
    let username = creds.kaggle_username.as_ref().ok_or("Kaggle credentials required. Set KAGGLE_USERNAME and KAGGLE_KEY.")?;
    let key = creds.kaggle_key.as_ref().ok_or("Kaggle key required.")?;
    let agent = ureq::Agent::new();
    let url = format!("https://www.kaggle.com/api/v1/datasets/list?search={query}&page=1");
    let response = agent
        .get(&url)
        .set("Authorization", &format!("Basic {}", base64_encode(&format!("{username}:{key}"))))
        .call()
        .map_err(|e| format!("Kaggle search failed: {e}"))?;
    let body = response.into_string().map_err(|e| format!("Read error: {e}"))?;
    let mut listings = Vec::new();
    let inner = body.trim().strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(&body);
    for obj_str in split_json_objects(inner) {
        let pairs = parse_simple_json_object(&obj_str);
        let mut id = String::new();
        let mut desc = String::new();
        let mut downloads = 0u64;
        for (key, val) in &pairs {
            match key.as_str() {
                "ref" | "id" => id = val.clone(),
                "title" => desc = val.clone(),
                "totalViews" | "downloadCount" => downloads = val.parse().unwrap_or(0),
                _ => {}
            }
        }
        if !id.is_empty() {
            listings.push(DatasetListing { id, description: desc, downloads, tags: Vec::new() });
        }
        if listings.len() >= limit { break; }
    }
    Ok(listings)
}

/// Format dataset listings as a readable table.
pub fn format_listings(listings: &[DatasetListing]) -> String {
    if listings.is_empty() { return "No datasets found.".into(); }
    let mut out = String::new();
    out.push_str(&format!("Found {} datasets:\n\n", listings.len()));
    for (i, ds) in listings.iter().enumerate() {
        out.push_str(&format!("  {}. {}\n", i + 1, ds.id));
        if !ds.description.is_empty() {
            let desc = if ds.description.len() > 80 { &ds.description[..80] } else { &ds.description };
            out.push_str(&format!("     {desc}...\n"));
        }
        if ds.downloads > 0 {
            out.push_str(&format!("     Downloads: {}\n", ds.downloads));
        }
        out.push('\n');
    }
    out
}

/// Split a JSON array body into individual object strings.
fn split_json_objects(body: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth = 0i32;
    let mut start = None;
    for (i, ch) in body.char_indices() {
        match ch {
            '{' => {
                if depth == 0 { start = Some(i); }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        objects.push(body[s..=i].to_string());
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }
    objects
}

// ==================== Dataset builder (create your own) ====================

/// Builder for creating custom datasets programmatically.
///
/// # Example
/// ```ignore
/// use rust_nn::data::DatasetBuilder;
///
/// let dataset = DatasetBuilder::new("my-dataset")
///     .add_float_row(&[1.0, 2.0, 3.0], 0.0)
///     .add_float_row(&[4.0, 5.0, 6.0], 1.0)
///     .add_float_row(&[7.0, 8.0, 9.0], 0.0)
///     .build();
/// ```
#[derive(Debug)]
pub struct DatasetBuilder {
    name: String,
    float_columns: HashMap<String, Vec<f32>>,
    text_columns: HashMap<String, Vec<String>>,
    num_rows: usize,
}

impl DatasetBuilder {
    /// Create a new dataset builder.
    pub fn new(name: impl Into<String>) -> Self {
        DatasetBuilder {
            name: name.into(),
            float_columns: HashMap::new(),
            text_columns: HashMap::new(),
            num_rows: 0,
        }
    }

    /// Add a row of float features with a float label.
    pub fn add_float_row(mut self, features: &[f32], label: f32) -> Self {
        for (i, &val) in features.iter().enumerate() {
            self.float_columns.entry(format!("f{i}")).or_default().push(val);
        }
        self.float_columns.entry("label".into()).or_default().push(label);
        self.num_rows += 1;
        self
    }

    /// Add a row of text features with a text label.
    pub fn add_text_row(mut self, text: &str, label: &str) -> Self {
        self.text_columns.entry("text".into()).or_default().push(text.into());
        self.text_columns.entry("label".into()).or_default().push(label.into());
        self.num_rows += 1;
        self
    }

    /// Add a named float column.
    pub fn add_float_column(mut self, name: &str, data: Vec<f32>) -> Self {
        self.num_rows = self.num_rows.max(data.len());
        self.float_columns.insert(name.into(), data);
        self
    }

    /// Add a named text column.
    pub fn add_text_column(mut self, name: &str, data: Vec<String>) -> Self {
        self.num_rows = self.num_rows.max(data.len());
        self.text_columns.insert(name.into(), data);
        self
    }

    /// Build the dataset.
    pub fn build(self) -> Dataset {
        let mut ds = Dataset::new(self.name);
        ds.num_rows = self.num_rows;
        for (name, data) in self.float_columns {
            ds.add_float_column(&name, data);
        }
        for (name, data) in self.text_columns {
            ds.add_text_column(&name, data);
        }
        ds
    }

    /// Build and save as CSV.
    pub fn build_csv(&self, path: &str) -> std::io::Result<()> {
        let mut content = String::new();
        // Header
        let mut headers: Vec<&str> = self.float_columns.keys().map(|s| s.as_str()).collect();
        headers.extend(self.text_columns.keys().map(|s| s.as_str()));
        content.push_str(&headers.join(","));
        content.push('\n');
        // Rows
        for row in 0..self.num_rows {
            let mut fields = Vec::new();
            for h in &headers {
                if let Some(col) = self.float_columns.get(*h) {
                    fields.push(col.get(row).copied().unwrap_or(0.0).to_string());
                } else if let Some(col) = self.text_columns.get(*h) {
                    fields.push(format!("\"{}\"", col.get(row).map(|s| s.as_str()).unwrap_or("")));
                }
            }
            content.push_str(&fields.join(","));
            content.push('\n');
        }
        std::fs::write(path, content)
    }
}

/// Load a HuggingFace dataset with credentials.
pub fn load_huggingface_auth(
    dataset: &str,
    split: &str,
    max_rows: usize,
    creds: &Credentials,
) -> Result<Dataset, String> {
    load_huggingface(dataset, split, max_rows, creds.hf_token.as_deref())
}

/// Load a Kaggle dataset with credentials (downloads CSV file).
pub fn load_kaggle_auth(
    dataset: &str,
    file: &str,
    creds: &Credentials,
) -> Result<Dataset, String> {
    let username = creds.kaggle_username.as_ref().ok_or("Kaggle username required")?;
    let key = creds.kaggle_key.as_ref().ok_or("Kaggle key required")?;
    let url = format!("https://www.kaggle.com/api/v1/datasets/download/{dataset}");
    let agent = ureq::Agent::new();
    let response = agent
        .get(&url)
        .set("Authorization", &format!("Basic {}", base64_encode(&format!("{username}:{key}"))))
        .call()
        .map_err(|e| format!("Kaggle download failed: {e}"))?;
    let body = response.into_string().map_err(|e| format!("Read error: {e}"))?;
    if file.ends_with(".csv") {
        parse_delimited(&body, ',')
    } else {
        Err(format!("File '{file}' needs ZIP extraction. Extract manually and use load_csv()."))
    }
}

#[cfg(test)]
mod tests_creds {
    use super::*;

    #[test]
    fn credentials_builder() {
        let c = Credentials::new()
            .with_hf_token("hf_test123")
            .with_kaggle("user", "key456");
        assert!(c.has_hf());
        assert!(c.has_kaggle());
    }

    #[test]
    fn credentials_empty() {
        let c = Credentials::new();
        assert!(!c.has_hf());
        assert!(!c.has_kaggle());
    }

    #[test]
    fn dataset_builder_float() {
        let ds = DatasetBuilder::new("test")
            .add_float_row(&[1.0, 2.0], 0.0)
            .add_float_row(&[3.0, 4.0], 1.0)
            .build();
        assert_eq!(ds.num_rows, 2);
        assert!(ds.get_float("f0").is_some());
        assert!(ds.get_float("label").is_some());
    }

    #[test]
    fn dataset_builder_text() {
        let ds = DatasetBuilder::new("text_ds")
            .add_text_row("hello world", "positive")
            .add_text_row("bad day", "negative")
            .build();
        assert_eq!(ds.num_rows, 2);
        assert!(ds.get_text("text").is_some());
    }

    #[test]
    fn dataset_builder_named_columns() {
        let ds = DatasetBuilder::new("named")
            .add_float_column("x", vec![1.0, 2.0, 3.0])
            .add_float_column("y", vec![4.0, 5.0, 6.0])
            .add_text_column("category", vec!["a".into(), "b".into(), "c".into()])
            .build();
        assert_eq!(ds.num_rows, 3);
        assert!(ds.get_float("x").is_some());
        assert!(ds.get_text("category").is_some());
    }

    #[test]
    fn dataset_builder_csv() {
        let builder = DatasetBuilder::new("csv_test")
            .add_float_column("a", vec![1.0, 2.0])
            .add_float_column("b", vec![3.0, 4.0]);
        builder.build_csv("/tmp/test_builder.csv").unwrap();
        let content = std::fs::read_to_string("/tmp/test_builder.csv").unwrap();
        assert!(content.contains("a") && content.contains("b"));
        assert!(content.contains("1"));
    }

    #[test]
    fn split_json_objects_basic() {
        let json = r#"[{"id":"a","x":1},{"id":"b","x":2}]"#;
        let inner = json.trim().strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap();
        let objs = split_json_objects(inner);
        assert_eq!(objs.len(), 2);
        assert!(objs[0].contains("\"a\""));
    }

    #[test]
    fn format_listings_display() {
        let listings = vec![
            DatasetListing { id: "test/ds1".into(), description: "Test dataset".into(), downloads: 42, tags: vec![] },
        ];
        let formatted = format_listings(&listings);
        assert!(formatted.contains("test/ds1"));
        assert!(formatted.contains("42"));
    }
}

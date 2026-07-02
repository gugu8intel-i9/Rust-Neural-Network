//! Model serialization: save/load tensors and full models in a compact binary format.
//!
//! # Format (`rnnb` — "rust-nn binary")
//!
//! ```text
//! Magic:   "RNNB" (4 bytes)
//! Version: u32 (little-endian)
//! Count:   u32 (number of named tensors)
//! For each tensor:
//!   name_len: u32
//!   name:     [u8; name_len]            (UTF-8)
//!   ndim:     u32
//!   dims:     [u64; ndim]               (little-endian)
//!   numel:    u32  (= product of dims)
//!   data:     [u8; numel * 4]           (little-endian f32)
//! ```
//!
//! Also supports **safetensors** export/import — the de-facto industry standard format — via
//! [`safetensors_export`] / [`safetensors_import`], enabling interop with HuggingFace, PyTorch, etc.

use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

const MAGIC: &[u8; 4] = b"RNNB";
const VERSION: u32 = 1;

/// Serialize named tensors to the compact `rnnb` binary format.
pub fn serialize(named: &[(&str, &Tensor)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64 + named.len() * 128);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(named.len() as u32).to_le_bytes());

    for (name, tensor) in named {
        let name_bytes = name.as_bytes();
        let data: Vec<f32> = tensor.data().iter().copied().collect();
        let shape = tensor.shape();

        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in &shape {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        for &v in &data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// Deserialize named tensors from the `rnnb` binary format.
pub fn deserialize(bytes: &[u8]) -> Result<Vec<(String, Tensor)>, String> {
    if bytes.len() < 12 || &bytes[0..4] != MAGIC {
        return Err("invalid magic header".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(format!("unsupported version {version}"));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut pos = 12;
    let mut result = Vec::with_capacity(count);

    for _ in 0..count {
        let name_len = read_u32(bytes, &mut pos)? as usize;
        let name = String::from_utf8(bytes[pos..pos + name_len].to_vec())
            .map_err(|e| format!("invalid name: {e}"))?;
        pos += name_len;

        let ndim = read_u32(bytes, &mut pos)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(bytes, &mut pos)? as usize);
        }

        let numel = read_u32(bytes, &mut pos)? as usize;
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            data.push(f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }

        let tensor = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| format!("shape error: {e}"))?,
            true,
        );
        result.push((name, tensor));
    }
    Ok(result)
}

/// Serialize a model's parameters (named by index) to `rnnb`.
pub fn save_model(model: &dyn Module) -> Vec<u8> {
    let params = model.parameters();
    let named: Vec<(&str, &Tensor)> = params
        .iter()
        .enumerate()
        .map(|(i, t)| (leak_str(format!("param_{i}")), t))
        .collect();
    serialize(&named)
}

/// Serialize a model's parameters with custom names.
pub fn save_model_named(model: &dyn Module, names: &[String]) -> Vec<u8> {
    let params = model.parameters();
    let named: Vec<(&str, &Tensor)> = params
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let name = names.get(i).cloned().unwrap_or_else(|| format!("param_{i}"));
            (leak_str(name), t)
        })
        .collect();
    serialize(&named)
}

/// Load parameters from `rnnb` data into the model's existing parameter tensors (in-place copy).
/// The number of loaded tensors must match the model's parameter count.
pub fn load_model(model: &dyn Module, bytes: &[u8]) -> Result<usize, String> {
    let loaded = deserialize(bytes)?;
    let params = model.parameters();
    if loaded.len() != params.len() {
        return Err(format!(
            "parameter count mismatch: model has {}, checkpoint has {}",
            params.len(),
            loaded.len()
        ));
    }
    for ((_, src), dst) in loaded.iter().zip(params.iter()) {
        let data = src.data();
        dst.0.write().unwrap().data.assign(&data);
    }
    Ok(loaded.len())
}

// ==================== safetensors interop ====================

/// Export named tensors in the **safetensors** format (JSON header + raw f32 data).
///
/// Format: 8-byte little-endian header length, then a JSON object mapping name → {dtype, shape,
/// data_offsets}, then the concatenated raw data bytes. Compatible with HuggingFace safetensors.
pub fn safetensors_export(named: &[(&str, &Tensor)]) -> Vec<u8> {
    let mut data_section = Vec::new();
    let mut entries: Vec<(String, String, Vec<u64>, [usize; 2])> = Vec::new();

    for (name, tensor) in named {
        let raw: Vec<u8> = tensor
            .data()
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let start = data_section.len();
        data_section.extend_from_slice(&raw);
        let end = data_section.len();
        let shape: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
        entries.push((
            name.to_string(),
            "F32".to_string(),
            shape,
            [start, end],
        ));
    }

    // Build JSON header.
    let mut json = String::from("{");
    for (i, (name, dtype, shape, offsets)) in entries.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","),
            offsets[0],
            offsets[1]
        ));
    }
    json.push('}');

    let header_bytes = json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_section.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data_section);
    out
}

/// Import named tensors from the **safetensors** format.
pub fn safetensors_import(bytes: &[u8]) -> Result<Vec<(String, Tensor)>, String> {
    if bytes.len() < 8 {
        return Err("data too short".into());
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + header_len > bytes.len() {
        return Err("header length exceeds data".into());
    }
    let header_str = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|e| format!("invalid header JSON: {e}"))?;
    let header: serde_json_compat::Value =
        serde_json_compat::parse(header_str).ok_or::<String>("invalid header JSON".into())?;
    let map = header.as_object().ok_or("header is not an object")?;

    let data_start = 8 + header_len;
    let mut result = Vec::new();

    for (name, val) in map {
        let obj = val.as_object().ok_or("entry is not an object")?;
        let shape: Vec<usize> = obj
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_int().map(|i| i as usize)).collect())
            .unwrap_or_default();
        let offsets = obj
            .get("data_offsets")
            .and_then(|s| s.as_array())
            .ok_or("missing data_offsets")?;
        let start = offsets[0].as_int().ok_or("bad offset")? as usize;
        let end = offsets[1].as_int().ok_or("bad offset")? as usize;

        let raw = &bytes[data_start + start..data_start + end];
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let shape_ref: Vec<usize> = if shape.is_empty() { vec![1] } else { shape.clone() };
        let tensor = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape_ref), data)
                .map_err(|e| format!("shape error: {e}"))?,
            true,
        );
        result.push((name.clone(), tensor));
    }
    Ok(result)
}

// ==================== helpers ====================

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > bytes.len() {
        return Err("unexpected end of data".into());
    }
    let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > bytes.len() {
        return Err("unexpected end of data".into());
    }
    let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

/// Leak a string to get a &'static str (for the serialize API). Small leak, acceptable for
/// serialization (the strings are short and few).
fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// A minimal JSON parser for the safetensors header (avoids adding a serde_json dependency).
mod serde_json_compat {
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub enum Value {
        Object(HashMap<String, Value>),
        Array(Vec<Value>),
        Int(i64),
        Str(String),
        Null,
    }

    impl Value {
        pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
            match self {
                Value::Object(m) => Some(m),
                _ => None,
            }
        }
        pub fn as_array(&self) -> Option<&Vec<Value>> {
            match self {
                Value::Array(a) => Some(a),
                _ => None,
            }
        }
        pub fn as_int(&self) -> Option<i64> {
            match self {
                Value::Int(i) => Some(*i),
                _ => None,
            }
        }
    }

    pub fn parse(s: &str) -> Option<Value> {
        let mut chars = s.chars().peekable();
        parse_value(&mut chars)
    }

    fn skip_ws(chars: &mut std::iter::Peekable<std::str::Chars>) {
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() {
                chars.next();
            } else {
                break;
            }
        }
    }

    fn parse_value(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<Value> {
        skip_ws(chars);
        match chars.peek()? {
            '{' => parse_object(chars),
            '[' => parse_array(chars),
            '"' => parse_string(chars).map(Value::Str),
            c if c.is_ascii_digit() || *c == '-' => parse_number(chars),
            _ => None,
        }
    }

    fn parse_object(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<Value> {
        chars.next(); // consume '{'
        let mut map = HashMap::new();
        skip_ws(chars);
        if chars.peek() == Some(&'}') {
            chars.next();
            return Some(Value::Object(map));
        }
        loop {
            skip_ws(chars);
            let key = parse_string(chars)?;
            skip_ws(chars);
            if chars.next() != Some(':') {
                return None;
            }
            let val = parse_value(chars)?;
            map.insert(key, val);
            skip_ws(chars);
            match chars.next() {
                Some(',') => continue,
                Some('}') => break,
                _ => return None,
            }
        }
        Some(Value::Object(map))
    }

    fn parse_array(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<Value> {
        chars.next(); // consume '['
        let mut arr = Vec::new();
        skip_ws(chars);
        if chars.peek() == Some(&']') {
            chars.next();
            return Some(Value::Array(arr));
        }
        loop {
            let val = parse_value(chars)?;
            arr.push(val);
            skip_ws(chars);
            match chars.next() {
                Some(',') => continue,
                Some(']') => break,
                _ => return None,
            }
        }
        Some(Value::Array(arr))
    }

    fn parse_string(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<String> {
        if chars.next() != Some('"') {
            return None;
        }
        let mut s = String::new();
        while let Some(c) = chars.next() {
            match c {
                '"' => return Some(s),
                '\\' => {
                    if let Some(esc) = chars.next() {
                        s.push(match esc {
                            'n' => '\n',
                            't' => '\t',
                            '\\' => '\\',
                            '"' => '"',
                            other => other,
                        });
                    }
                }
                _ => s.push(c),
            }
        }
        None
    }

    fn parse_number(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<Value> {
        let mut s = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' {
                s.push(c);
                chars.next();
            } else {
                break;
            }
        }
        s.parse::<i64>().ok().map(Value::Int).or_else(|| s.parse::<f64>().ok().map(|_| Value::Int(0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Module, Sequential, ReLU};

    #[test]
    fn serialize_deserialize_roundtrip() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::from_vec(vec![5.0, 6.0], vec![2]);
        let named = vec![("weight", &t1), ("bias", &t2)];
        let bytes = serialize(&named);
        let loaded = deserialize(&bytes).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "weight");
        assert_eq!(loaded[0].1.shape(), vec![2, 2]);
        let d: Vec<f32> = loaded[0].1.data().iter().copied().collect();
        assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].0, "bias");
    }

    #[test]
    fn save_load_model_roundtrip() {
        let model = Sequential::new()
            .add(Linear::new(4, 8, true))
            .add(ReLU)
            .add(Linear::new(8, 2, true));
        let params_before: Vec<Vec<f32>> = model
            .parameters()
            .iter()
            .map(|t| t.data().iter().copied().collect())
            .collect();

        let bytes = save_model(&model);
        assert!(bytes.starts_with(b"RNNB"));

        // Mutate the model.
        for p in &model.parameters() {
            let mut inner = p.0.write().unwrap();
            inner.data.fill(0.0);
        }

        // Load back.
        let n = load_model(&model, &bytes).unwrap();
        assert_eq!(n, params_before.len());

        // Verify values match.
        let params_after: Vec<Vec<f32>> = model
            .parameters()
            .iter()
            .map(|t| t.data().iter().copied().collect())
            .collect();
        for (before, after) in params_before.iter().zip(params_after.iter()) {
            for (a, b) in before.iter().zip(after.iter()) {
                assert!((a - b).abs() < 1e-6, "value mismatch after load");
            }
        }
    }

    #[test]
    fn safetensors_export_import_roundtrip() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);
        let named = vec![("layer1.weight", &t1), ("layer2.weight", &t2)];

        let bytes = safetensors_export(&named);
        let loaded = safetensors_import(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "layer1.weight");
        assert_eq!(loaded[0].1.shape(), vec![2, 2]);
        let d: Vec<f32> = loaded[0].1.data().iter().copied().collect();
        assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].0, "layer2.weight");
        let d2: Vec<f32> = loaded[1].1.data().iter().copied().collect();
        assert_eq!(d2, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn handles_empty_model() {
        let bytes = serialize(&[]);
        assert!(bytes.starts_with(b"RNNB"));
        let loaded = deserialize(&bytes).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn handles_3d_tensors() {
        let t = Tensor::from_vec(vec![1.0; 24], vec![2, 3, 4]);
        let named = vec![("x", &t)];
        let bytes = serialize(&named);
        let loaded = deserialize(&bytes).unwrap();
        assert_eq!(loaded[0].1.shape(), vec![2, 3, 4]);
    }

    #[test]
    fn safetensors_empty_shape() {
        let t = Tensor::from_vec(vec![42.0], vec![1]);
        let named = vec![("scalar", &t)];
        let bytes = safetensors_export(&named);
        let loaded = safetensors_import(&bytes).unwrap();
        assert_eq!(loaded[0].0, "scalar");
    }

    #[test]
    fn invalid_magic_rejected() {
        let result = deserialize(b"XXXXrest of data");
        assert!(result.is_err());
    }

    #[test]
    fn file_io_roundtrip() {
        // Simulate file I/O via Vec<u8>.
        let model = Sequential::new().add(Linear::new(3, 5, true));
        let bytes = save_model(&model);
        // "Write" to file is just keeping the Vec.
        // "Read" back:
        let _ = load_model(&model, &bytes).unwrap();
    }

    #[test]
    fn save_model_named_custom_names() {
        let model = Sequential::new().add(Linear::new(3, 5, true));
        let names = vec!["weight".to_string(), "bias".to_string()];
        let bytes = save_model_named(&model, &names);
        let loaded = deserialize(&bytes).unwrap();
        assert_eq!(loaded[0].0, "weight");
        assert_eq!(loaded[1].0, "bias");
    }
}

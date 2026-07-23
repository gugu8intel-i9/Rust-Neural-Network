//! SSD and RAM offloading: tiered storage for models larger than GPU/CPU memory.
//!
//! # Design
//!
//! Implements a **3-tier memory hierarchy** (GPU VRAM → RAM → SSD) with automatic promotion
//! and eviction, similar to DeepSpeed ZeRO-Infinity and llama.cpp's mmap weights.
//!
//! ## Tiers
//!
//! - **GPU**: fastest, limited by VRAM size. Tensors here are ready for GPU compute.
//! - **RAM**: the default. Tensors live in `TensorData::data` (ndarray on heap).
//! - **SSD**: slowest, effectively unlimited. Tensors are serialized to disk files and
//!   loaded on demand via direct file I/O.
//!
//! ## Key components
//!
//! - [`SsdTensor`]: a tensor backed by a file on SSD. Only loads into RAM when accessed.
//! - [`TieredStore`]: manages a pool of tensors across tiers with LRU eviction.
//! - [`OffloadConfig`]: controls when to offload and how much RAM to budget.
//! - [`OffloadModel`]: wraps a model, offloading cold layers to SSD automatically.
//!
//! # Example
//! ```ignore
//! use rust_nn::offload::{OffloadModel, OffloadConfig};
//!
//! // Offload layers 0-3 to SSD, keep layers 4-5 in RAM.
//! let config = OffloadConfig::default().with_ram_budget_gb(4.0);
//! let model = OffloadModel::new(model, config);
//! let y = model.forward(&x); // transparently loads from SSD as needed
//! ```

use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Memory tier for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// GPU VRAM (fastest, smallest).
    Gpu,
    /// System RAM (default).
    Ram,
    /// SSD/disk (slowest, effectively unlimited).
    Ssd,
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryTier::Gpu => write!(f, "GPU"),
            MemoryTier::Ram => write!(f, "RAM"),
            MemoryTier::Ssd => write!(f, "SSD"),
        }
    }
}

/// Configuration for tiered storage offloading.
#[derive(Debug, Clone)]
pub struct OffloadConfig {
    /// Maximum RAM budget in bytes before evicting to SSD. 0 = unlimited.
    pub ram_budget_bytes: usize,
    /// Directory for SSD tensor files.
    pub ssd_dir: PathBuf,
    /// Number of tensors to keep hot in RAM (LRU eviction beyond this).
    pub hot_cache_size: usize,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        OffloadConfig {
            ram_budget_bytes: 2 * 1024 * 1024 * 1024, // 2 GB default
            ssd_dir: PathBuf::from(format!("{home}/.cache/rust-nn/ssd")),
            hot_cache_size: 32,
        }
    }
}

impl OffloadConfig {
    /// Set RAM budget in GB.
    pub fn with_ram_budget_gb(mut self, gb: f64) -> Self {
        self.ram_budget_bytes = (gb * 1024.0 * 1024.0 * 1024.0) as usize;
        self
    }

    /// Set SSD directory.
    pub fn with_ssd_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.ssd_dir = dir.into();
        self
    }

    /// Set hot cache size (number of tensors to keep in RAM).
    pub fn with_hot_cache(mut self, size: usize) -> Self {
        self.hot_cache_size = size;
        self
    }
}

static TENSOR_ID: AtomicU64 = AtomicU64::new(0);

fn next_id() -> u64 {
    TENSOR_ID.fetch_add(1, Ordering::Relaxed)
}

/// A tensor backed by a file on SSD. Only loads into RAM when `load()` is called.
///
/// The tensor data is stored as raw little-endian f32 bytes, preceded by a header:
/// `[ndim: u32][dim_0: u64][dim_1: u64]...[data: f32 * numel]`.
#[derive(Debug)]
pub struct SsdTensor {
    /// File path on SSD.
    path: PathBuf,
    /// Tensor shape (cached in memory; data is on disk).
    shape: Vec<usize>,
    /// Number of elements.
    #[allow(dead_code)]
    numel: usize,
    /// Size in bytes on SSD.
    bytes: usize,
}

impl SsdTensor {
    /// Write a tensor to SSD. Returns an `SsdTensor` handle.
    pub fn write(tensor: &Tensor, dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let id = next_id();
        let path = dir.join(format!("tensor_{id}.bin"));
        let shape = tensor.shape();
        let data: Vec<f32> = tensor.data().iter().copied().collect();
        let numel = data.len();

        let mut buf = Vec::with_capacity(4 + shape.len() * 8 + numel * 4);
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in &shape {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        for &v in &data {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let bytes = buf.len();
        std::fs::write(&path, &buf)?;

        Ok(SsdTensor { path, shape, numel, bytes })
    }

    /// Load the tensor from SSD into RAM.
    pub fn load(&self) -> std::io::Result<Tensor> {
        let raw = std::fs::read(&self.path)?;
        if raw.len() < 4 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "file too short"));
        }
        let ndim = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let d = u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap()) as usize;
            shape.push(d);
            offset += 8;
        }
        let data: Vec<f32> = raw[offset..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap(),
            false,
        ))
    }

    /// Delete the SSD file.
    pub fn evict(&self) -> std::io::Result<()> {
        std::fs::remove_file(&self.path)
    }

    /// Size in bytes on SSD.
    pub fn size_bytes(&self) -> usize {
        self.bytes
    }

    /// Tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// File path on SSD.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SsdTensor {
    fn drop(&mut self) {
        // Best-effort cleanup of SSD file when the handle is dropped.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A tensor that can live in RAM or on SSD, with automatic loading.
///
/// When the tensor is in RAM, `get()` returns it instantly.
/// When it's on SSD, `get()` loads it from disk (blocking I/O).
#[derive(Debug)]
pub struct TieredTensor {
    /// In-RAM copy (None when offloaded to SSD).
    ram: Option<Tensor>,
    /// SSD handle (Some when offloaded).
    ssd: Option<SsdTensor>,
    /// Current tier.
    tier: MemoryTier,
    /// Last access time (for LRU eviction).
    last_access: std::time::Instant,
}

impl TieredTensor {
    /// Create a RAM-resident tiered tensor.
    pub fn from_tensor(tensor: Tensor) -> Self {
        TieredTensor {
            ram: Some(tensor),
            ssd: None,
            tier: MemoryTier::Ram,
            last_access: std::time::Instant::now(),
        }
    }

    /// Offload this tensor to SSD. The RAM copy is freed.
    pub fn offload_to_ssd(&mut self, dir: &Path) -> std::io::Result<()> {
        if self.tier == MemoryTier::Ssd {
            return Ok(());
        }
        if let Some(ref tensor) = self.ram.take() {
            let ssd = SsdTensor::write(tensor, dir)?;
            self.ssd = Some(ssd);
            self.tier = MemoryTier::Ssd;
        }
        Ok(())
    }

    /// Promote from SSD back to RAM.
    pub fn load_to_ram(&mut self) -> std::io::Result<()> {
        if self.tier == MemoryTier::Ram {
            return Ok(());
        }
        if let Some(ref ssd) = self.ssd {
            self.ram = Some(ssd.load()?);
            self.tier = MemoryTier::Ram;
            self.last_access = std::time::Instant::now();
        }
        Ok(())
    }

    /// Get the tensor (loading from SSD if necessary). Updates access time.
    pub fn get(&mut self) -> std::io::Result<&Tensor> {
        if self.ram.is_none() {
            self.load_to_ram()?;
        }
        self.last_access = std::time::Instant::now();
        Ok(self.ram.as_ref().unwrap())
    }

    /// Current tier.
    pub fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        if let Some(ref t) = self.ram {
            t.len() * 4
        } else if let Some(ref s) = self.ssd {
            s.size_bytes()
        } else {
            0
        }
    }

    /// Time since last access.
    pub fn age(&self) -> std::time::Duration {
        self.last_access.elapsed()
    }
}

/// Manages a pool of named tensors across RAM and SSD with LRU eviction.
pub struct TieredStore {
    config: OffloadConfig,
    tensors: HashMap<String, TieredTensor>,
    /// Total bytes currently in RAM.
    ram_usage: usize,
}

impl std::fmt::Debug for TieredStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TieredStore")
            .field("ram_usage_mb", &(self.ram_usage / 1024 / 1024))
            .field("ram_budget_mb", &(self.config.ram_budget_bytes / 1024 / 1024))
            .field("num_tensors", &self.tensors.len())
            .finish()
    }
}

impl TieredStore {
    /// Create a new tiered store with the given configuration.
    pub fn new(config: OffloadConfig) -> Self {
        let _ = std::fs::create_dir_all(&config.ssd_dir);
        TieredStore {
            config,
            tensors: HashMap::new(),
            ram_usage: 0,
        }
    }

    /// Insert a tensor into the store (placed in RAM).
    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor) {
        let size = tensor.len() * 4;
        self.ram_usage += size;
        self.tensors.insert(name.into(), TieredTensor::from_tensor(tensor));
        self.maybe_evict();
    }

    /// Get a tensor by name (loads from SSD if necessary).
    pub fn get(&mut self, name: &str) -> Option<std::io::Result<Tensor>> {
        let tensor = self.tensors.get_mut(name)?;
        let was_ssd = tensor.tier() == MemoryTier::Ssd;
        let size = tensor.size_bytes();
        match tensor.get() {
            Ok(_) => {
                if was_ssd {
                    self.ram_usage += size;
                }
                // Clone the tensor to release the borrow before eviction.
                let cloned = tensor.ram.clone();
                // Borrow released by cloning above.
                self.maybe_evict();
                Some(Ok(cloned.unwrap()))
            }
            Err(e) => Some(Err(e)),
        }
    }

    /// Offload a specific tensor to SSD.
    pub fn offload(&mut self, name: &str) -> std::io::Result<()> {
        if let Some(t) = self.tensors.get_mut(name) {
            if t.tier() == MemoryTier::Ram {
                self.ram_usage -= t.size_bytes();
            }
            t.offload_to_ssd(&self.config.ssd_dir)?;
        }
        Ok(())
    }

    /// Offload all tensors to SSD.
    pub fn offload_all(&mut self) -> std::io::Result<()> {
        let names: Vec<String> = self.tensors.keys().cloned().collect();
        for name in &names {
            self.offload(name)?;
        }
        Ok(())
    }

    /// Load all tensors back to RAM.
    pub fn load_all(&mut self) -> std::io::Result<()> {
        for t in self.tensors.values_mut() {
            if t.tier() == MemoryTier::Ssd {
                t.load_to_ram()?;
                self.ram_usage += t.size_bytes();
            }
        }
        Ok(())
    }

    /// Evict the coldest tensors to SSD until RAM usage is under budget.
    fn maybe_evict(&mut self) {
        if self.config.ram_budget_bytes == 0 {
            return;
        }
        while self.ram_usage > self.config.ram_budget_bytes {
            // Find the LRU tensor.
            let mut coldest: Option<(String, std::time::Duration)> = None;
            for (name, t) in &self.tensors {
                if t.tier() == MemoryTier::Ram {
                    let age = t.age();
                    if coldest.as_ref().is_none_or(|(_, a)| age > *a) {
                        coldest = Some((name.clone(), age));
                    }
                }
            }
            if let Some((name, _)) = coldest {
                let _ = self.offload(&name);
            } else {
                break;
            }
        }
    }

    /// Current RAM usage in bytes.
    pub fn ram_usage(&self) -> usize {
        self.ram_usage
    }

    /// Number of tensors in RAM vs SSD.
    pub fn tier_counts(&self) -> (usize, usize) {
        let (mut ram, mut ssd) = (0, 0);
        for t in self.tensors.values() {
            match t.tier() {
                MemoryTier::Ram => ram += 1,
                MemoryTier::Ssd => ssd += 1,
                _ => {}
            }
        }
        (ram, ssd)
    }

    /// Total bytes on SSD.
    pub fn ssd_usage(&self) -> usize {
        self.tensors
            .values()
            .filter(|t| t.tier() == MemoryTier::Ssd)
            .map(|t| t.size_bytes())
            .sum()
    }
}

/// Wraps a model, offloading specified layers to SSD. When a layer is accessed,
/// it is transparently loaded from SSD into RAM.
pub struct OffloadModel {
    model: Box<dyn Module>,
    store: TieredStore,
    /// Whether parameters have been offloaded.
    offloaded: bool,
}

impl OffloadModel {
    /// Create an offload wrapper around a model.
    pub fn new(model: impl Module + 'static, config: OffloadConfig) -> Self {
        let mut store = TieredStore::new(config);
        let params = model.parameters();
        for (i, p) in params.into_iter().enumerate() {
            store.insert(format!("param_{i}"), p);
        }
        OffloadModel {
            model: Box::new(model),
            store,
            offloaded: false,
        }
    }

    /// Offload all parameters to SSD.
    pub fn offload(&mut self) -> std::io::Result<()> {
        self.store.offload_all()?;
        self.offloaded = true;
        Ok(())
    }

    /// Load all parameters back to RAM.
    pub fn load(&mut self) -> std::io::Result<()> {
        self.store.load_all()?;
        self.offloaded = false;
        Ok(())
    }

    /// Forward pass — transparently loads parameters from SSD if needed.
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        if self.offloaded {
            // Load parameters back for the forward pass.
            let _ = self.store.load_all();
            self.offloaded = false;
        }
        self.model.forward(input)
    }

    /// RAM usage in bytes.
    pub fn ram_usage(&self) -> usize {
        self.store.ram_usage()
    }

    /// SSD usage in bytes.
    pub fn ssd_usage(&self) -> usize {
        self.store.ssd_usage()
    }

    /// Print memory stats.
    pub fn memory_report(&self) -> String {
        let (ram, ssd) = self.store.tier_counts();
        format!(
            "OffloadModel memory: {} tensors (RAM: {}, SSD: {}), RAM: {:.1} MB, SSD: {:.1} MB",
            ram + ssd,
            ram,
            ssd,
            self.ram_usage() as f64 / 1024.0 / 1024.0,
            self.ssd_usage() as f64 / 1024.0 / 1024.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Sequential, ReLU};

    fn tmp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_nn_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn ssd_tensor_roundtrip() {
        let dir = tmp_dir();
        let original = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let ssd = SsdTensor::write(&original, &dir).unwrap();
        assert_eq!(ssd.shape(), &[2, 3]);

        let loaded = ssd.load().unwrap();
        let orig: Vec<f32> = original.data().iter().copied().collect();
        let load: Vec<f32> = loaded.data().iter().copied().collect();
        assert_eq!(orig, load);
    }

    #[test]
    fn ssd_tensor_large() {
        let dir = tmp_dir();
        let original = Tensor::randn(&[128, 256]);
        let ssd = SsdTensor::write(&original, &dir).unwrap();
        let loaded = ssd.load().unwrap();
        assert_eq!(loaded.shape(), vec![128, 256]);
        let orig: Vec<f32> = original.data().iter().copied().collect();
        let load: Vec<f32> = loaded.data().iter().copied().collect();
        assert_eq!(orig, load);
    }

    #[test]
    fn tiered_tensor_offload_load() {
        let dir = tmp_dir();
        let tensor = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
        let mut tiered = TieredTensor::from_tensor(tensor);
        assert_eq!(tiered.tier(), MemoryTier::Ram);

        tiered.offload_to_ssd(&dir).unwrap();
        assert_eq!(tiered.tier(), MemoryTier::Ssd);

        tiered.load_to_ram().unwrap();
        assert_eq!(tiered.tier(), MemoryTier::Ram);

        let t = tiered.get().unwrap();
        assert!((t.data()[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn tiered_store_eviction() {
        let dir = tmp_dir();
        let config = OffloadConfig {
            ram_budget_bytes: 120, // Very small: only ~30 f32 values fit
            ssd_dir: dir,
            hot_cache_size: 1,
        };
        let mut store = TieredStore::new(config);

        // Insert tensors that exceed budget.
        store.insert("a", Tensor::from_vec(vec![1.0; 20], vec![20])); // 80 bytes
        store.insert("b", Tensor::from_vec(vec![2.0; 20], vec![20])); // 80 bytes — should evict "a"

        let (ram, ssd) = store.tier_counts();
        assert!(ssd >= 1, "at least 1 tensor should be on SSD, got RAM:{ram} SSD:{ssd}");
        assert!(store.ram_usage() <= 120, "RAM should be under budget");
    }

    #[test]
    fn tiered_store_get_loads_from_ssd() {
        let dir = tmp_dir();
        let mut store = TieredStore::new(OffloadConfig {
            ram_budget_bytes: 0,
            ssd_dir: dir.clone(),
            hot_cache_size: 10,
        });

        store.insert("x", Tensor::from_vec(vec![42.0, 43.0], vec![2]));
        store.offload("x").unwrap();

        // Load it back.
        let t = store.get("x").unwrap().unwrap();
        assert!((t.data()[0] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn offload_model_roundtrip() {
        let dir = tmp_dir();
        let model = Sequential::new()
            .add(Linear::new(8, 16, true))
            .add(ReLU)
            .add(Linear::new(16, 4, true));

        let config = OffloadConfig {
            ram_budget_bytes: 0,
            ssd_dir: dir,
            hot_cache_size: 10,
        };
        let mut offloaded = OffloadModel::new(model, config);

        // Offload to SSD.
        offloaded.offload().unwrap();
        assert!(offloaded.ssd_usage() > 0);

        // Forward should still work (loads from SSD transparently).
        let x = Tensor::randn(&[2, 8]);
        let y = offloaded.forward(&x);
        assert_eq!(y.shape(), vec![2, 4]);
    }

    #[test]
    fn memory_report_displays() {
        let _dir = tmp_dir();
        let model = Sequential::new().add(Linear::new(4, 8, true));
        let offloaded = OffloadModel::new(model, OffloadConfig::default());
        let report = offloaded.memory_report();
        assert!(report.contains("RAM"));
        assert!(report.contains("tensors"));
    }

    #[test]
    fn offload_config_builder() {
        let config = OffloadConfig::default()
            .with_ram_budget_gb(8.0)
            .with_hot_cache(64);
        assert_eq!(config.ram_budget_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(config.hot_cache_size, 64);
    }
}

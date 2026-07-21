//! High-performance byte-level BPE tokenizer with information-theoretic merge scoring.
//!
//! # Design / innovations
//!
//! **Pluggable merge scoring.** Unlike standard BPE (which merges the most *frequent* adjacent
//! pair), this tokenizer supports three scoring strategies via [`MergeScoring`]:
//!   - [`Frequency`](MergeScoring::Frequency) — classic BPE (most frequent pair).
//!   - [`PMI`](MergeScoring::PMI) — Pointwise Mutual Information `count(uv)/(count(u)·count(v))`,
//!     the WordPiece insight: merges pairs that occur together far more than chance would
//!     predict, yielding more semantically coherent subwords.
//!   - [`Hybrid`](MergeScoring::Hybrid) — a normalized blend of both, balancing raw coverage
//!     with statistical significance.
//!
//! **Flat contiguous vocab storage.** All token bytes live in one `Vec<u8>` with an offsets
//! array, so [`BpeTokenizer::decode`] is a pair of `memcpy`s per token (no per-token allocation,
//! no 100k-way pointer chase) — the layout used by the fastest known decoders.
//!
//! **Parallel training & batch encoding.** Pair counting during training is parallelized with
//! `rayon` across chunks; [`BpeTokenizer::encode_batch`] / [`decode_batch`] parallelize across
//! texts.
//!
//! **Byte-level.** Every input byte is representable, so there are **no `<unk>` tokens** and
//! `decode(encode(text)) == text` is guaranteed for any input.
//!
//! **Regex-free pre-tokenizer.** A hand-written byte scanner (no `regex` dependency) splits text
//! into chunks (letters / digits / whitespace-prefixed words / multibyte groups) before BPE.
//!
//! References: BPE (Sennrich et al. 2016); WordPiece / PMI scoring; entropy-driven pre-tokenization
//! (Hu et al. 2025); flat-storage decoders (mojo-tokenizer, rs-bpe).

use rayon::prelude::*;
use std::collections::HashMap;

/// How merge candidates are scored during training.
#[derive(Debug, Clone, Copy, Default)]
pub enum MergeScoring {
    #[default]
    /// Classic BPE: merge the most frequent adjacent pair. Maximizes compression.
    Frequency,
    /// Pointwise Mutual Information: `count(uv) / (count(u) * count(v))`. Merges pairs that
    /// co-occur far more than chance predicts — finds more meaningful subwords (WordPiece idea).
    PMI,
    /// A normalized blend `α·freq_norm + (1-α)·pmi_norm`. `alpha` in `[0,1]`; `0.5` is balanced.
    Hybrid { alpha: f32 },
}

/// A high-performance byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Merge rules in priority order. `merges[i]` is the `(left_id, right_id)` pair merged at
    /// step `i`; its result is assigned id `256 + i`.
    pub merges: Vec<(u32, u32)>,
    /// `(left_id, right_id) -> rank` for O(1) priority lookup during encoding.
    merge_rank: HashMap<(u32, u32), u32>,
    /// Contiguous byte storage for ALL tokens (flat decode buffer).
    vocab_bytes: Vec<u8>,
    /// `offsets[id]..offsets[id+1]` indexes `vocab_bytes`. Length = num_tokens + 1.
    vocab_offsets: Vec<u32>,
    /// Number of tokens (256 base bytes + merges + specials).
    pub vocab_size: usize,
    /// Scoring strategy used during training (stored for reference).
    pub scoring: MergeScoring,
    /// Special tokens (e.g. `<pad>`, `<bos>`), mapped from text to id.
    special: HashMap<String, u32>,
}

impl BpeTokenizer {
    /// Build a tokenizer from an explicit merge list (e.g. loaded from a file). The first 256 ids
    /// are raw bytes `0..256`.
    pub fn from_merges(merges: Vec<(u32, u32)>) -> Self {
        let mut merge_rank = HashMap::new();
        for (rank, &(l, r)) in merges.iter().enumerate() {
            merge_rank.insert((l, r), rank as u32);
        }
        let vocab_size = 256 + merges.len();
        let mut tok = BpeTokenizer {
            merges,
            merge_rank,
            vocab_bytes: Vec::new(),
            vocab_offsets: Vec::new(),
            vocab_size,
            scoring: MergeScoring::Frequency,
            special: HashMap::new(),
        };
        tok.rebuild_flat_storage();
        tok
    }

    /// Rebuild the flat contiguous byte buffer and offsets array from the merge list.
    /// `vocab_bytes` holds raw bytes for id 0..256, then each merged token's concatenated bytes.
    fn rebuild_flat_storage(&mut self) {
        let n = self.vocab_size;
        // First, compute each token's bytes. Tokens 0..256 are single bytes.
        let mut token_bytes: Vec<Vec<u8>> = Vec::with_capacity(n);
        for b in 0..256u32 {
            token_bytes.push(vec![b as u8]);
        }
        for &(l, r) in &self.merges {
            let mut combined = Vec::new();
            if (l as usize) < token_bytes.len() {
                combined.extend_from_slice(&token_bytes[l as usize]);
            }
            if (r as usize) < token_bytes.len() {
                combined.extend_from_slice(&token_bytes[r as usize]);
            }
            token_bytes.push(combined);
        }

        // Flatten into one contiguous buffer with an offsets array.
        let mut vocab_bytes = Vec::new();
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        for tb in &token_bytes {
            vocab_bytes.extend_from_slice(tb);
            offsets.push(vocab_bytes.len() as u32);
        }
        // Append special tokens.
        for text in self.special.keys() {
            vocab_bytes.extend_from_slice(text.as_bytes());
            offsets.push(vocab_bytes.len() as u32);
        }
        self.vocab_bytes = vocab_bytes;
        self.vocab_offsets = offsets;
    }

    /// Train a byte-level BPE tokenizer on `corpus`, learning merges until the vocabulary reaches
    /// `vocab_size`. Pair counting is parallelized with rayon.
    ///
    /// `vocab_size` must be >= 256 (the base bytes). The scoring strategy selects which pairs are
    /// merged first.
    pub fn train(corpus: &str, vocab_size: usize, scoring: MergeScoring) -> Self {
        assert!(vocab_size >= 256, "vocab_size must be >= 256 (base bytes)");
        let target_merges = vocab_size - 256;

        // Pre-tokenize the corpus into "words" (byte chunks).
        let chunks: Vec<(usize, &[u8])> = pre_tokenize(corpus.as_bytes());
        // Represent each word as a Vec<u32> of initial byte ids.
        let mut words: Vec<Vec<u32>> = chunks
            .iter()
            .map(|(_, c)| c.iter().map(|&b| b as u32).collect())
            .filter(|w: &Vec<u32>| !w.is_empty())
            .collect();

        // Deduplicate identical words for speed (keep multiplicities).
        let mut word_counts: HashMap<Vec<u32>, u64> = HashMap::new();
        for w in words.drain(..) {
            *word_counts.entry(w).or_insert(0) += 1;
        }
        let mut words: Vec<Vec<u32>> = word_counts.keys().cloned().collect();
        let mut counts: Vec<u64> = word_counts.values().copied().collect();

        let mut merges: Vec<(u32, u32)> = Vec::with_capacity(target_merges);

        for _ in 0..target_merges {
            if words.is_empty() {
                break;
            }
            // Count adjacent pairs in parallel across words.
            // (pair, count)
            let pair_counts: HashMap<(u32, u32), u64> = count_pairs_parallel(&words, &counts);
            if pair_counts.is_empty() {
                break;
            }

            // Compute per-token counts (needed for PMI / hybrid).
            let token_counts = if matches!(scoring, MergeScoring::Frequency) {
                HashMap::new()
            } else {
                count_tokens(&words, &counts)
            };

            // Score every candidate pair and pick the best.
            let best = pair_counts
                .iter()
                .map(|(&pair, &c_uv)| (pair, score_pair(pair, c_uv, &token_counts, scoring)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(pair, _)| pair);

            let Some((l, r)) = best else { break };

            // Apply the merge to every word.
            let new_id = (256 + merges.len()) as u32;
            for (w, _cnt) in words.iter_mut().zip(counts.iter_mut()) {
                apply_merge(w, l, r, new_id);
            }
            // Re-merge duplicate words that became identical after the merge.
            merge_identical(&mut words, &mut counts);

            merges.push((l, r));
        }

        let mut tok = BpeTokenizer {
            merges,
            merge_rank: HashMap::new(),
            vocab_bytes: Vec::new(),
            vocab_offsets: Vec::new(),
            vocab_size: 0,
            scoring,
            special: HashMap::new(),
        };
        for (rank, &(l, r)) in tok.merges.iter().enumerate() {
            tok.merge_rank.insert((l, r), rank as u32);
        }
        tok.vocab_size = 256 + tok.merges.len();
        tok.rebuild_flat_storage();
        tok
    }

    /// Register a special token string, returning its id.
    pub fn add_special(&mut self, text: &str) -> u32 {
        if let Some(&id) = self.special.get(text) {
            return id;
        }
        let id = self.vocab_size as u32;
        self.special.insert(text.to_string(), id);
        self.vocab_size += 1;
        // Append to flat storage.
        let off = self.vocab_bytes.len() as u32;
        self.vocab_bytes.extend_from_slice(text.as_bytes());
        self.vocab_offsets.push(self.vocab_bytes.len() as u32);
        let _ = off;
        id
    }

    /// Decode a single token id to its raw bytes (a slice into the flat buffer — zero-copy).
    pub fn id_to_bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        if i + 1 < self.vocab_offsets.len() {
            let start = self.vocab_offsets[i] as usize;
            let end = self.vocab_offsets[i + 1] as usize;
            &self.vocab_bytes[start..end]
        } else {
            &[]
        }
    }

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.vocab_size
    }

    /// Returns true if the tokenizer has no learned merges and no special tokens.
    pub fn is_empty(&self) -> bool {
        self.merges.is_empty() && self.special.is_empty()
    }

    /// Encode a string into token ids, applying pre-tokenization then BPE merges.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let chunks = pre_tokenize(bytes);
        let mut ids = Vec::with_capacity(bytes.len() / 3 + 8);
        for (_, chunk) in chunks {
            self.encode_chunk(chunk, &mut ids);
        }
        ids
    }

    /// Encode with byte offsets: each `(id, range)` where `range` indexes the input bytes.
    /// Enables mapping tokens back to source spans (zero-copy alignment).
    pub fn encode_with_offsets(&self, text: &str) -> Vec<(u32, std::ops::Range<usize>)> {
        let bytes = text.as_bytes();
        let chunks = pre_tokenize(bytes);
        let mut out = Vec::new();
        for (start, chunk) in chunks {
            let mut ids = Vec::new();
            self.encode_chunk(chunk, &mut ids);
            // Distribute the chunk's byte range across its tokens proportionally is not exact for
            // merged tokens, so we record the *chunk* range per token group. For precise per-token
            // spans we decode each id's length and walk the offset.
            let mut off = start;
            for id in ids {
                let len = self.id_to_bytes(id).len();
                let end = (off + len).min(start + chunk.len());
                out.push((id, off..end));
                off = end;
            }
        }
        out
    }

    /// Encode one pre-tokenized chunk into ids (appended to `out`).
    fn encode_chunk(&self, chunk: &[u8], out: &mut Vec<u32>) {
        if chunk.is_empty() {
            return;
        }
        let mut tokens: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
        // Repeatedly merge the lowest-rank adjacent pair until none remain.
        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx: Option<usize> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                if let Some(&rank) = self.merge_rank.get(&(tokens[i], tokens[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(idx) = best_idx else { break };
            let new_id = 256 + best_rank;
            tokens[idx] = new_id;
            tokens.remove(idx + 1);
        }
        out.extend(tokens);
    }

    /// Decode token ids back to a `String` (lossy on partial codepoints, but the byte sequence
    /// is always exactly recovered for a full encoding).
    pub fn decode(&self, ids: &[u32]) -> String {
        // Collect bytes from the flat buffer (memcpy per token), then convert to UTF-8.
        let mut total = 0usize;
        for &id in ids {
            total += self.id_to_bytes(id).len();
        }
        let mut out = Vec::with_capacity(total);
        for &id in ids {
            out.extend_from_slice(self.id_to_bytes(id));
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Decode to raw bytes (exact, no UTF-8 conversion).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            out.extend_from_slice(self.id_to_bytes(id));
        }
        out
    }

    /// Encode many texts in parallel (rayon).
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>> {
        texts.par_iter().map(|t| self.encode(t)).collect()
    }

    /// Decode many id-lists in parallel (rayon).
    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> Vec<String> {
        batch.par_iter().map(|ids| self.decode(ids)).collect()
    }

    /// Count tokens without allocating the id vector (fast).
    pub fn count_tokens(&self, text: &str) -> usize {
        let bytes = text.as_bytes();
        let chunks = pre_tokenize(bytes);
        let mut count = 0usize;
        for (_, chunk) in chunks {
            let mut tokens: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
            loop {
                let mut best_rank = u32::MAX;
                let mut best_idx: Option<usize> = None;
                for i in 0..tokens.len().saturating_sub(1) {
                    if let Some(&rank) = self.merge_rank.get(&(tokens[i], tokens[i + 1])) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = Some(i);
                        }
                    }
                }
                let Some(idx) = best_idx else { break };
                tokens[idx] = 256 + best_rank;
                tokens.remove(idx + 1);
            }
            count += tokens.len();
        }
        count
    }

    /// Compression analytics: tokens produced per character (lower = more compression).
    pub fn compression_ratio(&self, text: &str) -> f64 {
        let chars = text.chars().count().max(1);
        let tokens = self.count_tokens(text);
        tokens as f64 / chars as f64
    }

    /// Serialize the tokenizer to a compact text format (merges as `left right` per line).
    pub fn save(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("# scoring={:?}\n", self.scoring));
        s.push_str(&format!("# merges={}\n", self.merges.len()));
        for &(l, r) in &self.merges {
            s.push_str(&format!("{l} {r}\n"));
        }
        if !self.special.is_empty() {
            s.push_str("# specials\n");
            for (text, id) in &self.special {
                s.push_str(&format!("{id} {}\n", text.replace('\n', "\\n")));
            }
        }
        s
    }

    /// Load a tokenizer from the text format produced by [`save`].
    pub fn load(text: &str) -> Self {
        let mut merges = Vec::new();
        let mut scoring = MergeScoring::Frequency;
        let mut specials: Vec<(String, u32)> = Vec::new();
        let mut in_specials = false;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("# scoring=") {
                scoring = match rest {
                    "Frequency" => MergeScoring::Frequency,
                    "PMI" => MergeScoring::PMI,
                    s if s.starts_with("Hybrid") => MergeScoring::Hybrid { alpha: 0.5 },
                    _ => MergeScoring::Frequency,
                };
                continue;
            }
            if line == "# specials" {
                in_specials = true;
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            if in_specials {
                let mut parts = line.splitn(2, ' ');
                let id: u32 = parts.next().unwrap().parse().unwrap_or(0);
                let t = parts.next().unwrap_or("").replace("\\n", "\n");
                specials.push((t, id));
            } else {
                let mut parts = line.split_whitespace();
                let l: u32 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                let r: u32 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                merges.push((l, r));
            }
        }
        let mut tok = Self::from_merges(merges);
        tok.scoring = scoring;
        for (text, _id) in specials {
            tok.add_special(&text);
        }
        tok
    }
}

// ============================ Pre-tokenizer (regex-free byte scanner) ============================

/// Split a byte sequence into chunks: whitespace-prefixed word runs, letter runs, digit runs,
/// grouped multibyte (non-ASCII) runs, and individual ASCII punctuation. Every input byte ends up
/// in exactly one chunk (a partition), guaranteeing a lossless round-trip.
///
/// Returns `(byte_offset, chunk)` pairs.
fn pre_tokenize(bytes: &[u8]) -> Vec<(usize, &[u8])> {
    let mut chunks = Vec::new();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        let start = i;
        if is_space_byte(bytes[i]) {
            // Consume a run of whitespace.
            i += 1;
            while i < n && is_space_byte(bytes[i]) {
                i += 1;
            }
            // Attach a following letter/digit run (GPT-style "leading space" token).
            if i < n && (is_ascii_letter(bytes[i]) || is_ascii_digit(bytes[i])) {
                let cls = byte_class(bytes[i]);
                while i < n && byte_class(bytes[i]) == cls {
                    i += 1;
                }
            }
        } else if is_ascii_letter(bytes[i]) || is_ascii_digit(bytes[i]) {
            let cls = byte_class(bytes[i]);
            i += 1;
            while i < n && byte_class(bytes[i]) == cls {
                i += 1;
            }
        } else if bytes[i] >= 0x80 {
            // Group consecutive non-ASCII bytes (keeps multibyte UTF-8 sequences together).
            i += 1;
            while i < n && bytes[i] >= 0x80 {
                i += 1;
            }
        } else {
            // Single ASCII punctuation / symbol byte.
            i += 1;
        }
        chunks.push((start, &bytes[start..i]));
    }
    chunks
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteClass {
    Letter,
    Digit,
    Other,
}

#[inline]
fn byte_class(b: u8) -> ByteClass {
    if is_ascii_letter(b) {
        ByteClass::Letter
    } else if is_ascii_digit(b) {
        ByteClass::Digit
    } else {
        ByteClass::Other
    }
}

#[inline]
fn is_ascii_letter(b: u8) -> bool {
    b.is_ascii_alphabetic()
}

#[inline]
fn is_ascii_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

#[inline]
fn is_space_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

// ============================ Training helpers ============================

/// Count adjacent pairs across all words, in parallel. `counts[i]` is the multiplicity of `words[i]`.
fn count_pairs_parallel(words: &[Vec<u32>], counts: &[u64]) -> HashMap<(u32, u32), u64> {
    words
        .par_iter()
        .zip(counts.par_iter())
        .fold(HashMap::new, |mut local, (word, &cnt)| {
            if word.len() < 2 {
                return local;
            }
            for w in 0..word.len() - 1 {
                *local.entry((word[w], word[w + 1])).or_insert(0) += cnt;
            }
            local
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        })
}

/// Count individual token occurrences (for PMI scoring).
fn count_tokens(words: &[Vec<u32>], counts: &[u64]) -> HashMap<u32, u64> {
    let mut map = HashMap::new();
    for (word, &cnt) in words.iter().zip(counts.iter()) {
        for &t in word {
            *map.entry(t).or_insert(0) += cnt;
        }
    }
    map
}

/// Score a candidate pair `(l, r)` given its co-occurrence count `c_uv` and per-token counts.
/// Higher is better. PMI and hybrid are normalized so the comparison is well-scaled.
fn score_pair(pair: (u32, u32), c_uv: u64, token_counts: &HashMap<u32, u64>, scoring: MergeScoring) -> f64 {
    let (l, r) = pair;
    let c_uv = c_uv as f64;
    match scoring {
        MergeScoring::Frequency => c_uv,
        MergeScoring::PMI => {
            let cl = token_counts.get(&l).copied().unwrap_or(1) as f64;
            let cr = token_counts.get(&r).copied().unwrap_or(1) as f64;
            c_uv / (cl * cr)
        }
        MergeScoring::Hybrid { alpha } => {
            let cl = token_counts.get(&l).copied().unwrap_or(1) as f64;
            let cr = token_counts.get(&r).copied().unwrap_or(1) as f64;
            let pmi = c_uv / (cl * cr);
            // Normalize: freq on a log scale, pmi likewise, so neither dominates trivially.
            let freq_norm = (c_uv + 1.0).ln();
            let pmi_norm = (pmi + 1.0).ln();
            alpha as f64 * freq_norm + (1.0 - alpha as f64) * pmi_norm
        }
    }
}

/// Replace every adjacent `(l, r)` in `word` with `new_id` (in place).
fn apply_merge(word: &mut Vec<u32>, l: u32, r: u32, new_id: u32) {
    let mut i = 0;
    while i + 1 < word.len() {
        if word[i] == l && word[i + 1] == r {
            word[i] = new_id;
            word.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

/// After a merge, identical words may have appeared; re-bucket them to keep the word list compact.
fn merge_identical(words: &mut Vec<Vec<u32>>, counts: &mut Vec<u64>) {
    let mut bucket: HashMap<Vec<u32>, u64> = HashMap::new();
    for (w, c) in words.drain(..).zip(counts.drain(..)) {
        *bucket.entry(w).or_insert(0) += c;
    }
    for (w, c) in bucket {
        words.push(w);
        counts.push(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ascii() {
        let tok = BpeTokenizer::train("hello world hello there world hello", 280, MergeScoring::Frequency);
        let text = "hello world hello there";
        let ids = tok.encode(text);
        let back = tok.decode(&ids);
        assert_eq!(back, text, "round-trip must recover the original text");
    }

    #[test]
    fn round_trip_unicode() {
        let corpus = "héllo wörld 你好世界 Привет مرحبا 🦀🎉";
        let tok = BpeTokenizer::train(corpus, 400, MergeScoring::Frequency);
        let ids = tok.encode(corpus);
        let back = tok.decode(&ids);
        assert_eq!(back, corpus, "byte-level must round-trip unicode/emoji");
    }

    #[test]
    fn round_trip_bytes_exact() {
        let tok = BpeTokenizer::train("the quick brown fox jumps over the lazy dog", 320, MergeScoring::Frequency);
        let bytes = b"\xff\xfe\x00\x01 the quick fox";
        let text = String::from_utf8_lossy(bytes);
        let ids = tok.encode(&text);
        assert_eq!(tok.decode_bytes(&ids), text.as_bytes());
    }

    #[test]
    fn vocab_size_respected() {
        // Small corpus may run out of pairs before reaching the target; vocab_size <= target.
        let tok = BpeTokenizer::train("abracadabra banana cabana", 280, MergeScoring::Frequency);
        assert!(tok.vocab_size <= 280);
        assert_eq!(tok.merges.len(), tok.vocab_size - 256);
    }

    #[test]
    fn merges_reduce_token_count() {
        let corpus = "banana banana banana banana banana banana";
        let tok = BpeTokenizer::train(corpus, 320, MergeScoring::Frequency);
        // With merges, "banana banana" should produce fewer tokens than raw bytes.
        let text = "banana banana banana";
        let ids = tok.encode(text);
        assert!(ids.len() < text.len(), "merges must reduce token count vs raw bytes");
    }

    #[test]
    fn pmi_scoring_works() {
        let corpus = "the quick brown fox the the the the the the the the the the";
        let tok = BpeTokenizer::train(corpus, 320, MergeScoring::PMI);
        // Still round-trips.
        let text = "the quick brown fox";
        assert_eq!(tok.decode(&tok.encode(text)), text);
        assert!(tok.vocab_size >= 256);
    }

    #[test]
    fn hybrid_scoring_works() {
        let corpus = "abracadabra banana cabana abracadabra banana";
        let tok = BpeTokenizer::train(corpus, 310, MergeScoring::Hybrid { alpha: 0.5 });
        let text = "banana cabana";
        assert_eq!(tok.decode(&tok.encode(text)), text);
    }

    #[test]
    fn pre_tokenize_partitions() {
        let text = "hello, world! 123 你好";
        let bytes = text.as_bytes();
        let chunks = pre_tokenize(bytes);
        // Reassembly must equal the input (partition property).
        let mut rebuilt = Vec::new();
        for (_, c) in &chunks {
            rebuilt.extend_from_slice(c);
        }
        assert_eq!(&rebuilt[..], bytes);
        // No empty chunks.
        assert!(chunks.iter().all(|(_, c)| !c.is_empty()));
    }

    #[test]
    fn pre_tokenize_word_boundaries() {
        let chunks = pre_tokenize(b"hello world");
        // "hello" and " world" (leading space attached).
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].1, b"hello");
        assert_eq!(chunks[1].1, b" world");
    }

    #[test]
    fn encode_with_offsets_aligns() {
        let tok = BpeTokenizer::train("hello world hello there world", 300, MergeScoring::Frequency);
        let text = "hello world";
        let spans = tok.encode_with_offsets(text);
        // Reconstruct bytes from spans.
        let bytes = text.as_bytes();
        let mut rebuilt = Vec::new();
        for (_, range) in &spans {
            rebuilt.extend_from_slice(&bytes[range.clone()]);
        }
        assert_eq!(rebuilt, bytes);
    }

    #[test]
    fn batch_encode_decode_parallel() {
        let tok = BpeTokenizer::train("one two three four five six seven", 300, MergeScoring::Frequency);
        let texts: Vec<String> = vec!["one two".into(), "three four".into(), "five six".into()];
        let batch = tok.encode_batch(&texts);
        assert_eq!(batch.len(), 3);
        let decoded = tok.decode_batch(&batch);
        for (d, t) in decoded.iter().zip(texts.iter()) {
            assert_eq!(d, t);
        }
    }

    #[test]
    fn count_tokens_matches_encode_len() {
        let tok = BpeTokenizer::train("banana banana banana", 300, MergeScoring::Frequency);
        let text = "banana banana";
        assert_eq!(tok.count_tokens(text), tok.encode(text).len());
    }

    #[test]
    fn save_load_round_trips() {
        let tok = BpeTokenizer::train("hello world hello there world hello", 290, MergeScoring::Frequency);
        let serialized = tok.save();
        let tok2 = BpeTokenizer::load(&serialized);
        let text = "hello world";
        assert_eq!(tok.encode(text), tok2.encode(text), "loaded tokenizer must encode identically");
    }

    #[test]
    fn special_tokens() {
        let mut tok = BpeTokenizer::train("hello world hello world hello", 270, MergeScoring::Frequency);
        let base_vs = tok.vocab_size as u32;
        let pad_id = tok.add_special("<pad>");
        let bos_id = tok.add_special("<bos>");
        assert_eq!(pad_id, base_vs);
        assert_eq!(bos_id, base_vs + 1);
        assert_eq!(tok.id_to_bytes(pad_id), b"<pad>");
        assert_eq!(tok.id_to_bytes(bos_id), b"<bos>");
    }

    #[test]
    fn flat_decode_is_correct() {
        let tok = BpeTokenizer::train("alphabet alphabet alphabet", 320, MergeScoring::Frequency);
        let ids = tok.encode("alphabet");
        // id_to_bytes must concatenate to "alphabet".
        let mut s = Vec::new();
        for &id in &ids {
            s.extend_from_slice(tok.id_to_bytes(id));
        }
        assert_eq!(&s, b"alphabet");
    }

    #[test]
    fn deterministic_encoding() {
        let tok = BpeTokenizer::train("the cat sat on the mat", 300, MergeScoring::Frequency);
        let a = tok.encode("the cat sat");
        let b = tok.encode("the cat sat");
        assert_eq!(a, b, "encoding must be deterministic");
    }

    #[test]
    fn empty_and_single_byte() {
        let tok = BpeTokenizer::train("abc", 258, MergeScoring::Frequency);
        assert!(tok.encode("").is_empty());
        let ids = tok.encode("a");
        assert_eq!(tok.decode(&ids), "a");
    }
}

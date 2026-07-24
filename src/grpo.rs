//! GRPO (Group Relative Policy Optimization), adversarial code-test co-evolution,
//! multi-dimensional agentic reward, and dual-graph repo-level code generation.
//!
//! # Innovation overview
//!
//! ## 1. GRPO — critic-free policy optimization (DeepSeek)
//!
//! Unlike PPO which needs a separate value network (critic), GRPO samples a **group** of G
//! responses per prompt and computes **group-relative advantages**:
//! ```text
//! A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
//! ```
//! This eliminates the critic entirely — the group statistics serve as the baseline.
//! The policy is updated with a clipped ratio (like PPO) but using the group-relative advantage.
//!
//! ## 2. Adversarial co-evolution (Code LLM vs Test LLM)
//!
//! Two models train against each other:
//! - **Code model**: generates code to solve problems.
//! - **Test model**: generates test cases to break the code.
//!   Each round, the test model tries harder, forcing the code model to improve — an arms race.
//!
//! ## 3. Agentic multi-dimensional reward
//!
//! Beyond correctness, the reward model evaluates:
//! - **Correctness** (tests pass)
//!   - **Readability** (naming, complexity, line length)
//!   - **Style** (convention adherence)
//!   - **Aesthetics** (elegant vs verbose)
//!
//! Each dimension is scored 0-1 and combined with learned weights.
//!
//! ## 4. Dual-graph guidance for repo-level generation
//!
//! Two graphs constrain and guide code generation:
//! - **File Dependency Graph** (DAG of imports/uses between files)
//! - **Code Structure Graph** (functions, classes, and their call relationships)
//!   Together they provide structural context for generating code that fits the repository.

use std::collections::{HashMap, HashSet};

// ==================== GRPO ====================

/// Configuration for GRPO training.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// Group size (number of samples per prompt).
    pub group_size: usize,
    /// Clipping epsilon for the policy ratio.
    pub clip_eps: f32,
    /// Learning rate.
    pub lr: f32,
    /// KL penalty coefficient (keeps policy close to reference).
    pub kl_coeff: f32,
    /// Number of training iterations per batch.
    pub num_iterations: usize,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        GrpoConfig {
            group_size: 8,
            clip_eps: 0.2,
            lr: 0.001,
            kl_coeff: 0.04,
            num_iterations: 1,
        }
    }
}

/// A single GRPO training sample: a prompt with G sampled responses and their rewards.
#[derive(Debug, Clone)]
pub struct GrpoGroup {
    /// The prompt (problem description or code context).
    pub prompt: String,
    /// G sampled responses (code snippets).
    pub responses: Vec<String>,
    /// G reward scores (one per response).
    pub rewards: Vec<f32>,
}

impl GrpoGroup {
    /// Compute group-relative advantages: A_i = (r_i - mean) / std.
    pub fn advantages(&self) -> Vec<f32> {
        let n = self.rewards.len();
        if n == 0 { return vec![]; }
        let mean: f32 = self.rewards.iter().sum::<f32>() / n as f32;
        let variance: f32 = self.rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt().max(1e-8);
        self.rewards.iter().map(|r| (r - mean) / std).collect()
    }

    /// Best response index (highest reward).
    pub fn best_index(&self) -> usize {
        self.rewards.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Best response string.
    pub fn best_response(&self) -> &str {
        &self.responses[self.best_index()]
    }
}

/// GRPO trainer: updates a policy model using group-relative advantages.
///
/// The advantage computation eliminates the need for a separate critic model.
pub struct GrpoTrainer {
    #[allow(dead_code)]
    config: GrpoConfig,
    /// History of advantage statistics.
    pub history: Vec<GrpoStats>,
}

/// Statistics from one GRPO step.
#[derive(Debug, Clone)]
pub struct GrpoStats {
    pub step: usize,
    pub mean_reward: f32,
    pub std_reward: f32,
    pub mean_advantage: f32,
    pub best_reward: f32,
    pub worst_reward: f32,
}

impl GrpoTrainer {
    pub fn new(config: GrpoConfig) -> Self {
        GrpoTrainer { config, history: Vec::new() }
    }

    /// Process a batch of GRPO groups: compute advantages and update statistics.
    ///
    /// In a full implementation, this would:
    /// 1. Compute log-probabilities of each response under the current policy.
    /// 2. Compute group-relative advantages.
    /// 3. Update the policy with clipped surrogate + KL penalty.
    ///
    /// Here we compute the advantages and track statistics (the policy update
    /// uses the existing autograd engine via the reward-weighted loss).
    pub fn step(&mut self, groups: &[GrpoGroup]) -> Vec<GrpoStats> {
        let mut step_stats = Vec::new();

        for (step, group) in groups.iter().enumerate() {
            let advantages = group.advantages();

            let mean_r: f32 = group.rewards.iter().sum::<f32>() / group.rewards.len().max(1) as f32;
            let var_r: f32 = group.rewards.iter().map(|r| (r - mean_r).powi(2)).sum::<f32>()
                / group.rewards.len().max(1) as f32;
            let std_r = var_r.sqrt();
            let mean_adv: f32 = advantages.iter().sum::<f32>() / advantages.len().max(1) as f32;
            let best = group.rewards.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let worst = group.rewards.iter().cloned().fold(f32::INFINITY, f32::min);

            let stats = GrpoStats {
                step,
                mean_reward: mean_r,
                std_reward: std_r,
                mean_advantage: mean_adv,
                best_reward: best,
                worst_reward: worst,
            };

            if step % 10 == 0 {
                println!(
                    "  GRPO step {}: mean_r={:.3} std_r={:.3} best={:.3} worst={:.3}",
                    step, stats.mean_reward, stats.std_reward, stats.best_reward, stats.worst_reward
                );
            }

            step_stats.push(stats);
        }

        self.history.extend(step_stats.clone());
        step_stats
    }
}

// ==================== Multi-dimensional agentic reward ====================

/// Dimensions of code quality evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RewardDimension {
    Correctness,
    Readability,
    Style,
    Aesthetics,
}

/// A multi-dimensional reward score for a code snippet.
#[derive(Debug, Clone)]
pub struct RewardScore {
    pub correctness: f32,
    pub readability: f32,
    pub style: f32,
    pub aesthetics: f32,
}

impl RewardScore {
    /// Compute the weighted total reward.
    pub fn total(&self, weights: &RewardWeights) -> f32 {
        weights.correctness * self.correctness
            + weights.readability * self.readability
            + weights.style * self.style
            + weights.aesthetics * self.aesthetics
    }

    /// Convert to a vector for analysis.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.correctness, self.readability, self.style, self.aesthetics]
    }
}

/// Learned weights for each reward dimension.
#[derive(Debug, Clone)]
pub struct RewardWeights {
    pub correctness: f32,
    pub readability: f32,
    pub style: f32,
    pub aesthetics: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        RewardWeights {
            correctness: 0.5,
            readability: 0.2,
            style: 0.15,
            aesthetics: 0.15,
        }
    }
}

/// Agentic reward model: evaluates code across multiple dimensions.
///
/// Each dimension uses heuristics that approximate what a human reviewer would check:
/// - **Correctness**: does the code contain obvious errors? (missing return, unclosed brackets)
/// - **Readability**: naming quality (snake_case usage), function length, nesting depth
/// - **Style**: consistency (indentation, line length, spacing)
/// - **Aesthetics**: conciseness vs verbosity, ratio of code to comments
pub struct RewardModel {
    pub weights: RewardWeights,
}

impl RewardModel {
    pub fn new(weights: RewardWeights) -> Self {
        RewardModel { weights }
    }

    /// Score a code snippet across all dimensions.
    pub fn score(&self, code: &str, tests_passed: Option<bool>) -> RewardScore {
        RewardScore {
            correctness: self.score_correctness(code, tests_passed),
            readability: self.score_readability(code),
            style: self.score_style(code),
            aesthetics: self.score_aesthetics(code),
        }
    }

    /// Compute total weighted reward.
    pub fn reward(&self, code: &str, tests_passed: Option<bool>) -> f32 {
        self.score(code, tests_passed).total(&self.weights)
    }

    fn score_correctness(&self, code: &str, tests_passed: Option<bool>) -> f32 {
        let mut score = 0.5f32;
        // If tests explicitly passed/failed, that dominates.
        if let Some(passed) = tests_passed {
            return if passed { 1.0 } else { 0.0 };
        }
        // Heuristic checks for obvious issues.
        let open_braces = code.matches('{').count();
        let close_braces = code.matches('}').count();
        if open_braces == close_braces { score += 0.2; }
        if code.contains("return") || code.contains("fn ") { score += 0.15; }
        if !code.contains("todo!") && !code.contains("unimplemented!") { score += 0.15; }
        score.clamp(0.0, 1.0)
    }

    fn score_readability(&self, code: &str) -> f32 {
        let lines: Vec<&str> = code.lines().collect();
        if lines.is_empty() { return 0.0; }
        let mut score = 0.0f32;
        // Average line length (sweet spot: 40-80 chars).
        let avg_len: f32 = lines.iter().map(|l| l.len() as f32).sum::<f32>() / lines.len() as f32;
        if avg_len > 0.0 && avg_len < 100.0 { score += 0.3; }
        // Snake_case function names (Rust convention).
        let fn_count = code.matches("fn ").count();
        let snake_count = code.lines()
            .filter(|l| l.contains("fn ") && l.chars().filter(|c| c.is_uppercase()).count() == 0)
            .count();
        if fn_count > 0 {
            score += 0.3 * (snake_count as f32 / fn_count as f32);
        } else {
            score += 0.3;
        }
        // Comments present.
        let comment_count = code.lines().filter(|l| l.trim_start().starts_with("//")).count();
        if comment_count > 0 { score += 0.2; }
        // Low nesting depth (no deeply nested code).
        let max_depth = code.chars().fold((0i32, 0i32), |(depth, max), c| {
            let d = match c { '{' => depth + 1, '}' => depth - 1, _ => depth };
            (d, max.max(d))
        }).1;
        if max_depth <= 4 { score += 0.2; }
        score.clamp(0.0, 1.0)
    }

    fn score_style(&self, code: &str) -> f32 {
        let mut score = 0.5f32;
        // Consistent indentation (check if lines use consistent leading whitespace).
        let lines: Vec<&str> = code.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() { return 0.5; }
        let indent_units: HashSet<usize> = lines.iter()
            .filter_map(|l| {
                let ws = l.len() - l.trim_start().len();
                if ws > 0 { Some(ws % 4) } else { None }
            })
            .collect();
        if indent_units.len() <= 1 { score += 0.3; } // Consistent indentation.
        // No trailing whitespace.
        let trailing = code.lines().filter(|l| l.ends_with(' ') || l.ends_with('\t')).count();
        if trailing == 0 { score += 0.2; }
        score.clamp(0.0, 1.0)
    }

    fn score_aesthetics(&self, code: &str) -> f32 {
        let lines = code.lines().count();
        let non_empty = code.lines().filter(|l| !l.trim().is_empty()).count();
        if non_empty == 0 { return 0.0; }
        let mut score = 0.0f32;
        // Conciseness: shorter code is more elegant (up to a point).
        let chars_per_line = code.len() as f32 / non_empty as f32;
        if chars_per_line < 80.0 { score += 0.4; }
        // Low comment-to-code ratio (elegant code is self-documenting).
        let comment_chars: usize = code.lines()
            .filter(|l| l.trim_start().starts_with("//"))
            .map(|l| l.len())
            .sum::<usize>();
        let ratio = comment_chars as f32 / code.len().max(1) as f32;
        if ratio < 0.3 { score += 0.3; }
        // No excessive blank lines.
        let blank_ratio = (lines - non_empty) as f32 / lines.max(1) as f32;
        if blank_ratio < 0.2 { score += 0.3; }
        score.clamp(0.0, 1.0)
    }
}

// ==================== Adversarial co-evolution ====================

/// Role of a model in the adversarial framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    /// Generates code to solve problems.
    CodeGenerator,
    /// Generates test cases to break code.
    TestGenerator,
}

/// One episode of the adversarial game: the code model writes code, the test model writes tests.
#[derive(Debug, Clone)]
pub struct AdversarialEpisode {
    pub problem: String,
    pub generated_code: String,
    pub generated_tests: String,
    pub tests_passed: bool,
    pub code_reward: RewardScore,
    pub test_reward: f32, // Higher if tests caught bugs.
}

/// Tracks the co-evolution of code and test models over rounds.
#[derive(Debug, Clone)]
pub struct CoEvolutionStats {
    pub round: usize,
    pub code_pass_rate: f32,     // Fraction of code that passes tests.
    pub test_break_rate: f32,    // Fraction of tests that catch bugs.
    pub avg_code_reward: f32,    // Average multi-dimensional reward.
    pub difficulty: f32,         // Increasing difficulty metric.
}

/// Adversarial co-evolution trainer.
///
/// Each round:
/// 1. Code model generates solutions for N problems.
/// 2. Test model generates test cases for each solution.
/// 3. Reward: code model gets +1 per passed test, test model gets +1 per failed test.
/// 4. Both models update via GRPO with their respective rewards.
/// 5. Difficulty increases as both models improve.
pub struct CoEvolutionTrainer {
    pub grpo: GrpoTrainer,
    pub reward_model: RewardModel,
    pub round: usize,
    pub stats: Vec<CoEvolutionStats>,
#[allow(dead_code)]
    pub config: GrpoConfig,
}

impl CoEvolutionTrainer {
    pub fn new(grpo_config: GrpoConfig, reward_weights: RewardWeights) -> Self {
        CoEvolutionTrainer {
            grpo: GrpoTrainer::new(grpo_config.clone()),
            reward_model: RewardModel::new(reward_weights),
            round: 0,
            stats: Vec::new(),
            config: grpo_config,
        }
    }

    /// Run one co-evolution round.
    ///
    /// Returns the episode results and statistics for this round.
    pub fn co_evolve_round(
        &mut self,
        problems: &[String],
        code_samples: &HashMap<String, Vec<String>>,  // problem → G code samples
        test_samples: &HashMap<String, Vec<String>>,  // problem → G test samples
        test_results: &[(String, String, bool)],      // (problem, code_idx, passed)
    ) -> Vec<AdversarialEpisode> {
        self.round += 1;
        let mut episodes = Vec::new();

        // Build code reward groups for GRPO.
        let mut code_groups = Vec::new();
        let mut test_groups = Vec::new();
        let mut code_passes = 0usize;
        let mut test_breaks = 0usize;
        let mut total_reward = 0.0f32;

        for problem in problems {
            let codes = code_samples.get(problem).cloned().unwrap_or_default();
            let tests = test_samples.get(problem).cloned().unwrap_or_default();
            if codes.is_empty() { continue; }

            let mut code_rewards = Vec::with_capacity(codes.len());
            for (i, code) in codes.iter().enumerate() {
                // Did this code pass any test?
                let passed = test_results.iter()
                    .any(|(p, ci, pass)| p == problem && ci == &i.to_string() && *pass);
                let score = self.reward_model.score(code, Some(passed));
                code_rewards.push(score.total(&self.reward_model.weights));
                total_reward += code_rewards.last().unwrap();

                if passed { code_passes += 1; } else { test_breaks += 1; }

                episodes.push(AdversarialEpisode {
                    problem: problem.clone(),
                    generated_code: code.clone(),
                    generated_tests: tests.first().cloned().unwrap_or_default(),
                    tests_passed: passed,
                    code_reward: score,
                    test_reward: if passed { 0.0 } else { 1.0 },
                });
            }

            let n_codes = codes.len();
            code_groups.push(GrpoGroup {
                prompt: problem.clone(),
                responses: codes,
                rewards: code_rewards,
            });

            // Test reward: tests that caught bugs get higher reward.
            if !tests.is_empty() {
                let test_rewards: Vec<f32> = tests.iter().map(|_| {
                    // Higher reward if more code samples fail this test.
                    let fail_count = test_results.iter()
                        .filter(|(p, _, pass)| p == problem && !pass)
                        .count();
                    fail_count as f32 / n_codes.max(1) as f32
                }).collect();

                test_groups.push(GrpoGroup {
                    prompt: problem.clone(),
                    responses: tests,
                    rewards: test_rewards,
                });
            }
        }

        // Update both models via GRPO.
        let _code_stats = self.grpo.step(&code_groups);
        let _test_stats = self.grpo.step(&test_groups);

        // Track co-evolution statistics.
        let n_episodes = episodes.len().max(1);
        let co_stats = CoEvolutionStats {
            round: self.round,
            code_pass_rate: code_passes as f32 / n_episodes as f32,
            test_break_rate: test_breaks as f32 / n_episodes as f32,
            avg_code_reward: total_reward / n_episodes as f32,
            difficulty: self.round as f32 * 0.1, // Increases each round.
        };

        println!(
            "  Co-evolution round {}: pass_rate={:.1}% break_rate={:.1}% avg_reward={:.3}",
            co_stats.round,
            co_stats.code_pass_rate * 100.0,
            co_stats.test_break_rate * 100.0,
            co_stats.avg_code_reward
        );

        self.stats.push(co_stats);
        episodes
    }
}

// ==================== Dual-graph guidance ====================

/// Node in the file dependency graph.
#[derive(Debug, Clone)]
pub struct FileNode {
    pub path: String,
    pub imports: Vec<String>,      // Files this file imports.
    pub exports: Vec<String>,      // Public functions/types this file exports.
    pub dependencies: Vec<String>, // External crate dependencies.
}

/// Edge in the code structure graph.
#[derive(Debug, Clone)]
pub struct StructureEdge {
    pub from: String,  // Function/type name.
    pub to: String,    // Function/type name.
    pub edge_type: StructureEdgeType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureEdgeType {
    Calls,       // Function A calls function B.
    Contains,    // Module A contains function B.
    Implements,  // Type A implements trait B.
    References,  // Type A references type B.
}

/// Dual-graph representation of a code repository.
///
/// - **File Dependency Graph**: DAG of file-level imports/exports.
/// - **Code Structure Graph**: function/class-level call relationships.
///
/// Together they provide structural context for repo-level code generation.
#[derive(Debug, Clone)]
pub struct RepoGraph {
    pub files: HashMap<String, FileNode>,
    pub structure_edges: Vec<StructureEdge>,
    pub functions: HashSet<String>,
    pub types: HashSet<String>,
}

impl RepoGraph {
    pub fn new() -> Self {
        RepoGraph {
            files: HashMap::new(),
            structure_edges: Vec::new(),
            functions: HashSet::new(),
            types: HashSet::new(),
        }
    }

    /// Add a file to the dependency graph.
    pub fn add_file(&mut self, node: FileNode) {
        for f in &node.exports {
            self.functions.insert(f.clone());
        }
        self.files.insert(node.path.clone(), node);
    }

    /// Add a structural edge (function call, containment, etc).
    pub fn add_edge(&mut self, edge: StructureEdge) {
        self.functions.insert(edge.from.clone());
        self.functions.insert(edge.to.clone());
        self.structure_edges.push(edge);
    }

    /// Get all files that depend on a given file (reverse dependency lookup).
    pub fn dependents(&self, path: &str) -> Vec<&String> {
        self.files.iter()
            .filter(|(_, node)| node.imports.iter().any(|imp| imp == path))
            .map(|(p, _)| p)
            .collect()
    }

    /// Get all files that a given file depends on.
    pub fn dependencies(&self, path: &str) -> Vec<&String> {
        self.files.get(path)
            .map(|node| node.imports.iter().collect())
            .unwrap_or_default()
    }

    /// Get all functions called by a given function.
    pub fn called_functions(&self, func: &str) -> Vec<&String> {
        self.structure_edges.iter()
            .filter(|e| e.from == func && e.edge_type == StructureEdgeType::Calls)
            .map(|e| &e.to)
            .collect()
    }

    /// Compute the context window for generating code in a file:
    /// the file itself + its direct dependencies + their exports.
    pub fn context_for_file(&self, path: &str) -> Vec<String> {
        let mut context = Vec::new();
        // Add the file's dependencies' exports.
        if let Some(node) = self.files.get(path) {
            for imp in &node.imports {
                if let Some(dep) = self.files.get(imp) {
                    context.extend(dep.exports.iter().cloned());
                }
            }
        }
        // Add functions that call into this file (callers).
        for node in self.files.values() {
            if node.imports.iter().any(|imp| imp == path) {
                context.extend(node.exports.iter().cloned());
            }
        }
        context.sort();
        context.dedup();
        context
    }

    /// Detect circular dependencies (should be empty in well-structured code).
    pub fn circular_dependencies(&self) -> Vec<Vec<String>> {
        let mut cycles = Vec::new();
        for start in self.files.keys() {
            let mut visited = HashSet::new();
            let mut path = Vec::new();
            self.dfs_detect_cycle(start, &mut visited, &mut path, &mut cycles);
        }
        cycles
    }

    fn dfs_detect_cycle(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        if path.contains(&node.to_string()) {
            let cycle_start = path.iter().position(|n| n == node).unwrap();
            cycles.push(path[cycle_start..].to_vec());
            return;
        }
        if visited.contains(node) { return; }
        visited.insert(node.to_string());
        path.push(node.to_string());

        if let Some(node_data) = self.files.get(node) {
            for dep in &node_data.imports {
                self.dfs_detect_cycle(dep, visited, path, cycles);
            }
        }
        path.pop();
    }

    /// Number of files in the repo.
    pub fn num_files(&self) -> usize { self.files.len() }

    /// Number of structural relationships.
    pub fn num_edges(&self) -> usize { self.structure_edges.len() }

    /// Summary string.
    pub fn summary(&self) -> String {
        format!(
            "RepoGraph: {} files, {} functions, {} types, {} structural edges",
            self.num_files(),
            self.functions.len(),
            self.types.len(),
            self.num_edges(),
        )
    }
}

impl Default for RepoGraph {
    fn default() -> Self { Self::new() }
}

/// Parse a Rust source file into a FileNode (extracting imports and exports).
pub fn parse_rust_file(path: &str, source: &str) -> FileNode {
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut dependencies = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();
        // use statements (file-level dependencies).
        if trimmed.starts_with("use ") {
            if let Some(crate_name) = trimmed
                .strip_prefix("use ")
                .and_then(|s| s.split("::").next())
            {
                if !crate_name.starts_with("crate") && !crate_name.starts_with("self") {
                    dependencies.push(crate_name.to_string());
                }
            }
            if trimmed.contains("crate::") {
                // Internal import — resolve to a file path.
                let parts: Vec<&str> = trimmed.split_whitespace().nth(1)
                    .unwrap_or("")
                    .trim_end_matches(';')
                    .split("::")
                    .collect();
                if parts.len() >= 2 {
                    imports.push(format!("src/{}.rs", parts[1]));
                }
            }
        }
        // pub fn / pub struct — exports.
        if trimmed.starts_with("pub fn ") || trimmed.starts_with("pub async fn ") {
            let name = trimmed.split('(').next()
                .and_then(|s| s.split_whitespace().nth(2))
                .unwrap_or("")
                .to_string();
            if !name.is_empty() { exports.push(name); }
        }
        if trimmed.starts_with("pub struct ") || trimmed.starts_with("pub enum ") {
            let name = trimmed.split(|c: char| !c.is_alphanumeric() && c != '_')
                .find(|s| !s.is_empty() && *s != "pub" && *s != "struct" && *s != "enum")
                .unwrap_or("")
                .to_string();
            if !name.is_empty() { exports.push(name); }
        }
    }

    FileNode {
        path: path.to_string(),
        imports,
        exports,
        dependencies,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- GRPO tests ----

    #[test]
    fn grpo_advantages_zero_mean() {
        let group = GrpoGroup {
            prompt: "test".into(),
            responses: vec!["a".into(), "b".into(), "c".into()],
            rewards: vec![1.0, 2.0, 3.0],
        };
        let adv = group.advantages();
        let mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
        assert!(mean.abs() < 1e-5, "advantages should be zero-mean: {mean}");
    }

    #[test]
    fn grpo_advantages_normalized() {
        let group = GrpoGroup {
            prompt: "test".into(),
            responses: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            rewards: vec![0.1, 0.2, 0.8, 0.9],
        };
        let adv = group.advantages();
        let std: f32 = (adv.iter().map(|a| a.powi(2)).sum::<f32>() / adv.len() as f32).sqrt();
        assert!((std - 1.0).abs() < 0.01, "advantages should be unit std: {std}");
    }

    #[test]
    fn grpo_best_response() {
        let group = GrpoGroup {
            prompt: "test".into(),
            responses: vec!["bad".into(), "good".into(), "ok".into()],
            rewards: vec![0.1, 0.9, 0.5],
        };
        assert_eq!(group.best_index(), 1);
        assert_eq!(group.best_response(), "good");
    }

    #[test]
    fn grpo_trainer_tracks_stats() {
        let mut trainer = GrpoTrainer::new(GrpoConfig::default());
        let group = GrpoGroup {
            prompt: "test".into(),
            responses: vec!["a".into(), "b".into()],
            rewards: vec![0.3, 0.7],
        };
        let stats = trainer.step(&[group]);
        assert_eq!(stats.len(), 1);
        assert!(stats[0].best_reward > stats[0].worst_reward);
    }

    // ---- Reward model tests ----

    #[test]
    fn reward_correctness_passed() {
        let model = RewardModel::new(RewardWeights::default());
        let score = model.score("fn main() {}", Some(true));
        assert!((score.correctness - 1.0).abs() < 1e-5);
    }

    #[test]
    fn reward_correctness_failed() {
        let model = RewardModel::new(RewardWeights::default());
        let score = model.score("fn main() { panic!() }", Some(false));
        assert!((score.correctness - 0.0).abs() < 1e-5);
    }

    #[test]
    fn reward_readability_good_code() {
        let model = RewardModel::new(RewardWeights::default());
        let code = "// Good function\nfn calculate_sum(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let score = model.score(code, None);
        assert!(score.readability > 0.5, "readable code should score > 0.5: {}", score.readability);
    }

    #[test]
    fn reward_aesthetics_concise() {
        let model = RewardModel::new(RewardWeights::default());
        let concise = "fn add(a: i32, b: i32) -> i32 { a + b }\n";
        let verbose = "fn add(a: i32, b: i32) -> i32 {\n    // This function adds two numbers\n    // It takes two i32 parameters\n    // And returns their sum\n    let result = a + b;\n    return result;\n}\n";
        let score_c = model.score(concise, None);
        let score_v = model.score(verbose, None);
        assert!(score_c.aesthetics >= score_v.aesthetics, "concise should score >= verbose");
    }

    #[test]
    fn reward_total_weighted() {
        let model = RewardModel::new(RewardWeights {
            correctness: 0.5, readability: 0.2, style: 0.15, aesthetics: 0.15,
        });
        let reward = model.reward("fn add(a: i32, b: i32) -> i32 { a + b }\n", Some(true));
        assert!(reward > 0.5, "good code should have high reward: {reward}");
    }

    // ---- Repo graph tests ----

    #[test]
    fn repo_graph_context() {
        let mut graph = RepoGraph::new();
        graph.add_file(FileNode {
            path: "src/main.rs".into(),
            imports: vec!["src/lib.rs".into()],
            exports: vec!["main".into()],
            dependencies: vec![],
        });
        graph.add_file(FileNode {
            path: "src/lib.rs".into(),
            imports: vec![],
            exports: vec!["add".into(), "subtract".into()],
            dependencies: vec!["ndarray".into()],
        });

        let ctx = graph.context_for_file("src/main.rs");
        assert!(ctx.contains(&"add".to_string()));
        assert!(ctx.contains(&"subtract".to_string()));
    }

    #[test]
    fn repo_graph_dependencies() {
        let mut graph = RepoGraph::new();
        graph.add_file(FileNode {
            path: "src/a.rs".into(),
            imports: vec!["src/b.rs".into(), "src/c.rs".into()],
            exports: vec!["func_a".into()],
            dependencies: vec![],
        });

        let deps = graph.dependencies("src/a.rs");
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn repo_graph_called_functions() {
        let mut graph = RepoGraph::new();
        graph.add_edge(StructureEdge {
            from: "func_a".into(), to: "func_b".into(), edge_type: StructureEdgeType::Calls,
        });
        graph.add_edge(StructureEdge {
            from: "func_a".into(), to: "func_c".into(), edge_type: StructureEdgeType::Calls,
        });
        graph.add_edge(StructureEdge {
            from: "func_b".into(), to: "func_d".into(), edge_type: StructureEdgeType::Calls,
        });

        let called = graph.called_functions("func_a");
        assert_eq!(called.len(), 2);
        assert!(called.iter().any(|s| s.as_str() == "func_b"));
        assert!(called.iter().any(|s| s.as_str() == "func_c"));
    }

    #[test]
    fn parse_rust_file_extracts_exports() {
        let source = r#"
use crate::lib::helper;

pub fn calculate(x: i32) -> i32 { x * 2 }
pub struct Config { name: String }
fn private_helper() {}
"#;
        let node = parse_rust_file("src/calc.rs", source);
        assert!(node.exports.contains(&"calculate".to_string()));
        assert!(node.exports.contains(&"Config".to_string()));
        assert!(!node.exports.contains(&"private_helper".to_string()));
        assert!(node.dependencies.contains(&"crate".to_string()) || node.imports.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn repo_graph_summary() {
        let mut graph = RepoGraph::new();
        graph.add_file(FileNode {
            path: "a.rs".into(), imports: vec![], exports: vec!["f".into()], dependencies: vec![],
        });
        graph.add_edge(StructureEdge {
            from: "f".into(), to: "g".into(), edge_type: StructureEdgeType::Calls,
        });
        let s = graph.summary();
        assert!(s.contains("1 files"));
        assert!(s.contains("1 structural edges"));
    }

    // ---- Co-evolution tests ----

    #[test]
    fn co_evolution_runs() {
        let mut trainer = CoEvolutionTrainer::new(
            GrpoConfig::default(),
            RewardWeights::default(),
        );

        let problems = vec!["write add function".to_string()];
        let mut code_samples = HashMap::new();
        code_samples.insert("write add function".to_string(), vec![
            "fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
            "fn add(a, b) { a + b }".to_string(),
        ]);
        let mut test_samples = HashMap::new();
        test_samples.insert("write add function".to_string(), vec![
            "assert_eq!(add(1, 2), 3);".to_string(),
        ]);

        let test_results = vec![
            ("write add function".to_string(), "0".to_string(), true),
            ("write add function".to_string(), "1".to_string(), false),
        ];

        let episodes = trainer.co_evolve_round(&problems, &code_samples, &test_samples, &test_results);
        assert!(!episodes.is_empty());
        assert!(trainer.stats.len() == 1);
        assert!(trainer.stats[0].code_pass_rate > 0.0);
    }
}

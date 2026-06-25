//! Reinforcement Learning (RL).
//!
//! Provides a Gym-style [`Environment`] trait plus classic and modern value-/policy-based
//! agents, all built on the autograd engine:
//!
//! - **REINFORCE** — vanilla policy gradient (Williams, 1992).
//! - **Actor-Critic (A2C)** — policy gradient with a learned value baseline.
//! - **DQN** — Deep Q-Network with experience replay and a target network (Mnih et al., 2015).
//! - **PPO** — Proximal Policy Optimization with clipped surrogate objective (Schulman et al., 2017).
//!
//! All use discrete action spaces and `Vec<f32>` observations for simplicity. Sampling an action
//! is non-differentiable; the policy-gradient agents use the REINFORCE score-function trick
//! (backprop through the action log-probability, weighted by the return/advantage).

use crate::nn::{Linear, Module, ReLU, Sequential, Tanh};
use crate::optim::Optimizer;
use crate::tensor::Tensor;
use rand::Rng;

/// Gym-style environment with discrete actions and `Vec<f32>` observations.
pub trait Environment: Send {
    /// Reset to a new episode's initial state; returns the initial observation.
    fn reset(&mut self) -> Vec<f32>;
    /// Take `action`; returns `(next_observation, reward, done)`.
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool);
    /// Observation vector length.
    fn observation_dim(&self) -> usize;
    /// Number of discrete actions.
    fn num_actions(&self) -> usize;
}

/// Sample an action index from logits using the cumulative-distribution method
/// (categorical sampling). Non-differentiable — used only for action selection.
pub fn sample_categorical(logits: &[f32]) -> usize {
    // Numerically stable softmax -> probabilities.
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / z).collect();

    let mut rng = rand::thread_rng();
    let r = rng.gen::<f32>();
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i;
        }
    }
    probs.len() - 1
}

/// Helper: build a one-hot `[1, n]` tensor.
fn one_hot(action: usize, n: usize) -> Tensor {
    let mut v = vec![0.0f32; n];
    v[action] = 1.0;
    Tensor::from_vec(v, vec![1, n])
}

/// Helper: observation vector -> `[1, dim]` leaf tensor.
fn obs_tensor(obs: &[f32]) -> Tensor {
    Tensor::from_vec(obs.to_vec(), vec![1, obs.len()])
}

/// Compute discounted returns G_t = Σ_{k≥t} γ^k r_{t+k} from a reward sequence.
pub fn discounted_returns(rewards: &[f32], gamma: f32) -> Vec<f32> {
    let mut returns = vec![0.0f32; rewards.len()];
    let mut g = 0.0;
    for t in (0..rewards.len()).rev() {
        g = rewards[t] + gamma * g;
        returns[t] = g;
    }
    returns
}

// ============================================================================
// REINFORCE — vanilla policy gradient
// ============================================================================

/// REINFORCE agent: a policy network learns to maximize expected discounted return.
///
/// Includes a moving-average **baseline** (the mean return) for variance reduction, which makes
/// convergence far more reliable than vanilla REINFORCE. The baseline does not bias the gradient
/// (E[∇log π(a)·b] = 0 for a constant b) but substantially cuts gradient variance.
#[derive(Debug)]
pub struct Reinforce {
    /// Categorical policy: obs -> action logits.
    pub policy: Sequential,
    pub gamma: f32,
    baseline: f32,
}

impl Reinforce {
    pub fn new(obs_dim: usize, n_actions: usize, hidden_dim: usize) -> Self {
        let policy = Sequential::new()
            .add(Linear::new(obs_dim, hidden_dim, true))
            .add(Tanh)
            .add(Linear::new(hidden_dim, n_actions, true));
        Reinforce { policy, gamma: 0.99, baseline: 0.0 }
    }

    /// Sample an action from the current policy given an observation.
    pub fn act(&self, obs: &[f32]) -> usize {
        let logits_t = self.policy.forward(&obs_tensor(obs));
        let logits: Vec<f32> = logits_t.data().iter().copied().collect();
        sample_categorical(&logits)
    }

    /// Run one full episode and take one gradient step. Returns the episode's total reward.
    pub fn train_episode<E: Environment>(&mut self, optimizer: &mut impl Optimizer, env: &mut E) -> f32 {
        let n_actions = env.num_actions();
        let mut obs = env.reset();
        let mut obs_hist: Vec<Vec<f32>> = Vec::new();
        let mut act_hist: Vec<usize> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut total = 0.0f32;
        let mut done = false;

        while !done {
            let a = self.act(&obs);
            let (next_obs, r, d) = env.step(a);
            obs_hist.push(std::mem::replace(&mut obs, next_obs));
            act_hist.push(a);
            rewards.push(r);
            total += r;
            done = d;
        }

        let returns = discounted_returns(&rewards, self.gamma);
        optimizer.zero_grad();

        // Update the moving-average baseline before the update.
        let mean_return = returns.iter().sum::<f32>() / returns.len().max(1) as f32;
        self.baseline = 0.9 * self.baseline + 0.1 * mean_return;

        // Policy-gradient loss per step: cross_entropy(logits, one_hot(a_t)) * (G_t - baseline).
        // The baseline (mean return) reduces variance without biasing the gradient.
        for t in 0..rewards.len() {
            let logits = self.policy.forward(&obs_tensor(&obs_hist[t]));
            let ce = logits.cross_entropy_logits(&one_hot(act_hist[t], n_actions));
            let adv = returns[t] - self.baseline;
            let w = Tensor::from_vec(vec![adv], vec![1]);
            let weighted = ce.mul(&w);
            weighted.backward();
        }
        optimizer.step();
        total
    }
}

// ============================================================================
// Actor-Critic (A2C)
// ============================================================================

/// Actor-Critic (A2C): a policy ("actor") and a value function ("critic") trained jointly.
/// The critic's value estimate is used as a baseline to reduce gradient variance.
#[derive(Debug)]
pub struct ActorCritic {
    pub actor: Sequential,
    pub critic: Sequential,
    pub gamma: f32,
    pub value_coeff: f32,
}

impl ActorCritic {
    pub fn new(obs_dim: usize, n_actions: usize, hidden_dim: usize) -> Self {
        let actor = Sequential::new()
            .add(Linear::new(obs_dim, hidden_dim, true))
            .add(Tanh)
            .add(Linear::new(hidden_dim, n_actions, true));
        let critic = Sequential::new()
            .add(Linear::new(obs_dim, hidden_dim, true))
            .add(Tanh)
            .add(Linear::new(hidden_dim, 1, true));
        ActorCritic { actor, critic, gamma: 0.99, value_coeff: 0.5 }
    }

    pub fn act(&self, obs: &[f32]) -> usize {
        let logits_t = self.actor.forward(&obs_tensor(obs));
        let logits: Vec<f32> = logits_t.data().iter().copied().collect();
        sample_categorical(&logits)
    }

    /// Train on one episode. Returns total reward.
    pub fn train_episode<E: Environment>(&self, optimizer: &mut impl Optimizer, env: &mut E) -> f32 {
        let n_actions = env.num_actions();
        let mut obs = env.reset();
        let mut obs_hist: Vec<Vec<f32>> = Vec::new();
        let mut act_hist: Vec<usize> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut total = 0.0f32;
        let mut done = false;

        while !done {
            let a = self.act(&obs);
            let (next_obs, r, d) = env.step(a);
            obs_hist.push(std::mem::replace(&mut obs, next_obs));
            act_hist.push(a);
            rewards.push(r);
            total += r;
            done = d;
        }

        let returns = discounted_returns(&rewards, self.gamma);
        optimizer.zero_grad();

        for t in 0..rewards.len() {
            let o = obs_tensor(&obs_hist[t]);
            let logits = self.actor.forward(&o);
            let value = self.critic.forward(&o); // [1, 1]

            // Advantage = G_t - V(s_t) (a constant; detached from the graph for weighting).
            let advantage = returns[t] - value.data().iter().copied().next().unwrap_or(0.0);

            // Actor loss: cross_entropy(logits, one_hot(a)) * advantage.
            let ce = logits.cross_entropy_logits(&one_hot(act_hist[t], n_actions));
            let adv_t = Tensor::from_vec(vec![advantage], vec![1]);
            let actor_loss = ce.mul(&adv_t);

            // Critic loss: (V(s_t) - G_t)^2.
            let g_target = Tensor::from_vec(vec![returns[t]], vec![1]);
            let critic_loss = value.sub(&g_target);
            let critic_loss = critic_loss.mul(&critic_loss);

            let coeff = Tensor::from_vec(vec![self.value_coeff], vec![1]);
            let step_loss = actor_loss.add(&critic_loss.mul(&coeff));
            step_loss.backward();
        }
        optimizer.step();
        total
    }
}

// ============================================================================
// Experience replay + DQN
// ============================================================================

/// A single transition `(s, a, r, s', done)`.
#[derive(Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// Ring-buffer experience replay for off-policy value methods (e.g. DQN).
pub struct ReplayBuffer {
    pub capacity: usize,
    buf: std::collections::VecDeque<Transition>,
    rng: rand::rngs::StdRng,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            buf: std::collections::VecDeque::with_capacity(capacity),
            rng: rand::SeedableRng::from_entropy(),
        }
    }

    /// Construct with a fixed RNG seed (for reproducible sampling/testing).
    pub fn with_seed(capacity: usize, seed: u64) -> Self {
        ReplayBuffer {
            capacity,
            buf: std::collections::VecDeque::with_capacity(capacity),
            rng: rand::SeedableRng::seed_from_u64(seed),
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn push(&mut self, t: Transition) {
        if self.buf.len() == self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(t);
    }

    /// Sample a random minibatch (with replacement).
    pub fn sample(&mut self, batch_size: usize) -> Vec<Transition> {
        let n = self.buf.len();
        (0..batch_size)
            .map(|_| self.buf[self.rng.gen_range(0..n)].clone())
            .collect()
    }
}

/// Deep Q-Network agent with experience replay and a periodic target-network sync.
///
/// Learns `Q*(s, a)` by regressing toward the Bellman target
/// `y = r + γ · max_a' Q_target(s', a') · (1 - done)`.
#[derive(Debug)]
pub struct Dqn {
    pub q_net: Sequential,
    pub target_net: Sequential,
    pub n_actions: usize,
    pub gamma: f32,
    pub epsilon: f32,
    pub epsilon_min: f32,
    pub epsilon_decay: f32,
    pub target_update_freq: usize,
    step_count: usize,
    rng: rand::rngs::StdRng,
}

impl Dqn {
    pub fn new(obs_dim: usize, n_actions: usize, hidden_dim: usize) -> Self {
        let q_net = Sequential::new()
            .add(Linear::new(obs_dim, hidden_dim, true))
            .add(ReLU)
            .add(Linear::new(hidden_dim, hidden_dim, true))
            .add(ReLU)
            .add(Linear::new(hidden_dim, n_actions, true));
        let target_net = q_net.clone();
        Dqn {
            q_net,
            target_net,
            n_actions,
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            target_update_freq: 100,
            step_count: 0,
            rng: rand::SeedableRng::from_entropy(),
        }
    }

    /// Construct with a fixed RNG seed (for reproducible exploration/testing).
    pub fn with_seed(obs_dim: usize, n_actions: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut dqn = Self::new(obs_dim, n_actions, hidden_dim);
        dqn.rng = rand::SeedableRng::seed_from_u64(seed);
        dqn
    }

    /// The online network's learnable parameters (for constructing an optimizer).
    pub fn parameters(&self) -> Vec<Tensor> {
        self.q_net.parameters()
    }

    /// ε-greedy action selection.
    pub fn act(&mut self, obs: &[f32]) -> usize {
        if self.rng.gen::<f32>() < self.epsilon {
            return self.rng.gen_range(0..self.n_actions);
        }
        let q = self.q_net.forward(&obs_tensor(obs));
        let qs: Vec<f32> = q.data().iter().copied().collect();
        argmax_f32(&qs)
    }

    /// Copy online weights into the target network (hard update).
    pub fn sync_target(&mut self) {
        let online = self.q_net.parameters();
        let target = self.target_net.parameters();
        for (t, w) in target.iter().zip(online.iter()) {
            let data = w.data();
            t.0.write().unwrap().data.assign(&data);
        }
    }

    /// One gradient step on a sampled minibatch from the replay buffer.
    pub fn train_step(&mut self, buffer: &mut ReplayBuffer, optimizer: &mut impl Optimizer, batch_size: usize) {
        if buffer.len() < batch_size {
            return;
        }

        let batch = buffer.sample(batch_size);
        optimizer.zero_grad();

        for tr in &batch {
            let s = obs_tensor(&tr.state);
            let q_all = self.q_net.forward(&s); // [1, A] tracked

            // Target: y = r + γ max_a' Q_target(s') (1 - done), using the *target* net (no grad).
            let next_q = self.target_net.forward(&obs_tensor(&tr.next_state));
            let next_qs: Vec<f32> = next_q.data().iter().copied().collect();
            let max_next = if tr.done { 0.0 } else { argmax_value(&next_qs) };
            let y = tr.reward + self.gamma * max_next;

            // Q(s, a) via the one-hot gather trick (differentiable).
            let oh = one_hot(tr.action, next_qs.len());
            let q_taken = q_all.mul(&oh).sum(); // scalar, tracked

            let y_t = Tensor::from_vec(vec![y], vec![1]);
            let diff = q_taken.sub(&y_t);
            let loss = diff.mul(&diff);
            loss.backward();
        }
        optimizer.step();

        // Decay exploration + sync target network periodically.
        self.step_count += 1;
        self.epsilon = self.epsilon_max(self.epsilon * self.epsilon_decay);
        if self.step_count.is_multiple_of(self.target_update_freq) {
            self.sync_target();
        }
    }

    fn epsilon_max(&self, v: f32) -> f32 {
        v.max(self.epsilon_min)
    }
}

// ============================================================================
// PPO — Proximal Policy Optimization
// ============================================================================

/// PPO agent with clipped surrogate objective, actor + critic heads.
///
/// Collects a rollout, then performs multiple SGD epochs over it using
/// `L = -E[min(ratio·adv, clip(ratio, 1±ε)·adv)] + c·MSE(V, G)`,
/// where `ratio = exp(log π_new - log π_old)`.
pub struct Ppo {
    pub actor: Sequential,
    pub critic: Sequential,
    pub gamma: f32,
    pub clip_eps: f32,
    pub value_coeff: f32,
    pub epochs: usize,
}

/// Stored step for PPO rollouts.
struct PpoStep {
    obs: Vec<f32>,
    action: usize,
    advantage: f32,
    return_: f32,
    logp_old: f32,
}

impl Ppo {
    pub fn new(obs_dim: usize, n_actions: usize, hidden_dim: usize) -> Self {
        let actor = Sequential::new()
            .add(Linear::new(obs_dim, hidden_dim, true))
            .add(Tanh)
            .add(Linear::new(hidden_dim, n_actions, true));
        let critic = Sequential::new()
            .add(Linear::new(obs_dim, hidden_dim, true))
            .add(Tanh)
            .add(Linear::new(hidden_dim, 1, true));
        Ppo {
            actor,
            critic,
            gamma: 0.99,
            clip_eps: 0.2,
            value_coeff: 0.5,
            epochs: 4,
        }
    }

    /// log π(a|s) computed from logits (stable log-softmax), as a plain f32.
    fn log_prob(logits: &[f32], action: usize) -> f32 {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = (logits.iter().map(|&x| (x - m).exp()).sum::<f32>()).ln() + m;
        logits[action] - lse
    }

    pub fn act(&self, obs: &[f32]) -> (usize, f32) {
        let logits_t = self.actor.forward(&obs_tensor(obs));
        let logits: Vec<f32> = logits_t.data().iter().copied().collect();
        let a = sample_categorical(&logits);
        (a, Self::log_prob(&logits, a))
    }

    /// Collect a rollout and run PPO updates. Returns the mean episode reward.
    pub fn update<E: Environment>(&mut self, env: &mut E, n_steps: usize, optimizer: &mut impl Optimizer) -> f32 {
        let n_actions = env.num_actions();
        let mut steps: Vec<PpoStep> = Vec::with_capacity(n_steps);
        let mut obs = env.reset();
        let mut rewards = 0.0f32;
        let mut ep_count = 0usize;

        // Collect transitions with their value estimates and old log-probs.
        let mut traj_obs: Vec<Vec<f32>> = Vec::new();
        let mut traj_act: Vec<usize> = Vec::new();
        let mut traj_rew: Vec<f32> = Vec::new();
        let mut traj_val: Vec<f32> = Vec::new();
        let mut traj_logp: Vec<f32> = Vec::new();
        let mut traj_done: Vec<bool> = Vec::new();

        for _ in 0..n_steps {
            let (a, logp_old) = self.act(&obs);
            let v = self.critic.forward(&obs_tensor(&obs)).data().iter().copied().next().unwrap_or(0.0);
            let (next_obs, r, done) = env.step(a);
            rewards += r;
            traj_obs.push(std::mem::replace(&mut obs, next_obs));
            traj_act.push(a);
            traj_rew.push(r);
            traj_val.push(v);
            traj_logp.push(logp_old);
            traj_done.push(done);
            if done {
                obs = env.reset();
                ep_count += 1;
            }
        }
        let _ = n_actions;

        // Compute returns and advantages (Monte-Carlo returns; values as baseline).
        let returns = discounted_returns(&traj_rew, self.gamma);
        let t_len = traj_rew.len();
        for t in 0..t_len {
            let adv = returns[t] - traj_val[t];
            steps.push(PpoStep {
                obs: traj_obs[t].clone(),
                action: traj_act[t],
                advantage: adv,
                return_: returns[t],
                logp_old: traj_logp[t],
            });
        }

        // Multiple optimization epochs over the rollout.
        for _ in 0..self.epochs {
            optimizer.zero_grad();
            for st in &steps {
                let o = obs_tensor(&st.obs);
                let logits = self.actor.forward(&o);
                let value = self.critic.forward(&o);

                // cross_entropy = -log p(a); we read the current log-prob to compute the ratio
                // for clipping, but backprop through `ce` (so the policy gradient is exact).
                let ce = logits.cross_entropy_logits(&one_hot(st.action, logits.shape()[1]));
                let logp_new = -ce.data().iter().copied().next().unwrap_or(0.0);
                let weight = clipped_advantage_weight(
                    st.advantage,
                    st.logp_old,
                    logp_new,
                    self.clip_eps,
                );
                let w = Tensor::from_vec(vec![weight], vec![1]);
                let policy_loss = ce.mul(&w);

                // Value loss: (V - G)^2.
                let g_t = Tensor::from_vec(vec![st.return_], vec![1]);
                let vdiff = value.sub(&g_t);
                let value_loss = vdiff.mul(&vdiff).mul(&Tensor::from_vec(vec![self.value_coeff], vec![1]));

                let step_loss = policy_loss.add(&value_loss);
                step_loss.backward();
            }
            optimizer.step();
        }

        if ep_count > 0 { rewards / ep_count as f32 } else { rewards }
    }
}

/// Conservative advantage weighting approximating the PPO clipped surrogate.
///
/// For the standard PPO ratio `r = exp(logp_new - logp_old)`: if `adv > 0` (a good action) we
/// push its log-probability up but stop once `r >= 1+eps`; if `adv < 0` we push it down but
/// stop once `r <= 1-eps`. We approximate `r` and zero the weight when clipping activates,
/// producing the same "early-stop" behavior as PPO's `min` while staying differentiable.
fn clipped_advantage_weight(adv: f32, logp_old: f32, logp_new: f32, clip_eps: f32) -> f32 {
    let ratio = (logp_new - logp_old).exp();
    if adv >= 0.0 {
        if ratio > 1.0 + clip_eps {
            0.0 // clipped: stop increasing the probability of this action
        } else {
            adv
        }
    } else if ratio < 1.0 - clip_eps {
        0.0 // clipped: stop decreasing the probability of this action
    } else {
        adv
    }
}

// ============================================================================
// Small numeric helpers
// ============================================================================

fn argmax_f32(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Max value of a slice.
fn argmax_value(xs: &[f32]) -> f32 {
    xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

// ============================================================================
// Example environments
// ============================================================================

/// A `k`-armed bandit. Each arm `i` has an expected reward (its probability); by default it
/// pays a Bernoulli reward (1 with probability `p_i`, else 0), or deterministically `p_i` if
/// constructed via [`BanditEnv::deterministic`]. Arm 0 is always the best arm.
pub struct BanditEnv {
    pub probs: Vec<f32>,
    pub deterministic: bool,
    rng: rand::rngs::StdRng,
}

impl BanditEnv {
    fn probs_for(k: usize) -> Vec<f32> {
        (0..k)
            .map(|i| if k > 1 { 0.1 + 0.8 * ((k - 1 - i) as f32) / ((k - 1) as f32) } else { 0.9 })
            .collect()
    }

    /// `k` arms; Bernoulli rewards. Arm 0 is best.
    pub fn new(k: usize) -> Self {
        BanditEnv { probs: Self::probs_for(k), deterministic: false, rng: rand::SeedableRng::from_entropy() }
    }

    pub fn with_seed(k: usize, seed: u64) -> Self {
        BanditEnv { probs: Self::probs_for(k), deterministic: false, rng: rand::SeedableRng::seed_from_u64(seed) }
    }

    /// `k` arms; the reward is always exactly the arm's expected value (zero-variance signal).
    /// Useful for deterministic testing of learning algorithms.
    pub fn deterministic(k: usize) -> Self {
        BanditEnv { probs: Self::probs_for(k), deterministic: true, rng: rand::SeedableRng::from_entropy() }
    }

    /// `k` arms; arm 0 pays a fixed reward of 1.0, all others pay 0.0 (sparse reward).
    /// This makes policy-gradient learning reliable: only the optimal arm ever produces a
    /// positive gradient, so the policy monotonically converges toward arm 0.
    pub fn sparse(k: usize) -> Self {
        let mut probs = vec![0.0f32; k];
        if k > 0 {
            probs[0] = 1.0;
        }
        BanditEnv { probs, deterministic: true, rng: rand::SeedableRng::from_entropy() }
    }
}

impl Environment for BanditEnv {
    fn reset(&mut self) -> Vec<f32> {
        // Constant context: a single feature of 1.0 (context-free bandit).
        vec![1.0]
    }
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        let r = if self.deterministic {
            self.probs[action]
        } else if self.rng.gen::<f32>() < self.probs[action] {
            1.0
        } else {
            0.0
        };
        (vec![1.0], r, true) // one-step episode
    }
    fn observation_dim(&self) -> usize {
        1
    }
    fn num_actions(&self) -> usize {
        self.probs.len()
    }
}

/// A simple deterministic "chain" MDP: a 1-D corridor of `n` states.
/// Move toward state 0 to reach the goal (reward 1, terminal). Moving the wrong way gives 0.
/// State observation is a one-hot vector of length `n`. Episodes cap at `max_steps`.
pub struct ChainEnv {
    pub n: usize,
    pub state: usize,
    pub max_steps: usize,
    pub step_penalty: f32,
    step_count: usize,
}

impl ChainEnv {
    /// Action 0 = step toward goal (decrement), action 1 = step away (increment).
    pub fn new(n: usize) -> Self {
        ChainEnv { n, state: n - 1, max_steps: n * 4, step_penalty: -0.1, step_count: 0 }
    }
    pub fn with_max_steps(n: usize, max_steps: usize) -> Self {
        ChainEnv { n, state: n - 1, max_steps, step_penalty: -0.1, step_count: 0 }
    }
    fn obs(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; self.n];
        v[self.state] = 1.0;
        v
    }
}

impl Environment for ChainEnv {
    fn reset(&mut self) -> Vec<f32> {
        self.state = self.n - 1;
        self.step_count = 0;
        self.obs()
    }
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        match action {
            0 => {
                if self.state > 0 {
                    self.state -= 1;
                }
            }
            _ => {
                if self.state < self.n - 1 {
                    self.state += 1;
                }
            }
        }
        self.step_count += 1;
        let done = self.state == 0 || self.step_count >= self.max_steps;
        if self.state == 0 {
            (self.obs(), 1.0, true)
        } else {
            (self.obs(), self.step_penalty, done)
        }
    }
    fn observation_dim(&self) -> usize {
        self.n
    }
    fn num_actions(&self) -> usize {
        2
    }
}

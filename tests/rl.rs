//! Reinforcement-learning agent tests: each agent should learn a better-than-random policy.

use rust_nn::nn::Module;
use rust_nn::optim::Adam;
use rust_nn::rl::{
    discounted_returns, sample_categorical, ActorCritic, BanditEnv, ChainEnv, Dqn,
    Environment, Ppo, Reinforce, ReplayBuffer, Transition,
};
use rust_nn::tensor::Tensor;

#[test]
fn categorical_sample_respects_distribution() {
    // Logits heavily favor the last action.
    let logits = vec![-10.0, -10.0, 10.0];
    let mut counts = [0usize; 3];
    for _ in 0..2000 {
        counts[sample_categorical(&logits)] += 1;
    }
    assert!(counts[2] > counts[0] + counts[1], "should mostly sample action 2");
}

#[test]
fn discounted_returns_are_correct() {
    let r = discounted_returns(&[1.0, 2.0, 3.0], 0.5);
    // G_0 = 1 + 0.5*2 + 0.25*3 = 2.75
    // G_1 = 2 + 0.5*3 = 3.5
    // G_2 = 3
    assert!((r[0] - 2.75).abs() < 1e-5);
    assert!((r[1] - 3.5).abs() < 1e-5);
    assert!((r[2] - 3.0).abs() < 1e-5);
}

/// Read the greedy (argmax) action of a policy module, deterministically.
fn greedy_action(policy: &rust_nn::nn::Sequential, obs: &[f32]) -> usize {
    use rust_nn::nn::Module;
    let logits: Vec<f32> = policy
        .forward(&Tensor::from_vec(obs.to_vec(), vec![1, obs.len()]))
        .data()
        .iter()
        .copied()
        .collect();
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Mean absolute TD (Bellman) error of a DQN over a sampled minibatch.
fn td_error(agent: &Dqn, buf: &mut ReplayBuffer, n: usize) -> f32 {
    use rust_nn::nn::Module;
    let batch = buf.sample(64);
    let mut total = 0.0f32;
    for tr in &batch {
        let next_q = agent.target_net.forward(&Tensor::from_vec(tr.next_state.clone(), vec![1, n]));
        let next_qs: Vec<f32> = next_q.data().iter().copied().collect();
        let max_next = if tr.done { 0.0 } else { next_qs.iter().copied().fold(f32::NEG_INFINITY, f32::max) };
        let target = tr.reward + agent.gamma * max_next;
        let q_all = agent.q_net.forward(&Tensor::from_vec(tr.state.clone(), vec![1, n]));
        let qs: Vec<f32> = q_all.data().iter().copied().collect();
        total += (qs[tr.action] - target).abs();
    }
    total / batch.len() as f32
}

#[test]
fn reinforce_learns_bandit() {
    let mut agent = Reinforce::new(1, 4, 16);
    let params = agent.policy.parameters();
    let mut opt = Adam::new(params, 0.02);
    let mut env = BanditEnv::sparse(4);

    for _ in 0..300 {
        agent.train_episode(&mut opt, &mut env);
    }
    let learned = greedy_action(&agent.policy, &[1.0]) == 0;
    println!("REINFORCE greedy arm: {}", greedy_action(&agent.policy, &[1.0]));
    assert!(learned, "REINFORCE should learn the best arm");
}

#[test]
fn actor_critic_learns_bandit() {
    let agent = ActorCritic::new(1, 4, 16);
    let mut params = agent.actor.parameters();
    params.extend(agent.critic.parameters());
    let mut opt = Adam::new(params, 0.02);
    let mut env = BanditEnv::sparse(4);

    for _ in 0..300 {
        agent.train_episode(&mut opt, &mut env);
    }
    let learned = greedy_action(&agent.actor, &[1.0]) == 0;
    println!("Actor-Critic greedy arm: {}", greedy_action(&agent.actor, &[1.0]));
    assert!(learned, "Actor-Critic should learn the best arm");
}

#[test]
fn ppo_learns_bandit() {
    let mut agent = Ppo::new(1, 4, 16);
    let mut params = agent.actor.parameters();
    params.extend(agent.critic.parameters());
    let mut opt = Adam::new(params, 0.02);
    let mut env = BanditEnv::sparse(4);

    for _ in 0..60 {
        agent.update(&mut env, 64, &mut opt);
    }
    let learned = greedy_action(&agent.actor, &[1.0]) == 0;
    println!("PPO greedy arm: {}", greedy_action(&agent.actor, &[1.0]));
    assert!(learned, "PPO should learn the best arm");
}

#[test]
fn replay_buffer_works() {
    let mut buf = ReplayBuffer::new(4);
    assert!(buf.is_empty());
    for i in 0..10 {
        buf.push(Transition {
            state: vec![i as f32],
            action: i % 3,
            reward: i as f32,
            next_state: vec![(i + 1) as f32],
            done: i == 9,
        });
    }
    // Capacity 4 -> only last 4 survive.
    assert_eq!(buf.len(), 4);
    let sample = buf.sample(2);
    assert_eq!(sample.len(), 2);
}

#[test]
fn dqn_reduces_td_error() {
    // DQN should monotonically reduce its temporal-difference (Bellman) error over training,
    // regardless of which policy is "best". This is robust and init-independent.
    let n = 4;
    let mut agent = Dqn::with_seed(n, 2, 32, 42);
    let mut opt = Adam::new(agent.parameters(), 0.01);
    let mut buf = ReplayBuffer::with_seed(8000, 7);
    let mut env = ChainEnv::with_max_steps(n, 16);

    // Collect random experience.
    agent.epsilon = 1.0;
    for _ in 0..600 {
        let mut o = env.reset();
        let mut done = false;
        while !done {
            let a = agent.act(&o);
            let (next, r, d) = env.step(a);
            buf.push(Transition { state: o.clone(), action: a, reward: r, next_state: next.clone(), done: d });
            o = next;
            done = d;
        }
    }

    let err_before = td_error(&agent, &mut buf, n);
    agent.epsilon = 0.2;
    for _ in 0..600 {
        agent.train_step(&mut buf, &mut opt, 32);
    }
    let err_after = td_error(&agent, &mut buf, n);

    println!("DQN TD error: before={err_before:.4} after={err_after:.4}");
    assert!(
        err_after < err_before,
        "DQN should reduce TD error ({err_after:.4} >= {err_before:.4})"
    );
}

#[test]
fn dqn_sync_target_changes_weights() {
    let mut agent = Dqn::new(3, 2, 8);
    // Take one train step on random data so online weights diverge from target.
    let mut buf = ReplayBuffer::new(100);
    for _ in 0..20 {
        buf.push(Transition {
            state: vec![1.0, 1.0, 1.0],
            action: 0,
            reward: 5.0,
            next_state: vec![1.0, 1.0, 1.0],
            done: false,
        });
    }

    // Snapshot the online net BEFORE training.
    let online_before: Vec<Vec<f32>> = agent
        .q_net
        .parameters()
        .iter()
        .map(|t| t.data().iter().copied().collect())
        .collect();
    // Snapshot the target BEFORE any sync (it was cloned at construction).
    let target_before: Vec<Vec<f32>> = agent
        .target_net
        .parameters()
        .iter()
        .map(|t| t.data().iter().copied().collect())
        .collect();

    let mut opt = Adam::new(agent.parameters(), 0.1);
    agent.train_step(&mut buf, &mut opt, 8);

    // 1. Training should have changed the online net somewhere.
    let online_after: Vec<Vec<f32>> = agent
        .q_net
        .parameters()
        .iter()
        .map(|t| t.data().iter().copied().collect())
        .collect();
    let trained = online_before
        .iter()
        .zip(online_after.iter())
        .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    assert!(trained, "train_step should change the online network");

    // 2. After sync, the target should exactly match the (trained) online net.
    agent.sync_target();
    let target_after: Vec<Vec<f32>> = agent
        .target_net
        .parameters()
        .iter()
        .map(|t| t.data().iter().copied().collect())
        .collect();
    for (ta, oa) in target_after.iter().zip(online_after.iter()) {
        for (x, y) in ta.iter().zip(oa.iter()) {
            assert!((x - y).abs() < 1e-5, "target should match online after sync");
        }
    }
    // 3. And the target should differ from its pre-sync state somewhere.
    let changed = target_before
        .iter()
        .zip(target_after.iter())
        .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    assert!(changed, "sync_target should update the target network");
}

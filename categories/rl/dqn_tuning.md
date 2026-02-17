# DQN Tuning

## Overview: Replay Buffer, Target Network Updates, and Epsilon Decay

Training a DQN agent involves three interconnected mechanisms that together determine learning quality and stability: the experience replay buffer, the target network update schedule, and the epsilon decay strategy. Each serves a distinct purpose, but their interactions are what ultimately shape the agent's learning dynamics. Understanding these interactions is essential for effective hyperparameter tuning.

**DQN hyperparameters:**
- Learning rate — step size for gradient updates
- Gamma — discount factor for future rewards
- Buffer size — how many transitions the replay buffer holds
- Batch size — transitions sampled per gradient update
- Learning starts — steps of pure random exploration before training begins
- Exploration initial eps — starting epsilon (1.0, fully random)
- Exploration final eps — ending epsilon (0.05, mostly greedy)
- Exploration fraction — what fraction of training is used to decay epsilon
- Target update interval — how many steps between copying Q-network to target network

This document explains each mechanism in depth, discusses how they interact, and provides practical guidance for tuning them in the context of LunarLander-v3 with Stable-Baselines3.

## 1. Experience Replay Buffer

### What It Does

The replay buffer is a fixed-size memory that stores transitions $(s, a, r, s', \text{done})$ as the agent interacts with the environment. During training, random mini-batches are sampled from this buffer to compute gradient updates. The agent does not learn from transitions in the order they were experienced — it learns from randomized historical data.

### Why It Exists

Without replay, the agent would train on consecutive transitions, which are highly correlated (state $s_{t+1}$ is nearly identical to $s_t$). Training a neural network on correlated sequential data causes it to overfit to recent experience and forget earlier knowledge. The replay buffer breaks these temporal correlations by mixing old and new transitions in each training batch.

A secondary benefit is sample efficiency: each transition can be used for multiple gradient updates rather than being discarded after one use. This is a major advantage of DQN over on-policy methods like PPO, which must discard data after each update cycle.

### Buffer Size

The buffer size determines how far back the agent can remember.

A **small buffer** (e.g., 10,000 transitions) means old experience is overwritten quickly. The agent primarily trains on recent data, which reflects its current policy. This can be beneficial early in training when old experience from a random policy is low quality. However, it also means the agent forgets useful transitions quickly and the diversity of training data is limited.

A **large buffer** (e.g., 1,000,000 transitions) retains experience from much earlier in training. This increases data diversity and decorrelation, but introduces a different problem: the buffer is dominated by transitions generated under a very different (worse) policy. The agent may spend significant gradient updates fitting to outdated experience that no longer reflects how it should behave.

### Buffer Size in Practice

For LunarLander-v3, the state space is 8-dimensional and transitions are lightweight. Memory is not a constraint. A buffer of 50,000 to 200,000 is a reasonable range. Stable-Baselines3 defaults to 1,000,000, which works but may be unnecessarily large for this environment.

The key diagnostic: if the agent learns quickly at first but then plateaus or degrades, the buffer may be too large (stale data diluting useful recent experience). If the agent is unstable and oscillates, the buffer may be too small (insufficient diversity, correlated batches).

### Learning Starts

A related parameter is `learning_starts` — how many transitions are collected before training begins. If training starts too early, the buffer contains almost exclusively random-policy experience and the initial gradient updates may push the network in unhelpful directions. A value of 1,000 to 10,000 gives the buffer enough diversity to produce meaningful initial gradients.

### Batch Size

The mini-batch size determines how many transitions are sampled per gradient update. Larger batches provide more stable gradient estimates but are computationally more expensive and may smooth over important rare transitions. Smaller batches are noisier but update more frequently. For LunarLander-v3, batch sizes of 64 to 256 are standard.

## 2. Target Network Update Frequency

### What It Does

DQN maintains two copies of the Q-network: the **online network** $\theta$ (actively trained) and the **target network** $\theta^-$ (frozen copy used to compute regression targets). The target network's weights are periodically synchronized with the online network.

The target value for a transition is:

$$y = r + \gamma \cdot \max_{a'} Q(s', a'; \theta^-)$$

Because $\theta^-$ is frozen, this target is stable between updates, providing a fixed regression objective for the online network.

### Why It Exists

Without a separate target network, the same weights would appear on both sides of the loss:

$$\mathcal{L} = \left(r + \gamma \cdot \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta)\right)^2$$

Every gradient step would change both the prediction and the target simultaneously. This creates a feedback loop: the network adjusts its predictions, which shifts the targets, which changes the gradients, which adjusts the predictions again. The result is oscillation or divergence — the network chases its own shadow.

The frozen target network breaks this loop by holding the targets approximately constant for many gradient steps, allowing the online network to make stable progress toward a temporarily fixed objective.

### Hard Updates vs. Soft Updates

**Hard updates** copy the online weights to the target network every $C$ steps: $\theta^- \leftarrow \theta$ every $C$ gradient updates (or environment steps, depending on the implementation). Between updates, the target network is completely frozen. This creates a periodic discontinuity — target values jump suddenly at each sync, then remain constant.

**Soft updates** blend the weights at every step using an exponential moving average: $\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$, where $\tau$ is small (e.g., 0.005). This provides smoother target evolution without the periodic jumps. Soft updates are used in DDPG, TD3, and SAC, but standard DQN in SB3 uses hard updates.

### Update Frequency Tuning

In Stable-Baselines3, `target_update_interval` controls how many environment steps pass between hard updates.

**Too frequent** (e.g., every 1 step): the target network closely tracks the online network, effectively eliminating its stabilizing effect. Training approaches the unstable single-network regime.

**Too infrequent** (e.g., every 50,000 steps): the target network becomes very stale. The online network optimizes against outdated targets for too long, and when the target network finally updates, the jump in target values can destabilize learning. The agent may also learn slowly because the targets don't reflect the online network's improvements.

For LunarLander-v3, values between 1,000 and 10,000 steps are typical. The default in SB3 is 10,000. If you observe oscillations in the loss curve, try increasing the interval. If learning seems too slow, try decreasing it.

### Interaction with Replay Buffer

The target network update frequency and replay buffer size are coupled. A large buffer with frequent target updates means the agent trains on old data against frequently changing targets — both sides are mismatched. A small buffer with infrequent target updates means the agent trains on recent data against stale targets — the targets lag behind the data. The healthiest regime is when the buffer is large enough for diversity but not so large that it's dominated by pre-target-update experience, and the target updates are frequent enough to stay relevant but infrequent enough to provide stability.

## 3. Epsilon Decay

### What It Does

Epsilon ($\varepsilon$) controls the agent's exploration-exploitation trade-off via the $\varepsilon$-greedy policy. At each step, the agent takes a random action with probability $\varepsilon$ and the greedy action (highest Q-value) with probability $1 - \varepsilon$.

### The Decay Schedule

$\varepsilon$ is typically annealed from a high value to a low value over the course of training:

- **Initial epsilon** ($\varepsilon_{\text{start}}$): Usually 1.0 — the agent acts completely randomly at the start, filling the replay buffer with diverse experience.
- **Final epsilon** ($\varepsilon_{\text{end}}$): Usually 0.01 to 0.05 — the agent mostly exploits its learned Q-values but retains a small chance of random exploration to avoid getting permanently stuck.
- **Exploration fraction**: The fraction of total training steps over which $\varepsilon$ decays linearly from start to end. After this fraction, $\varepsilon$ remains at its final value for the rest of training.

### Why the Schedule Matters

**Decaying too fast** (small exploration fraction): The agent commits to its Q-value estimates before they're reliable. If early Q-values are wrong (which they always are), the agent locks onto a suboptimal policy and stops exploring the state space. It converges quickly but to a poor solution.

**Decaying too slowly** (large exploration fraction): The agent keeps taking random actions long after its Q-values are informative. This wastes environment steps on uninformative random exploration, slows learning, and fills the replay buffer with low-quality transitions generated under a near-random policy.

### Epsilon and the Replay Buffer Interaction

The epsilon schedule directly affects what enters the replay buffer. During high-epsilon phases, the buffer fills with transitions from a near-random policy. During low-epsilon phases, it fills with transitions from a near-greedy policy. If the buffer is large and epsilon decays slowly, the buffer will be dominated by random-policy transitions for a long time, and the agent trains on data that doesn't reflect its current behavior. If the buffer is small, old random transitions are overwritten quickly, and the training data stays current.

This interaction is important: a very large buffer combined with slow epsilon decay means the agent is training on mostly random experience even after its Q-values have improved. A smaller buffer or faster epsilon decay keeps the training data aligned with the agent's evolving policy.

### Practical Settings for LunarLander-v3

SB3's default `exploration_fraction` is 0.1 (epsilon decays over the first 10% of training). For LunarLander-v3, values between 0.1 and 0.3 are reasonable. If the agent plateaus early, try extending the exploration fraction. If it oscillates and never converges, try shortening it.

`exploration_initial_eps` defaults to 1.0 and `exploration_final_eps` defaults to 0.05. These are sensible starting points. Lowering the final epsilon to 0.01 can help if the agent's final evaluation performance is hurt by residual random actions.

## 4. How the Three Mechanisms Interact

The three mechanisms form a coupled system. Here are the key interaction patterns:

### Early Training

Epsilon is high, so transitions in the buffer come from a near-random policy. The target network is initialized randomly, so targets are meaningless. The online network trains on random data against random targets. Learning is noisy and progress is slow — this is expected. The agent is building an initial, rough approximation of the Q-function.

**What matters here:** the buffer needs to accumulate enough diverse transitions before training starts (`learning_starts`). The target network update frequency is less critical because everything is noisy anyway.

### Mid Training

Epsilon is decreasing, so the buffer contains a mix of random and increasingly policy-driven transitions. The Q-values are becoming meaningful, and the target network provides increasingly useful supervision. This is where real learning happens.

**What matters here:** the balance between buffer staleness and target freshness. If the buffer is too large, it's still dominated by early random experience. If the target network updates too frequently, training is unstable because targets shift faster than the network can track.

### Late Training

Epsilon is at its final low value, and the buffer is mostly filled with near-optimal transitions. The Q-values are close to convergence, and small updates refine the policy.

**What matters here:** stability. The target network should update infrequently enough that the near-converged Q-values aren't disrupted. The buffer should be large enough that the agent doesn't overfit to a narrow set of recent transitions.

## 5. Summary of SB3 DQN Hyperparameters

| Parameter | SB3 Name | Default | Role | Tuning Range (LunarLander) |
|-----------|----------|---------|------|---------------------------|
| Buffer size | `buffer_size` | 1,000,000 | How much experience to store | 50,000 – 200,000 |
| Batch size | `batch_size` | 32 | Transitions per gradient update | 64 – 256 |
| Learning starts | `learning_starts` | 50,000 | Steps before training begins | 1,000 – 10,000 |
| Target update interval | `target_update_interval` | 10,000 | Steps between target network syncs | 1,000 – 10,000 |
| Initial epsilon | `exploration_initial_eps` | 1.0 | Starting exploration rate | 1.0 |
| Final epsilon | `exploration_final_eps` | 0.05 | Minimum exploration rate | 0.01 – 0.05 |
| Exploration fraction | `exploration_fraction` | 0.1 | Fraction of training for epsilon decay | 0.1 – 0.3 |
| Discount factor | `gamma` | 0.99 | Weight of future rewards | 0.99 – 0.999 |
| Learning rate | `learning_rate` | 0.0001 | Step size for gradient updates | 0.0001 – 0.001 |

## 6. Diagnostic Checklist

When training a DQN agent and observing problems, use this to identify likely causes:

**Agent doesn't learn at all:** Check `learning_starts` — training may not have begun. Check the learning rate — too low means imperceptibly slow progress, too high means divergence.

**Agent learns then collapses:** Target network may be updating too frequently (unstable targets) or the buffer may be too small (overfitting to recent experience). Try increasing `target_update_interval` or `buffer_size`.

**Agent plateaus at a mediocre reward:** Epsilon may have decayed too fast (premature exploitation). Try increasing `exploration_fraction`. Alternatively, the buffer may be too large and dominated by low-quality early experience.

**Loss curve oscillates wildly:** Target network updates too frequent, batch size too small, or learning rate too high. Try increasing `target_update_interval`, increasing `batch_size`, or decreasing `learning_rate`.

**Agent performs well during training but poorly during evaluation:** Evaluation uses a greedy policy ($\varepsilon = 0$). If training relied heavily on exploration to reach certain states, the greedy policy may avoid those paths. This suggests the Q-values for some critical states are inaccurate. More training or a larger buffer may help.

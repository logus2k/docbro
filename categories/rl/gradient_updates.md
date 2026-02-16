# Gradient Updates and Sample Efficiency: DQN vs PPO

## Overview

When the project asks you to "report the number of gradient updates performed by each agent," it's pointing at a fundamental difference between DQN and PPO. Both agents interact with the same environment for the same number of total steps, but they use those steps in radically different ways. Understanding this difference is central to the sample efficiency comparison and to explaining why the two agents learn at different speeds.

## 1. What Is a Gradient Update?

A gradient update is a single step of optimization: compute the loss on a mini-batch of data, compute the gradient of the loss with respect to the network weights, and adjust the weights by a small amount in the direction that reduces the loss. One gradient update = one call to the optimizer's `step()` function.

This is the atomic unit of learning. Everything else — collecting experience, computing advantages, sampling from the replay buffer — is preparation. The gradient update is where the network actually changes.

## 2. How DQN Performs Gradient Updates

### The Update Schedule

DQN performs **one gradient update per environment step** (after the initial `learning_starts` phase). At every step:

1. The agent takes an action, observes a transition, stores it in the replay buffer.
2. A mini-batch of `batch_size` transitions is sampled from the buffer.
3. One gradient step is computed on the TD loss.

So if you train for 500,000 environment steps with `learning_starts` = 10,000, DQN performs approximately 490,000 gradient updates.

### Data Reuse

Each transition in the replay buffer can be sampled multiple times across different mini-batches. A transition stored at step 10,000 might appear in gradient updates at step 10,500, step 15,000, and step 30,000 (until it's eventually overwritten when the buffer is full). This is the replay buffer's core benefit: each unit of experience contributes to many gradient updates.

The **replay ratio** describes how many gradient updates are performed per environment step. For standard DQN, this is 1:1 — one gradient update per step. Some variants increase this (e.g., performing 4 gradient updates per step), which improves sample efficiency at the cost of increased computation and potential overfitting to buffer contents.

### Total Gradient Updates for DQN

$$\text{DQN gradient updates} \approx \text{total\_steps} - \text{learning\_starts}$$

Each update uses `batch_size` transitions, but those transitions are resampled from the buffer and may overlap with previous mini-batches. The total number of unique transitions seen across all gradient updates is much larger than the buffer size.

## 3. How PPO Performs Gradient Updates

### The Update Schedule

PPO operates in cycles of **collection** and **optimization**, which are fundamentally different from DQN's step-by-step approach.

**Collection phase:** The agent runs its current policy for `n_steps` environment steps (across `n_envs` parallel environments), collecting a batch of `n_steps × n_envs` transitions. During this phase, no gradient updates occur — the agent just gathers data.

**Optimization phase:** The collected batch is used for `n_epochs` passes. In each epoch, the batch is shuffled and divided into mini-batches of `batch_size` transitions. One gradient step is performed per mini-batch.

Then the entire batch is discarded and the cycle repeats.

### Counting PPO Gradient Updates

The number of gradient updates per cycle is:

$$\text{updates per cycle} = n\_epochs \times \left\lceil\frac{n\_steps \times n\_envs}{batch\_size}\right\rceil$$

The number of cycles in total training is:

$$\text{cycles} = \frac{\text{total\_steps}}{n\_steps \times n\_envs}$$

So the total gradient updates for PPO:

$$\text{PPO gradient updates} = \text{cycles} \times \text{updates per cycle}$$

### Concrete Example (SB3 Defaults)

With SB3 defaults for PPO: `n_steps` = 2048, `n_envs` = 1, `n_epochs` = 10, `batch_size` = 64. Training for 500,000 steps.

Updates per cycle: $10 \times \lceil 2048 / 64 \rceil = 10 \times 32 = 320$

Number of cycles: $500{,}000 / 2048 \approx 244$

Total gradient updates: $244 \times 320 = 78{,}080$

### Comparison

For 500,000 environment steps:

| | DQN | PPO |
|--|-----|-----|
| Gradient updates | ~490,000 | ~78,000 |
| Updates per env step | ~1.0 | ~0.16 |
| Data reuse | Many times (replay buffer) | `n_epochs` times then discarded |

DQN performs roughly **6× more gradient updates** than PPO for the same number of environment steps. This is a major structural difference.

## 4. What This Means for Sample Efficiency

### DQN's Advantage: Data Reuse

DQN's replay buffer allows each transition to be used for many gradient updates. A transition collected at step 10,000 might contribute to gradient updates over the next 100,000 steps (until it's overwritten). This extreme data reuse means DQN extracts more learning per unit of experience.

### PPO's Limitation: Data Disposal

PPO uses each batch for `n_epochs` passes and then throws it away. A transition collected in one cycle is never seen again. This is the fundamental cost of being on-policy: the data must come from the current policy, so old data is invalid once the policy changes.

### But Gradient Updates ≠ Learning

More gradient updates doesn't automatically mean faster or better learning. DQN's gradient updates are computed against a moving target (the target network's estimates), using stale data from the replay buffer that may not reflect the current policy. Many of these updates are correcting for distributional mismatch rather than making genuine progress.

PPO's fewer gradient updates are computed on fresh, on-policy data with accurate advantage estimates. Each update is more "on target" because the data directly reflects the current policy's behavior. Fewer updates, but each one is more informative.

### The Trade-off in Practice

For LunarLander-v3 with the same total environment steps:

**DQN** is likely more sample-efficient in terms of environment interactions — it can reach a given performance level with fewer total steps because it reuses data aggressively. But it may be less stable because it's training on off-policy data.

**PPO** is likely less sample-efficient — it needs more environment steps to reach the same performance because it discards data after each cycle. But it may be more stable because each update uses fresh, on-policy data with accurate advantages.

This is exactly what the project asks you to analyze and compare.

## 5. How to Report This

### Computing Gradient Updates

For DQN, log the number of gradient updates directly from SB3's training loop, or compute it as `total_steps - learning_starts` (for the default 1:1 ratio).

For PPO, compute it using the formula above, or log it from SB3's internals. The key numbers to report are `n_steps`, `n_envs`, `n_epochs`, `batch_size`, and `total_timesteps`.

### What to Compare

For a sample efficiency comparison, this should include:

**Learning curves aligned by environment steps:** Plot mean return vs. total environment steps for both agents. This shows which agent learns faster in terms of experience — which is the fair comparison since environment interaction is the expensive resource.

**Learning curves aligned by gradient updates:** Optionally, plot mean return vs. total gradient updates. This shows which agent extracts more learning per gradient step. DQN may look worse here (more updates for similar performance) because many updates are correcting for off-policy data.

**Update efficiency:** Report the ratio of gradient updates to environment steps for each agent. Discuss why they differ (replay buffer vs. on-policy collection) and what the implications are.

### Reporting Key Points

When discussing sample efficiency in a theoretical section:

DQN is **off-policy**: it can learn from any past experience, regardless of what policy generated it. This enables the replay buffer and data reuse, making each environment step count for many gradient updates. The cost is distributional mismatch — the data in the buffer doesn't perfectly reflect the current policy.

PPO is **on-policy**: it can only learn from data generated by the current policy. This forces it to discard data after each update cycle, making each environment step count for fewer gradient updates. The benefit is that every gradient update is computed on perfectly aligned data.

Neither approach is strictly better — they optimize different trade-offs. DQN trades data freshness for reuse; PPO trades reuse for freshness. The project gives you the opportunity to show empirically how this trade-off plays out in a concrete environment.

## 6. Advanced Consideration: Wall-Clock Time

Sample efficiency (environment steps to reach a performance level) is not the same as wall-clock efficiency (real time to reach a performance level). DQN collects one step, does one gradient update, repeats — this is inherently sequential. PPO can collect data across many parallel environments simultaneously, then do all gradient updates in a batch.

With `n_envs` = 8, PPO collects experience 8× faster in real time, even though it uses the same total number of environment steps. This parallelism is a practical advantage of PPO that doesn't show up in sample efficiency plots but matters for real-world training time.

If a project asks about training time or computational cost, this distinction is important. If it only asks about sample efficiency (performance vs. environment steps), parallelism is irrelevant — what matters is total steps, not how fast they were collected.

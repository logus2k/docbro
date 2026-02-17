# PPO Tuning

## Clip Range, Epochs per Batch, and Entropy Coefficient

PPO's stability and ease of tuning come from a small set of well-understood hyperparameters that control how aggressively the policy updates, how thoroughly each batch of experience is used, and how much the agent is encouraged to explore. The three most impactful parameters for tuning are the clip range, the number of optimization epochs per batch, and the entropy coefficient.

**PPO hyperparameters:**
- Learning rate — step size for gradient updates
- Gamma — discount factor for future rewards
- N steps — how many steps per collection phase (2048)
- N epochs — how many passes over the collected data (10)
- Batch size — mini-batch size during training
- Clip range — the epsilon for clipping (0.2)
- GAE lambda — bias-variance trade-off for advantage estimation (0.95)
- Entropy coefficient — floor to prevent exploration from collapsing too early

This document explains each mechanism, how they interact, and how to tune them effectively for LunarLander-v3 with Stable-Baselines3.

## 1. The Clip Range ($\epsilon$)

### What It Does

The clip range is the core mechanism that makes PPO stable. It limits how much the policy can change in a single update by clipping the probability ratio between the new and old policy.

The probability ratio is defined as:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

When $r_t = 1$, the new policy assigns the same probability to action $a_t$ as the old policy did. Values above 1 mean the action became more likely; below 1 means less likely.

The clipped objective is:

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \cdot A_t,\; \text{clip}(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon) \cdot A_t\right)\right]$$

The clip range $\epsilon$ (typically 0.2) defines a trust region: the ratio $r_t$ is clamped to $[1 - \epsilon, 1 + \epsilon]$. The $\min$ operator ensures that the clipped (more conservative) version is used whenever it would reduce the objective.

### How the Clipping Works

Consider two cases:

**Positive advantage** ($A_t > 0$): The action was better than expected, so the gradient wants to increase its probability (push $r_t$ above 1). The clip caps $r_t$ at $1 + \epsilon$, preventing the policy from becoming too confident about this action from a single batch.

**Negative advantage** ($A_t < 0$): The action was worse than expected, so the gradient wants to decrease its probability (push $r_t$ below 1). The clip floors $r_t$ at $1 - \epsilon$, preventing the policy from overcorrecting and assigning near-zero probability to an action based on limited evidence.

### Why It Matters

Without clipping (or any trust region constraint), a single batch with strong advantage signals can cause a massive policy change. The new policy then visits completely different states, generating data that contradicts the previous update, causing another large change in the opposite direction. This oscillation can be catastrophic.

The clip range is a safety valve. It says: "regardless of how good or bad this action looks in the current batch, don't change the policy by more than $\epsilon$ in probability ratio terms."

### Tuning the Clip Range

**Smaller $\epsilon$** (e.g., 0.1): More conservative updates. The policy changes very slowly, which increases stability but can make learning painfully slow. Useful when training is unstable or when the environment is very sensitive to policy changes.

**Larger $\epsilon$** (e.g., 0.3): More aggressive updates. The policy can change more per batch, which speeds up learning but risks overshooting and destabilizing training. Useful when training is stable but slow.

**Default:** SB3 uses $\epsilon = 0.2$, which is the value from the original PPO paper and works well across a wide range of problems including LunarLander-v3.

**Diagnostic:** If the learning curve is smooth but very slow, try increasing $\epsilon$. If the learning curve is erratic with sudden performance drops, try decreasing $\epsilon$.

### Clip Range Annealing

Some implementations anneal $\epsilon$ from a higher value to a lower value over training. The intuition is that early in training, larger updates are beneficial (the policy is far from optimal, so big steps are helpful), while late in training, smaller updates are safer (the policy is close to optimal, and large steps risk destabilizing it). SB3 does not anneal the clip range by default, but it can be implemented via a custom schedule.

## 2. Epochs per Batch (`n_epochs`)

### What It Does

After collecting a batch of experience (a fixed number of environment steps), PPO performs multiple passes (epochs) of gradient updates over that same batch. Each epoch shuffles the batch into mini-batches and performs one gradient step per mini-batch.

This is a critical difference from vanilla policy gradient methods (like REINFORCE or A2C), which use each batch exactly once and then discard it. PPO's clipping mechanism is what makes multiple epochs safe — without it, repeated updates on the same data would push the policy far from the behavior that generated the data, violating the on-policy assumption.

### Why Multiple Epochs Matter

A single pass over the batch extracts only a fraction of the useful gradient information. Multiple passes allow the network to learn more from each batch, improving sample efficiency. However, each additional epoch pushes the policy further from the policy that generated the data. Eventually, the probability ratios hit the clip boundaries and further updates have no effect — the clipping mechanism naturally limits how much can be extracted from a single batch.

### Tuning `n_epochs`

**Too few epochs** (e.g., 1): Each batch is underutilized. The agent needs more environment interaction to make the same amount of progress. This is wasteful in terms of sample efficiency.

**Too many epochs** (e.g., 20+): The policy moves as far as the clip allows on every batch, then spends additional epochs computing gradients that are entirely clipped (zero effective learning). This wastes computation without improving the policy. Worse, if the clipping is not perfectly tight, excessive epochs can cause subtle overfitting to the batch — the policy becomes too specialized to the states and actions in the current batch rather than learning general behavior.

**The sweet spot** (3–10 epochs): Enough passes to extract most of the useful information, few enough that the clipping mechanism hasn't saturated on most transitions. The default in SB3 is 10 for PPO.

### Interaction with Clip Range

Epochs and clip range are directly coupled:

- **Large $\epsilon$ + many epochs** = aggressive updates. The policy has room to move ($\epsilon$ is loose) and many opportunities to move (many epochs). This is the most aggressive setting and the most likely to destabilize.
- **Small $\epsilon$ + few epochs** = very conservative. The policy barely changes per batch. Extremely stable but potentially very slow.
- **Large $\epsilon$ + few epochs** = the policy could move a lot but doesn't get enough gradient steps to do so. Underutilizes the available update budget.
- **Small $\epsilon$ + many epochs** = the policy hits the clip walls quickly and subsequent epochs are wasted. Stable but computationally wasteful.

For LunarLander-v3, the default combination ($\epsilon = 0.2$, `n_epochs` = 10) is a reasonable starting point. If training is unstable, reduce `n_epochs` to 3–5 before reducing $\epsilon$.

### Interaction with Batch Size

The total batch of experience is collected over `n_steps` environment steps across `n_envs` parallel environments. The total batch size is `n_steps × n_envs`. This batch is then divided into `n_epochs × (batch_size / mini_batch_size)` gradient updates.

A larger total batch provides more diverse experience per update cycle, which makes multiple epochs more useful (there's more to learn from). A smaller total batch has less diversity, so multiple epochs are more likely to overfit.

## 3. Entropy Coefficient ($c_2$)

### What It Does

The entropy coefficient adds a bonus to the PPO objective that rewards the policy for maintaining high entropy — that is, for keeping the action probability distribution spread out rather than concentrated on a single action.

The combined PPO objective is:

$$L(\theta) = L^{CLIP}(\theta) - c_1 \cdot L^{V}(\theta) + c_2 \cdot H[\pi_\theta(\cdot|s_t)]$$

where $H[\pi_\theta(\cdot|s_t)]$ is the entropy of the policy's action distribution at state $s_t$. For a discrete action space with 4 actions (as in LunarLander-v3), maximum entropy is $\log(4) \approx 1.386$ (uniform distribution) and minimum is 0 (deterministic).

### Why It Exists

Without the entropy bonus, the policy gradient naturally pushes the policy toward determinism — if action $a$ has a positive advantage, its probability increases, which means it gets selected more often, which provides more data showing it has positive advantage, which increases its probability further. This positive feedback loop can cause **premature convergence**: the policy collapses to a deterministic choice before adequately exploring alternatives.

The entropy bonus counteracts this by penalizing low-entropy (near-deterministic) distributions. It says: "all else being equal, prefer a policy that keeps its options open." This serves as PPO's primary exploration mechanism — the equivalent of epsilon-greedy in DQN, but smoother and more principled.

### How Entropy Drives Exploration

In DQN, exploration is binary: either the agent takes the greedy action or a completely random action. In PPO, exploration is continuous: the policy maintains a probability distribution over all actions, and the entropy bonus controls how spread out that distribution is.

Early in training, when advantages are noisy and unreliable, the entropy bonus keeps the policy from collapsing onto a single action. As training progresses and advantages become more accurate, the policy naturally becomes more peaked (lower entropy) because the advantage signal increasingly dominates the entropy bonus. The policy sharpens as it becomes more confident.

### Tuning the Entropy Coefficient

**Too low** ($c_2 \approx 0$): No exploration incentive. The policy collapses to near-deterministic behavior early, potentially locking onto a suboptimal action in many states. Once collapsed, recovery is very difficult because the policy no longer generates diverse data.

**Too high** ($c_2 > 0.05$): The entropy bonus dominates the advantage signal. The policy stays near-uniform regardless of what the agent learns. The agent explores thoroughly but never commits to good actions, and reward performance plateaus at a mediocre level.

**The balance** ($c_2 = 0.0$ to $0.01$): Enough entropy pressure to prevent premature collapse, little enough that the advantage signal can drive meaningful policy improvement. SB3's default is 0.0 for PPO, which works for many environments but may need to be increased for environments where exploration is important.

For LunarLander-v3, values between 0.0 and 0.01 are reasonable. If the agent converges quickly to a poor policy (e.g., always fires the main engine or always does nothing), try increasing $c_2$ to 0.005 or 0.01. If the agent never seems to commit to a clear strategy, try decreasing it.

### Monitoring Entropy

SB3 logs the policy entropy during training. A healthy entropy curve typically shows high entropy early (near-uniform policy), gradually decreasing as the policy becomes more confident, and stabilizing at a moderate level. Warning signs:

- **Entropy drops to near zero quickly:** Premature convergence. Increase $c_2$.
- **Entropy stays high throughout training:** Policy never commits. Decrease $c_2$ or increase the number of training steps.
- **Entropy oscillates wildly:** Training instability. Check the clip range and learning rate.

## 4. Other Important PPO Hyperparameters

While clip range, epochs, and entropy are the primary tuning targets, several other parameters interact with them and are worth understanding.

### Learning Rate

Controls the step size for gradient updates. PPO is generally less sensitive to the learning rate than DQN, but extreme values cause problems. SB3 defaults to $3 \times 10^{-4}$.

**Too high:** Policy updates overshoot, causing oscillation even within the clip region. The ratio $r_t$ frequently hits the clip boundaries, wasting gradient information.

**Too low:** Each epoch of updates makes negligible progress. The agent needs many more batches to learn, wasting environment steps.

For LunarLander-v3, values between $1 \times 10^{-4}$ and $3 \times 10^{-3}$ are reasonable. Linear annealing of the learning rate (high early, low late) can help: fast initial learning followed by stable fine-tuning.

### `n_steps` (Rollout Length)

The number of environment steps collected per rollout before an update is performed. This determines the temporal horizon of the advantage estimates and the diversity of each batch.

**Short rollouts** (e.g., 128 steps): More frequent updates, but advantage estimates are computed over shorter horizons and rely more on the value function's accuracy (higher bias, lower variance).

**Long rollouts** (e.g., 2048 steps): Less frequent updates, but advantage estimates span longer trajectories and are more accurate (lower bias, higher variance). Also provides more diverse batches for the multiple epochs to learn from.

SB3 defaults to 2048 for PPO. For LunarLander-v3, values between 512 and 2048 are reasonable.

### GAE Lambda ($\lambda$)

Controls the bias-variance trade-off in Generalized Advantage Estimation:

$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

$\lambda = 1$ gives unbiased but high-variance Monte Carlo-like estimates. $\lambda = 0$ gives biased but low-variance single-step TD estimates. The default is 0.95, which works well in most settings. Lower values (0.9) can help if training is unstable due to noisy advantages.

### Value Function Coefficient ($c_1$)

Weight of the critic's loss in the combined objective. Defaults to 0.5. Increasing it prioritizes value function accuracy (better advantages but potentially worse policy updates). Rarely needs tuning.

## 5. Summary of SB3 PPO Hyperparameters

| Parameter | SB3 Name | Default | Role | Tuning Range (LunarLander) |
|-----------|----------|---------|------|---------------------------|
| Clip range | `clip_range` | 0.2 | Trust region size for policy updates | 0.1 – 0.3 |
| Epochs per batch | `n_epochs` | 10 | Passes over each collected batch | 3 – 10 |
| Entropy coefficient | `ent_coef` | 0.0 | Exploration incentive strength | 0.0 – 0.01 |
| Learning rate | `learning_rate` | 0.0003 | Gradient step size | 0.0001 – 0.003 |
| Rollout length | `n_steps` | 2048 | Environment steps per batch | 512 – 2048 |
| GAE lambda | `gae_lambda` | 0.95 | Advantage estimation bias-variance | 0.9 – 0.99 |
| Discount factor | `gamma` | 0.99 | Weight of future rewards | 0.99 – 0.999 |
| Value function coeff. | `vf_coef` | 0.5 | Weight of critic loss | 0.5 – 1.0 |
| Max grad norm | `max_grad_norm` | 0.5 | Gradient clipping for stability | 0.5 – 1.0 |
| Number of envs | `n_envs` | 1 | Parallel environments for data collection | 1 – 16 |

## 6. Diagnostic Checklist

**Agent doesn't learn at all:** Check learning rate (too low or too high). Verify the environment is set up correctly. Check that `n_steps` is long enough to capture meaningful trajectories.

**Agent learns then collapses:** Clip range may be too large (aggressive updates destabilize a good policy). Try reducing `clip_range` to 0.1 or reducing `n_epochs` to 3–5. Learning rate annealing can also help — reduce the learning rate in later training.

**Agent converges quickly to a bad policy:** Premature convergence — entropy collapsed. Increase `ent_coef` to 0.005–0.01. Check the entropy log to confirm it dropped to near zero.

**Agent never converges, reward stays mediocre:** Entropy may be too high (policy stays diffuse). Try reducing `ent_coef`. Alternatively, `n_steps` may be too short for meaningful advantage estimation.

**Learning curve is very noisy:** High variance in advantage estimates. Try increasing `n_steps` for longer rollouts, or decreasing `gae_lambda` slightly (e.g., 0.9) to reduce variance at the cost of some bias.

**Agent performs well in training but worse in evaluation:** PPO's stochastic policy samples actions during training but evaluation may use the mode (most likely action). If the policy is still somewhat entropic, training and evaluation behavior will differ. This gap should shrink as entropy decreases naturally over training.

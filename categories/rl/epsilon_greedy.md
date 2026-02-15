# Epsilon-Greedy vs. Entropy-Driven Exploration

## Overview

DQN and PPO use fundamentally different mechanisms to balance exploration and exploitation. DQN uses **epsilon-greedy** — a binary switch between random and greedy behavior. PPO uses **entropy regularization** — a continuous incentive to maintain diversity in the action probability distribution. Both serve the same purpose (preventing premature convergence to a suboptimal policy) but differ in how they achieve it, how they adapt over training, and what failure modes they exhibit.

This document provides a detailed comparison of both mechanisms, their theoretical properties, practical implications, and what to look for when analyzing them in the LunarLander-v3 project.

---

## 1. Epsilon-Greedy Exploration (DQN)

### Mechanism

At each timestep, the agent generates a random number $u \sim \text{Uniform}(0, 1)$:

- If $u < \varepsilon$: select a **random action** uniformly from the action space.
- If $u \geq \varepsilon$: select the **greedy action** $\arg\max_a Q(s, a)$.

$\varepsilon$ is annealed over training from a high value (typically 1.0) to a low value (typically 0.01–0.05).

### Properties

**State-independent.** The exploration probability is the same regardless of what state the agent is in. Whether the agent is in a well-understood state where the Q-values are reliable, or in a novel state where the Q-values are meaningless, the exploration rate is identical. This is wasteful — the agent explores unnecessarily in known states and insufficiently in unknown states.

**Action-independent.** When the agent does explore, it selects uniformly at random. It's equally likely to pick a known-bad action as a never-tried action. In LunarLander-v3 with 4 actions, a random exploration step has a 25% chance of selecting each action, regardless of their estimated values. An action with Q-value of -100 is just as likely to be explored as one with Q-value of +50.

**Binary.** At each step, the agent is either fully greedy or fully random. There is no middle ground — no "slightly exploratory" behavior. The policy is a hard mixture of two extremes.

**Externally scheduled.** The decay of $\varepsilon$ follows a predetermined schedule (linear, exponential, etc.) that is set before training begins. It doesn't adapt to the agent's actual learning progress. If the agent is struggling in a particular part of the state space, $\varepsilon$ continues to decay on schedule regardless.

### Epsilon Decay Visualization

Over 500,000 steps with `exploration_fraction` = 0.1, the decay looks like:

```
ε
1.0 |*
    | *
    |  *
    |   *
    |    *
    |     *
    |      *
    |       *
    |        *
0.05|         *************************************
    |____________________________________________
    0       50k                              500k
                    environment steps
```

For the first 50,000 steps, $\varepsilon$ decreases linearly. For the remaining 450,000 steps, it stays at 0.05. The agent is near-greedy for 90% of training.

---

## 2. Entropy-Driven Exploration (PPO)

### Mechanism

PPO's policy network outputs a probability distribution over actions. For LunarLander-v3 with 4 discrete actions, this is a categorical distribution $\pi_\theta(a|s) = [p_0, p_1, p_2, p_3]$ where $\sum_i p_i = 1$.

The entropy of this distribution measures its spread:

$$H[\pi_\theta(\cdot|s)] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

For 4 actions: maximum entropy is $\log(4) \approx 1.386$ (uniform distribution $[0.25, 0.25, 0.25, 0.25]$) and minimum entropy is 0 (deterministic, e.g., $[1, 0, 0, 0]$).

The entropy bonus in the PPO objective:

$$L(\theta) = L^{CLIP}(\theta) - c_1 \cdot L^{V}(\theta) + c_2 \cdot H[\pi_\theta(\cdot|s)]$$

adds a reward proportional to entropy. The policy is incentivized to stay spread out, counteracting the natural tendency of policy gradients to collapse toward determinism.

### Properties

**State-dependent.** The policy outputs a different probability distribution for each state. In states where the agent is confident (one action has much higher advantage), the distribution will be peaked even with the entropy bonus. In states where the agent is uncertain (advantages are similar), the distribution stays broad. Exploration is automatically concentrated where it's needed.

**Action-aware.** The probability assigned to each action reflects the agent's learned knowledge about that action's quality. A known-bad action will have low probability; a promising but uncertain action will have moderate probability. Exploration is directed toward plausible alternatives, not uniformly random.

**Continuous.** There is no binary switch. The policy always samples from a distribution, so there is always some probability of selecting any action. Exploration and exploitation are blended smoothly rather than being mutually exclusive.

**Self-adapting.** As the policy improves and advantages become clearer, the distribution naturally sharpens (entropy decreases) without any external schedule. The entropy coefficient $c_2$ sets the floor, but the actual entropy is determined by the interaction between the advantage signal and the entropy bonus. If the agent enters a new region of the state space where advantages are unclear, entropy will naturally increase there.

### Entropy Evolution During Training

A typical entropy curve for PPO on LunarLander-v3:

```
H
1.38|***
    |   ***
    |      **
    |        **
    |          ***
    |             ***
    |                ****
    |                    ********
0.2 |                            ***************
    |____________________________________________
    0                                        500k
                    environment steps
```

Entropy decreases gradually as the policy becomes more confident. The shape is emergent — it reflects the agent's actual learning progress, not a predetermined schedule. Plateaus indicate periods where the agent is exploring among similarly valued actions. Sharp drops indicate breakthrough moments where the agent discovers a clearly superior action.

---

## 3. Detailed Comparison

### Exploration Quality

**Epsilon-greedy** explores blindly. When it explores, it samples uniformly, so in LunarLander-v3, the agent is equally likely to fire the left engine, the right engine, the main engine, or do nothing — regardless of which direction the lander is tilting. This generates many useless transitions where the agent takes obviously counterproductive actions.

**Entropy-driven** exploration is informed by the current policy. If the agent has learned that firing the main engine is generally good when falling, the policy will assign it high probability. Entropy-driven exploration might still occasionally fire a side engine, but with lower probability than the main engine. The exploratory actions are concentrated in the "zone of plausibility" rather than the entire action space.

### Adaptation to State

Consider two states in LunarLander-v3:

**State A:** The lander is centered, descending slowly, near the landing pad. The optimal action is clearly "fire main engine" to slow descent. The agent's Q-values (DQN) or policy (PPO) strongly favor this action.

**State B:** The lander is off-center, high altitude, with lateral velocity. Multiple actions could be reasonable depending on the exact configuration. The agent is less certain.

With epsilon-greedy ($\varepsilon = 0.05$), both states have a 5% chance of random action. In State A, this is pure waste — the agent knows what to do and random exploration only hurts. In State B, 5% may be insufficient — the agent is uncertain and could benefit from more exploration.

With entropy-driven exploration, the policy in State A will be highly peaked (low entropy, near-deterministic), producing almost no exploratory actions. The policy in State B will be broader (higher entropy, more stochastic), naturally exploring more. The exploration rate adapts to the agent's uncertainty **per state**.

### Exploration During Evaluation

**DQN evaluation:** $\varepsilon$ is set to 0 (fully greedy). The agent always takes $\arg\max_a Q(s, a)$. Evaluation behavior is completely deterministic and reflects only the Q-values.

**PPO evaluation:** The policy can be evaluated in two ways. **Stochastic evaluation** samples from the learned distribution — the agent's behavior includes residual exploration from the entropy in the policy. **Deterministic evaluation** takes the mode (most likely action) — similar to DQN's greedy evaluation. SB3's `evaluate_policy` with `deterministic=True` uses the mode.

The project asks for 20 deterministic evaluation episodes per seed, so both agents will be evaluated greedily. However, the training behavior differs, and this affects what the agents learn.

### Impact on the Replay Buffer / Training Data

**DQN:** The replay buffer contains a mixture of exploratory (random) and greedy transitions. Early in training, almost all transitions are random. Late in training, most are greedy with occasional random actions. The agent must learn from this mixed data, and the Q-network must be robust to transitions generated by very different behavior policies.

**PPO:** The training data is always generated by the current policy, including its current entropy level. The data is consistent — it always reflects the same exploration-exploitation balance that the policy embodies. This consistency makes advantage estimation and policy gradient computation cleaner.

---

## 4. Failure Modes

### Epsilon-Greedy Failures

**Premature exploitation:** If $\varepsilon$ decays too fast, the agent commits to its Q-values before they're accurate. In LunarLander-v3, this might manifest as the agent learning to hover (which avoids crash penalties) but never discovering that landing is far more rewarding. Once $\varepsilon$ is low, the agent never randomly stumbles onto the landing behavior.

**Wasted exploration:** Even with $\varepsilon$ at 0.05 late in training, 5% of all actions are random. Over 500,000 steps, that's 25,000 random actions — most of which generate unhelpful transitions (e.g., firing the wrong engine at the wrong time). These transitions enter the replay buffer and the agent wastes gradient updates on them.

**Catastrophic exploration:** A single random action at a critical moment (e.g., firing the side engine when the lander is about to touch down) can cause a crash, generating a strongly negative transition. If this transition is sampled frequently from the replay buffer, it can disproportionately influence Q-values.

### Entropy-Driven Failures

**Entropy collapse:** If the entropy coefficient is too low (or zero), the policy can collapse to near-deterministic behavior early in training. Once collapsed, the policy generates highly homogeneous data, and the advantage estimates reinforce the current behavior. Recovery is extremely difficult because the policy no longer generates the diverse data needed to discover better actions.

**Persistent stochasticity:** If the entropy coefficient is too high, the policy never commits. In LunarLander-v3, this might look like the agent learning approximately correct behavior but always with enough randomness that it frequently crashes on landing. The entropy bonus prevents the policy from becoming precise enough for the delicate landing phase.

**False confidence:** The entropy is state-dependent, so if the value function is poorly calibrated, the policy might be confidently wrong — low entropy in a state where it should be uncertain. This is harder to detect than epsilon-greedy failures because the exploration metric (entropy) looks healthy overall even if it's misallocated across states.

---

## 5. What to Monitor and Report

### DQN Exploration Metrics

**Epsilon over time:** Plot $\varepsilon$ vs. environment steps. This is deterministic (based on the schedule) but useful for correlating with learning curve changes. Mark the point where epsilon reaches its final value — this is when the agent transitions from primarily exploring to primarily exploiting.

**Random action frequency:** The actual fraction of random actions should match $\varepsilon$. If using SB3's default logging, this is implicit in the schedule.

### PPO Exploration Metrics

**Policy entropy over time:** Plot the mean entropy across states vs. environment steps. SB3 logs this by default. This is the most informative exploration metric for PPO — it shows how the agent's exploration evolves as an emergent property of learning.

**Entropy per state (optional but insightful):** If feasible, visualize entropy as a function of state features. For example, plot entropy vs. altitude or entropy vs. horizontal position. This reveals where the agent is confident and where it's uncertain.

### Comparison for the Report

The project explicitly asks for a comparison of $\varepsilon$-greedy vs. entropy-driven exploration. Key points to address:

**Mechanism:** Epsilon-greedy is external, scheduled, state-independent, and binary. Entropy-driven exploration is internal, emergent, state-dependent, and continuous.

**Efficiency:** Entropy-driven exploration focuses exploratory actions where uncertainty is high, making better use of exploration budget. Epsilon-greedy wastes exploration in well-known states.

**Adaptivity:** Entropy naturally adapts to learning progress. Epsilon follows a fixed schedule regardless of what the agent has learned.

**Failure modes:** Epsilon-greedy can waste exploration and suffer from premature exploitation. Entropy-driven exploration can collapse prematurely or maintain excessive stochasticity.

**Empirical evidence:** Overlay the epsilon curve and entropy curve on the same plot (or side by side with aligned x-axes). Correlate changes in exploration metrics with changes in the learning curve. Identify moments where exploration behavior visibly affected learning progress.

---

## 6. Summary Table

| Property | Epsilon-Greedy (DQN) | Entropy-Driven (PPO) |
|----------|---------------------|---------------------|
| Type | External mechanism | Internal to the policy |
| Schedule | Predetermined decay | Emergent from learning |
| State dependence | None — same $\varepsilon$ everywhere | Full — entropy varies per state |
| Action selection during exploration | Uniform random | Weighted by learned policy |
| Exploration-exploitation boundary | Binary (random or greedy) | Continuous (probability distribution) |
| Adaptation to learning progress | None | Automatic |
| Control parameter | $\varepsilon$ schedule | Entropy coefficient $c_2$ |
| Failure mode: under-exploration | Fast $\varepsilon$ decay | Low $c_2$, entropy collapse |
| Failure mode: over-exploration | Slow $\varepsilon$ decay | High $c_2$, persistent stochasticity |
| Monitoring | Epsilon value (deterministic) | Policy entropy (emergent) |

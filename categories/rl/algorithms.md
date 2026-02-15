# Reinforcement Learning Algorithms

Reinforcement learning (RL) is a paradigm where an agent learns to make decisions by interacting with an environment, receiving rewards, and adjusting its behavior to maximize cumulative reward over time. Over the past decade, the combination of RL with deep neural networks (deep RL) has produced algorithms capable of superhuman game play, robotic control, scientific discovery, and language model alignment.

This document surveys the most well-known and widely used RL algorithms, organized by family. Each entry covers the core idea, key innovations, strengths, limitations, and typical use cases.

## 1. Value-Based Methods

Value-based methods learn a value function (typically the action-value function $Q(s, a)$) and derive a policy from it, usually by acting greedily: $\pi(s) = \arg\max_a Q(s, a)$. These methods are inherently restricted to **discrete action spaces** since they require evaluating or maximizing over all possible actions.

### 1.1 DQN (Deep Q-Network)

**Paper:** Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.

DQN replaces the tabular Q-table with a deep neural network that approximates $Q(s, a; \theta)$. Two key innovations stabilize training: an **experience replay buffer** that stores transitions and samples random mini-batches to break temporal correlations, and a **target network** (a periodically updated frozen copy of the Q-network) that provides stable regression targets.

The network is trained by minimizing the temporal difference loss:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \cdot \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

Action selection uses an $\varepsilon$-greedy policy, annealed from full exploration to near-greedy over training.

**Strengths:** Handles high-dimensional state spaces (raw pixels), sample-efficient due to replay, conceptually simple.
**Limitations:** Discrete actions only, overestimates Q-values, naive exploration strategy.
**Use cases:** Atari games, discrete control problems, foundational baseline for value-based methods.

### 1.2 Double DQN (DDQN)

**Paper:** van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.

Standard DQN uses the same network to both select and evaluate the best action in the target computation, which introduces a systematic overestimation bias. Double DQN decouples these steps: the **online network** selects the action, and the **target network** evaluates it:

$$y = r + \gamma \cdot Q\!\left(s',\; \underset{a'}{\arg\max}\; Q(s', a'; \theta);\; \theta^-\right)$$

This simple change significantly reduces overestimation and improves learning stability with no additional computational cost.

### 1.3 Dueling DQN

**Paper:** Wang et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML.

Dueling DQN modifies the network architecture to separately estimate the **state value** $V(s)$ and the **advantage** $A(s, a)$ of each action:

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')$$

This decomposition is beneficial in states where the choice of action has little impact on the outcome — the network can learn state values without needing to evaluate every action individually. The advantage stream only needs to capture relative differences between actions.

### 1.4 Prioritized Experience Replay (PER)

**Paper:** Schaul et al. (2016). "Prioritized Experience Replay." ICLR.

Replaces uniform random sampling from the replay buffer with priority-based sampling, where transitions with higher TD error are sampled more frequently. This focuses learning on the most surprising and informative experiences. Importance sampling weights correct for the introduced bias:

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

where $\beta$ is annealed from a small value to 1 over training. PER is a general technique that can be combined with any off-policy algorithm using a replay buffer.

### 1.5 Rainbow

**Paper:** Hessel et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning." AAAI.

Rainbow combines six orthogonal improvements to DQN into a single agent: Double DQN, Prioritized Experience Replay, Dueling architecture, multi-step returns (n-step bootstrapping), Distributional RL (C51), and Noisy Networks (NoisyNet). Ablation studies showed that each component contributes meaningfully, and the combined agent significantly outperforms any individual variant.

**C51** (Categorical DQN) learns the full distribution of returns rather than just the expected value, using a categorical distribution over a fixed set of atoms. **NoisyNet** replaces $\varepsilon$-greedy with learned parametric noise in the network weights, enabling state-dependent exploration.

Rainbow established that DQN improvements are largely complementary and set the standard for value-based performance on Atari benchmarks.

### 1.6 R2D2 (Recurrent Replay Distributed DQN)

**Paper:** Kapturowski et al. (2019). "Recurrent Experience Replay in Distributed Reinforcement Learning." ICLR.

R2D2 extends DQN with LSTM layers for partially observable environments and operates in a distributed setting with many actors collecting experience in parallel. It addresses the challenge of storing and replaying recurrent states by using stored hidden states with a burn-in period to warm up the LSTM at the start of each replayed sequence. R2D2 achieved state-of-the-art results on Atari, significantly surpassing Rainbow.

### 1.7 QR-DQN (Quantile Regression DQN)

**Paper:** Dabney et al. (2018). "Distributional Reinforcement Learning with Quantile Regression." AAAI.

An alternative to C51's categorical approach, QR-DQN learns the quantiles of the return distribution rather than approximating it with fixed atoms. The network outputs $N$ quantile values, trained with quantile regression loss (asymmetric Huber loss). This avoids C51's need to project onto a fixed support and is generally easier to implement while achieving comparable performance.

### 1.8 IQN (Implicit Quantile Network)

**Paper:** Dabney et al. (2018). "Implicit Quantile Networks for Distributional Reinforcement Learning." ICML.

IQN generalizes QR-DQN by learning a continuous map from quantile fractions $\tau \in [0, 1]$ to return values, rather than a fixed set of quantiles. At each training step, random quantile fractions are sampled and embedded via a cosine embedding, enabling the network to approximate the full inverse CDF of the return distribution. IQN enables risk-sensitive policies by choosing actions based on specific quantile ranges (e.g., optimistic or pessimistic behavior).

## 2. Policy Gradient Methods

Policy gradient methods directly parameterize and optimize the policy $\pi_\theta(a|s)$ without maintaining an explicit value function for action selection. The policy is updated by following the gradient of expected return with respect to its parameters.

### 2.1 REINFORCE

**Paper:** Williams (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine Learning.

The foundational policy gradient algorithm. REINFORCE samples complete episodes, then updates the policy using the score function estimator:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return from timestep $t$. A **baseline** (typically the mean return or a learned state value) is subtracted from $G_t$ to reduce variance without introducing bias.

**Strengths:** Conceptually simple, works with any differentiable policy, handles both discrete and continuous actions.
**Limitations:** Extremely high variance, requires complete episodes (no bootstrapping), very sample-inefficient.
**Use cases:** Educational, simple problems, foundation for all subsequent policy gradient methods.

### 2.2 TRPO (Trust Region Policy Optimization)

**Paper:** Schulman et al. (2015). "Trust Region Policy Optimization." ICML.

TRPO constrains policy updates to a trust region defined by a KL-divergence bound, ensuring that the new policy doesn't deviate too far from the old one:

$$\max_\theta \; \mathbb{E}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A_t\right] \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta$$

This constrained optimization is solved using conjugate gradients and a line search, guaranteeing monotonic improvement under certain assumptions. TRPO was the first algorithm to reliably train complex policies on continuous control tasks without catastrophic collapses.

**Strengths:** Theoretically grounded monotonic improvement, very stable.
**Limitations:** Complex implementation (second-order optimization), computationally expensive, hard to scale.
**Use cases:** Largely superseded by PPO, but still relevant for theoretical analysis and settings where strict trust regions are needed.

### 2.3 PPO (Proximal Policy Optimization)

**Paper:** Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

PPO approximates TRPO's trust region constraint with a much simpler clipped surrogate objective:

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \cdot A_t,\; \text{clip}(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon) \cdot A_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio. The clipping prevents large policy changes while allowing multiple epochs of updates per batch of collected data.

PPO uses Generalized Advantage Estimation (GAE) for computing advantages, typically runs parallel environment workers for data collection, and combines the clipped policy loss with a value function loss and entropy bonus.

**Strengths:** Simple, stable, scalable, easy to tune, works with discrete and continuous actions.
**Limitations:** On-policy (sample-inefficient), can be overly conservative due to fixed clipping.
**Use cases:** The default general-purpose RL algorithm. Used in OpenAI Five, RLHF for LLMs, robotics, and most practical RL applications.

## 3. Actor-Critic Methods

Actor-critic methods combine a **policy** (actor) with a **value function** (critic). The critic evaluates the actor's actions, providing lower-variance learning signals compared to pure policy gradient methods.

### 3.1 A2C / A3C (Advantage Actor-Critic)

**Papers:** Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." ICML.

**A3C** (Asynchronous Advantage Actor-Critic) runs multiple workers in parallel, each interacting with its own copy of the environment and asynchronously updating shared network parameters. The advantage $A_t = R_t - V(s_t)$ guides the policy gradient, reducing variance compared to REINFORCE.

**A2C** is the synchronous variant: all workers collect data, gradients are averaged, and a single update is applied. A2C is simpler to implement, easier to debug, and in practice achieves comparable performance to A3C while being more reproducible.

**Strengths:** Parallel data collection, lower variance than REINFORCE, works on-policy with both discrete and continuous actions.
**Limitations:** On-policy (no replay), less stable than PPO, sensitive to learning rate.
**Use cases:** Largely superseded by PPO, but remains a useful educational stepping stone.

### 3.2 DDPG (Deep Deterministic Policy Gradient)

**Paper:** Lillicrap et al. (2016). "Continuous control with deep reinforcement learning." ICLR.

DDPG adapts DQN's ideas to continuous action spaces using a **deterministic policy** — the actor outputs a single action rather than a distribution. Since the policy is deterministic, exploration is achieved by adding noise to the output (originally Ornstein-Uhlenbeck process noise, though Gaussian noise works equally well).

DDPG is off-policy, using a replay buffer and target networks for both the actor and critic. The critic is trained via TD learning on $Q(s, a)$, and the actor is updated by following the gradient of the critic's Q-value with respect to the action:

$$\nabla_\theta J \approx \mathbb{E}\left[\nabla_a Q(s, a; \phi)\big|_{a=\pi_\theta(s)} \cdot \nabla_\theta \pi_\theta(s)\right]$$

**Strengths:** Off-policy (sample-efficient), works with continuous actions, directly applicable to robotics.
**Limitations:** Notoriously brittle and sensitive to hyperparameters, prone to Q-value overestimation, exploration dependent on noise tuning.
**Use cases:** Largely superseded by TD3 and SAC, but foundational for continuous control RL.

### 3.3 TD3 (Twin Delayed DDPG)

**Paper:** Fujimoto et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." ICML.

TD3 directly addresses DDPG's instabilities with three targeted fixes:

**Twin critics:** Two independent Q-networks are trained, and the minimum of their predictions is used for target computation. This counteracts the overestimation bias that plagues single-critic methods.

**Delayed actor updates:** The actor (and target networks) are updated less frequently than the critics (e.g., every 2 critic updates). This ensures the actor's gradients come from a more converged critic.

**Target policy smoothing:** Gaussian noise is added to the target action used in the critic's target computation, acting as a regularizer that prevents the critic from exploiting narrow peaks in the Q-function.

**Strengths:** Much more stable than DDPG, off-policy, strong continuous control performance.
**Limitations:** Deterministic policy limits exploration, still requires careful tuning, less principled than SAC.
**Use cases:** Continuous control, robotics, any setting where DDPG was previously used.

### 3.4 SAC (Soft Actor-Critic)

**Paper:** Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML.

SAC maximizes a modified objective that includes an **entropy bonus**, encouraging the policy to remain as stochastic as possible while still achieving high reward:

$$J(\pi) = \sum_t \mathbb{E}\left[r(s_t, a_t) + \alpha \cdot \mathcal{H}[\pi(\cdot|s_t)]\right]$$

where $\alpha$ is the temperature parameter controlling the entropy-reward trade-off (often auto-tuned via a learned Lagrange multiplier). The policy outputs a **Gaussian distribution** (mean and log-std), from which actions are sampled using the reparameterization trick.

SAC uses twin critics (like TD3) and is fully off-policy with a replay buffer. The entropy regularization provides automatic, principled exploration and leads to policies that are robust to perturbations (multiple near-optimal behaviors are maintained rather than collapsing to a single deterministic action).

**Strengths:** State-of-the-art sample efficiency for continuous control, built-in exploration, robust and stable, auto-tuned temperature.
**Limitations:** Primarily continuous actions (discrete variants exist but are less common), entropy objective can conflict with tasks requiring precise deterministic behavior.
**Use cases:** Continuous control benchmarks (MuJoCo), robotics, sim-to-real transfer, any off-policy continuous control setting.

### 3.5 IMPALA (Importance Weighted Actor-Learner Architecture)

**Paper:** Espeholt et al. (2018). "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures." ICML.

IMPALA separates data collection (actors) from learning (learners) in a distributed architecture. Many actors run the policy in parallel and send trajectories to a centralized learner, which processes them in batches. Since the policy may have been updated between when data was collected and when it's consumed, IMPALA uses **V-trace** off-policy corrections — truncated importance sampling that clips the importance weights to control variance.

**Strengths:** Highly scalable, efficient GPU utilization, handles policy lag gracefully.
**Limitations:** Complex distributed infrastructure, V-trace introduces some bias.
**Use cases:** Large-scale training (DeepMind's DMLab, StarCraft), settings with many parallel environments.

## 4. Model-Based Methods

Model-based RL algorithms learn a **model of the environment** (transition dynamics, reward function, or both) and use it for planning, imagination, or data augmentation. They are typically more sample-efficient than model-free methods because the agent can "think" about consequences without actually interacting with the real environment.

### 4.1 World Models

**Paper:** Ha & Schmidhuber (2018). "World Models." NeurIPS.

World Models introduced a three-component architecture: a **VAE** encodes observations into a compact latent space, an **RNN** (MDN-RNN) predicts future latent states, and a **linear controller** selects actions based on the latent state and RNN hidden state. The controller can be trained entirely inside the "dream" — the learned model — using evolutionary strategies, without any further interaction with the real environment.

This demonstrated that a learned world model can be rich enough to train a policy purely through imagination, though the approach was limited to relatively simple environments (e.g., VizDoom, CarRacing).

### 4.2 Dreamer / DreamerV2 / DreamerV3

**Papers:** Hafner et al. (2020, 2021, 2023). "Dream to Control" / "Mastering Atari with Discrete World Models" / "Mastering Diverse Domains through World Models."

The Dreamer family learns a world model in a compact latent space and trains the policy entirely through imagined rollouts within that model using backpropagation through the learned dynamics.

**DreamerV1** uses a Recurrent State-Space Model (RSSM) combining deterministic and stochastic components. The actor and critic are trained on imagined trajectories via latent-space backpropagation.

**DreamerV2** replaced Gaussian latent variables with **discrete categorical** representations, improving robustness. It was the first model-based method to match model-free performance across the full Atari benchmark.

**DreamerV3** introduced a suite of normalization and scaling techniques (symlog predictions, percentile-based return scaling, fixed hyperparameters) that allow a single set of hyperparameters to work across vastly different domains — from Atari to continuous control to Minecraft (diamond collection without human demonstrations or reward shaping).

**Strengths:** Exceptional sample efficiency, learns transferable world representations, single hyperparameter set across domains (V3).
**Limitations:** Model errors compound over long imagination horizons, struggles with environments that are hard to model, additional complexity of maintaining a world model.
**Use cases:** Sample-constrained settings, complex visual environments, long-horizon tasks.

### 4.3 MuZero

**Paper:** Schrittwieser et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." Nature.

MuZero extends AlphaZero (see below) to environments where the rules are **unknown**. It learns three neural networks: a **representation function** that encodes observations into a hidden state, a **dynamics function** that predicts the next hidden state and reward given a state and action, and a **prediction function** that outputs a policy and value from a hidden state.

Crucially, MuZero's learned model operates entirely in a latent space — it never reconstructs observations. It only models what is necessary for planning: reward, value, and policy. Planning is done via Monte Carlo Tree Search (MCTS) in this learned latent space.

**Strengths:** Combines planning power of AlphaZero with model learning, works without known environment dynamics, state-of-the-art on Atari and board games simultaneously.
**Limitations:** Computationally expensive (MCTS at every step), difficult to scale to very long horizons, model accuracy degrades over many prediction steps.
**Use cases:** Board games, Atari, environments where planning significantly outperforms reactive policies.

### 4.4 AlphaZero / AlphaGo

**Papers:** Silver et al. (2016, 2017). "Mastering the game of Go with deep neural networks and tree search" / "Mastering the game of Go without human knowledge" / "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play."

**AlphaGo** combined supervised learning from human expert games with RL via self-play and MCTS. **AlphaGo Zero** eliminated human data entirely, learning purely from self-play. **AlphaZero** generalized this to Chess and Shogi with the same algorithm and hyperparameters.

The architecture uses a single neural network with two output heads (policy and value), trained on self-play data. During play, MCTS uses the network to evaluate positions and guide the search. The search results (visit counts) serve as an improved policy target for the next round of training.

**Strengths:** Superhuman performance in perfect-information board games, elegant self-play curriculum, combines learning with planning.
**Limitations:** Requires known game rules (perfect simulator), computationally very expensive, designed for two-player zero-sum games.
**Use cases:** Board games, combinatorial optimization, protein structure evaluation (AlphaFold used related ideas).

### 4.5 MBPO (Model-Based Policy Optimization)

**Paper:** Janner et al. (2019). "When to Trust Your Model: Model-Based Policy Optimization." NeurIPS.

MBPO learns an ensemble of environment models and uses them to generate short synthetic rollouts that augment the real replay buffer. A model-free algorithm (typically SAC) is then trained on the mixed real and synthetic data. The key insight is a theoretical bound showing that short model rollouts have bounded error even with imperfect models, so the rollout horizon should be kept short and gradually extended.

**Strengths:** Significantly more sample-efficient than pure model-free methods, simple to combine with existing algorithms, theoretical grounding for rollout length.
**Limitations:** Model ensemble adds computational overhead, still limited by model accuracy for longer rollouts.
**Use cases:** Sample-constrained continuous control, any setting where environment interaction is expensive.

## 5. Offline (Batch) RL

Offline RL learns policies from a **fixed dataset** of previously collected transitions, without any further environment interaction. This is crucial for domains where online exploration is dangerous, expensive, or impossible (e.g., healthcare, autonomous driving, industrial control).

The core challenge is **distributional shift**: the learned policy may want to take actions that are poorly represented in the dataset, leading to wildly incorrect Q-value estimates for those out-of-distribution actions.

### 5.1 CQL (Conservative Q-Learning)

**Paper:** Kumar et al. (2020). "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS.

CQL adds a regularizer to the Q-learning objective that **penalizes Q-values for out-of-distribution actions** and **boosts Q-values for in-distribution actions**:

$$\min_Q \; \alpha \cdot \mathbb{E}_{s}\left[\log \sum_a \exp Q(s, a) - \mathbb{E}_{a \sim \hat{\pi}_\beta}[Q(s, a)]\right] + \text{standard TD loss}$$

This produces a conservative (lower-bound) estimate of the true Q-function, ensuring that the policy doesn't exploit overestimated Q-values for unseen actions.

**Strengths:** Theoretically grounded lower bound, compatible with any Q-learning variant, effective in practice.
**Limitations:** Can be overly conservative, performance depends on $\alpha$ tuning.

### 5.2 BCQ (Batch-Constrained Q-Learning)

**Paper:** Fujimoto et al. (2019). "Off-Policy Deep Reinforcement Learning without Exploration." ICML.

BCQ constrains the policy to only select actions that are similar to those present in the dataset. It uses a conditional VAE trained on the dataset to generate candidate actions, then perturbs and ranks them using a Q-network. This directly prevents the agent from taking out-of-distribution actions.

### 5.3 IQL (Implicit Q-Learning)

**Paper:** Kostrikov et al. (2022). "Offline Reinforcement Learning with Implicit Q-Learning." ICLR.

IQL avoids querying out-of-distribution actions entirely by using **expectile regression** to learn the value function. Instead of computing $\max_a Q(s, a)$, which requires evaluating unseen actions, IQL trains the value function to approximate an upper expectile of the Q-value distribution over in-dataset actions. This implicitly performs policy improvement without ever evaluating actions outside the dataset.

**Strengths:** Simple, avoids explicit action constraint mechanisms, strong empirical performance.
**Limitations:** Expectile parameter requires tuning, conservative by design.

### 5.4 Decision Transformer

**Paper:** Chen et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling." NeurIPS.

Decision Transformer reframes offline RL as a **sequence modeling** problem. Instead of learning value functions or policies, it trains a GPT-style transformer on sequences of (return-to-go, state, action) tuples. At inference time, the desired return-to-go is provided as a conditioning signal, and the model autoregressively generates actions that are expected to achieve that return level.

This approach sidesteps the bootstrapping, value estimation, and distributional shift problems of traditional offline RL. The transformer simply learns to predict actions conditioned on the desired outcome.

**Strengths:** Simple supervised learning objective, leverages powerful transformer architectures, avoids Q-learning instabilities, return conditioning enables flexible goal specification.
**Limitations:** Requires high-return trajectories in the dataset, no stitching (struggles to combine good sub-trajectories from different episodes), performance generally below CQL/IQL on standard benchmarks.
**Use cases:** Offline RL, settings where sequence modeling infrastructure is already available, exploratory research into RL-as-sequence-modeling.

### 5.5 Trajectory Transformer

**Paper:** Janner et al. (2021). "Offline Reinforcement Learning as One Big Sequence Modeling Problem." NeurIPS.

Similar philosophy to Decision Transformer but models the full trajectory (states, actions, and rewards) jointly as a single sequence. It discretizes continuous values into bins and uses beam search at inference time to find high-reward action sequences. Unlike Decision Transformer, it can **stitch** together good subsequences from different trajectories because the beam search explicitly optimizes over possible futures.

## 6. Multi-Agent RL (MARL)

Multi-agent RL addresses environments with multiple interacting agents, where each agent's optimal behavior depends on the actions of others. This introduces challenges of non-stationarity (other agents are also learning), credit assignment (attributing team reward to individual agents), and scalability.

### 6.1 QMIX

**Paper:** Rashid et al. (2018). "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." ICML.

QMIX addresses cooperative multi-agent tasks by factoring the joint action-value function $Q_{tot}$ into individual per-agent Q-values $Q_i$, combined through a **monotonic mixing network** whose weights are produced by a hypernetwork conditioned on the global state. The monotonicity constraint ensures that $\arg\max$ over $Q_{tot}$ can be decomposed into per-agent $\arg\max$ operations, enabling decentralized execution.

**Strengths:** Centralized training with decentralized execution (CTDE), scalable to many agents.
**Limitations:** Monotonicity constraint limits expressiveness, cannot represent all cooperative relationships.

### 6.2 MAPPO (Multi-Agent PPO)

**Paper:** Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS.

MAPPO applies PPO independently to each agent (with shared or separate policies) using a centralized critic that observes the global state. Despite its simplicity, MAPPO matched or exceeded purpose-built MARL algorithms on standard benchmarks (SMAC, Hanabi, MPE), demonstrating that a well-tuned single-agent algorithm can be surprisingly effective in multi-agent settings.

### 6.3 MADDPG (Multi-Agent DDPG)

**Paper:** Lowe et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NeurIPS.

MADDPG uses centralized critics (one per agent) that observe all agents' actions and observations during training, while each agent's actor only observes its own local information during execution. This CTDE approach allows handling of mixed cooperative-competitive settings where agents may have different or conflicting objectives.

### 6.4 Self-Play and Population-Based Methods

Self-play methods (as in AlphaZero) train agents against copies of themselves or historical versions, creating an automatic curriculum of increasing difficulty. **Population-Based Training (PBT)** evolves a population of agents with different hyperparameters in parallel, periodically replacing poorly performing agents with mutated copies of better ones. OpenAI Five combined PPO with self-play and team reward to master Dota 2.

## 7. Hierarchical RL

Hierarchical RL decomposes complex, long-horizon tasks into multiple levels of abstraction. A high-level policy selects subgoals or options, and a low-level policy executes primitive actions to achieve them. This enables temporal abstraction, faster exploration, and transfer of low-level skills across tasks.

### 7.1 Option-Critic

**Paper:** Bacon et al. (2017). "The Option-Critic Architecture." AAAI.

The options framework (Sutton et al., 1999) formalizes temporally extended actions as "options," each consisting of an initiation set, an internal policy, and a termination condition. The Option-Critic architecture learns all components end-to-end using policy gradient, discovering meaningful options without manual specification.

### 7.2 HAM / MAXQ

Classical hierarchical frameworks. **HAM** (Hierarchy of Abstract Machines) constrains the policy space using partially specified finite state machines. **MAXQ** decomposes the value function into a hierarchy of sub-task values. Both require significant manual design of the hierarchy structure.

### 7.3 Feudal Networks (FeUdal Networks / FuN)

**Paper:** Vezhnevets et al. (2017). "FeUdal Networks for Hierarchical Reinforcement Learning." ICML.

FuN uses a Manager-Worker hierarchy. The **Manager** operates at a slower timescale, producing directional goals in a learned latent space. The **Worker** operates at every timestep and is rewarded for moving in the direction specified by the Manager. The Manager's goals are learned end-to-end and don't need to correspond to any pre-defined semantics.

### 7.4 Goal-Conditioned RL / HER

**Paper:** Andrychowicz et al. (2017). "Hindsight Experience Replay." NeurIPS.

Goal-conditioned RL trains policies of the form $\pi(a|s, g)$ where $g$ is a desired goal. **Hindsight Experience Replay (HER)** addresses the sparse reward problem by relabeling failed trajectories with goals that were actually achieved — turning every trajectory into a successful one for some goal. This dramatically accelerates learning in sparse-reward, goal-conditioned settings.

HER is a technique, not a standalone algorithm, and is typically combined with off-policy methods like DDPG, TD3, or SAC.

## 8. Exploration Methods

Efficient exploration is a fundamental challenge in RL, particularly in environments with sparse, deceptive, or delayed rewards. Standard $\varepsilon$-greedy or entropy bonuses are often insufficient. Dedicated exploration methods provide stronger incentives to visit novel states or gather informative experience.

### 8.1 ICM (Intrinsic Curiosity Module)

**Paper:** Pathak et al. (2017). "Curiosity-driven Exploration by Self-Supervised Prediction." ICML.

ICM generates an **intrinsic reward** based on prediction error. It learns a forward model that predicts the next state's representation given the current state and action. When the prediction error is high, the state is considered novel and a curiosity bonus is added to the extrinsic reward. Crucially, the representations are learned via an inverse model (predicting the action given two states), which filters out unpredictable but task-irrelevant aspects of the environment (e.g., random noise).

### 8.2 RND (Random Network Distillation)

**Paper:** Burda et al. (2019). "Exploration by Random Network Distillation." ICLR.

RND uses a simpler curiosity mechanism: a fixed, randomly initialized target network and a predictor network trained to match the target's outputs. For familiar states, the predictor matches well (low error). For novel states, the predictor hasn't been trained on similar inputs and produces high error, which serves as the intrinsic reward. RND was the first algorithm to achieve meaningful progress on Montezuma's Revenge without demonstrations.

### 8.3 Go-Explore

**Paper:** Ecoffet et al. (2021). "First return, then explore." Nature.

Go-Explore addresses the "detachment" problem — where an exploring agent forgets how to return to previously discovered promising states. It maintains an archive of visited states, periodically **returns** to underexplored states (via deterministic restoration or goal-conditioned policies), and **explores** from there. A robustification phase then trains a policy to reliably reproduce the best discovered trajectories. Go-Explore achieved superhuman scores on Montezuma's Revenge and Pitfall.

## 9. Imitation Learning and Inverse RL

These methods learn behavior from demonstrations rather than (or in addition to) reward signals. They are especially useful when defining a reward function is difficult but expert demonstrations are available.

### 9.1 Behavioral Cloning (BC)

The simplest approach: treat the expert's demonstrations as a supervised learning dataset and train a policy to predict the expert's action given the state. Fast and easy but suffers from **compounding errors** — small mistakes push the policy into states the expert never visited, where further mistakes accumulate.

### 9.2 DAgger (Dataset Aggregation)

**Paper:** Ross et al. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." AISTATS.

DAgger addresses compounding errors by iteratively running the learned policy, collecting states it visits, querying the expert for the correct action in those states, and retraining. This ensures the policy is trained on states it actually encounters, rather than only expert-visited states. It requires an interactive expert (which is not always available).

### 9.3 GAIL (Generative Adversarial Imitation Learning)

**Paper:** Ho & Ermon (2016). "Generative Adversarial Imitation Learning." NeurIPS.

GAIL frames imitation learning as a GAN problem. A **discriminator** learns to distinguish between expert trajectories and the agent's trajectories, and the **policy** is trained (via RL, typically TRPO or PPO) to fool the discriminator. The discriminator's output serves as the reward signal. GAIL can recover complex behaviors without explicit reward engineering and generalizes better than behavioral cloning.

**Strengths:** No need for reward function design, recovers nuanced expert behavior, handles continuous control well.
**Limitations:** Requires RL training loop (slow), mode collapse, sensitive to discriminator capacity.

### 9.4 IRL (Inverse Reinforcement Learning)

**Paper:** Ng & Russell (2000). "Algorithms for Inverse Reinforcement Learning." ICML.

IRL infers the reward function that the expert is implicitly optimizing, then uses standard RL to learn a policy that maximizes that inferred reward. **Maximum Entropy IRL** (Ziebart et al., 2008) and **Maximum Causal Entropy IRL** provide principled probabilistic frameworks. The recovered reward function can generalize to new environments or tasks, unlike the policy itself. IRL is foundational to RLHF approaches used in LLM alignment.

## 10. Meta-RL

Meta-RL algorithms learn to learn — they acquire a learning algorithm (or adaptive prior) from a distribution of tasks, enabling rapid adaptation to new tasks with minimal data. This is the RL counterpart of few-shot learning.

### 10.1 MAML (Model-Agnostic Meta-Learning)

**Paper:** Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML.

MAML learns an initialization of network parameters that can be quickly fine-tuned to a new task with a few gradient steps. The outer loop optimizes the initialization across many tasks, while the inner loop adapts to each specific task. For RL, the inner loop collects trajectories and updates the policy, and the outer loop optimizes for expected return after adaptation.

**Strengths:** Principled, general framework, works with any gradient-based model.
**Limitations:** Second-order gradients are expensive, inner loop requires multiple RL episodes, limited to shallow adaptation.

### 10.2 RL² (Learning to Reinforcement Learn)

**Paper:** Duan et al. (2016). "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning." arXiv:1611.02779.

RL² trains an RNN policy across many tasks, where the hidden state of the RNN effectively implements a learning algorithm. The RNN receives the action, reward, and termination signal from the previous step as additional input, allowing it to adapt its behavior based on accumulated experience within an episode or across episodes. The "learning algorithm" is implicitly encoded in the recurrent dynamics.

**Strengths:** Fully end-to-end, can discover complex adaptation strategies, no explicit inner loop.
**Limitations:** Requires many training tasks, limited generalization beyond the training task distribution, memory bottleneck of RNNs.

### 10.3 PEARL (Probabilistic Embeddings for Actor-critic RL)

**Paper:** Rakelly et al. (2019). "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables." ICML.

PEARL infers a probabilistic **task embedding** from a small number of transitions using an inference network, then conditions the policy and value function on this embedding. It's off-policy (using SAC as the base algorithm), making it more sample-efficient than MAML for meta-RL. The task embedding captures the essential characteristics of a task (e.g., goal location, dynamics parameters) in a compact latent vector.

## 11. Emerging Approaches

### 11.1 Diffusion Policies

**Paper:** Chi et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS.

Diffusion policies apply denoising diffusion probabilistic models to action generation. The policy represents the action distribution as a learned denoising process: starting from Gaussian noise, it iteratively refines through learned denoising steps to produce an action (or action sequence). This is conditioned on the current observation (often visual).

The key advantage is **expressiveness**: diffusion models can represent complex, multimodal action distributions far better than Gaussian policies (SAC) or deterministic policies (TD3). This is critical in manipulation tasks where multiple valid solutions exist for the same observation.

**Strengths:** Highly expressive multimodal distributions, excels at visuomotor manipulation, strong performance from limited demonstrations.
**Limitations:** Slow inference (multiple denoising steps), not yet proven at scale for online RL, primarily used in imitation learning settings.
**Use cases:** Robot manipulation, visuomotor control, imitation learning from human demonstrations.

### 11.2 RLHF (Reinforcement Learning from Human Feedback)

**Papers:** Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences." NeurIPS. / Ouyang et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS.

RLHF trains a **reward model** from human preference comparisons (which of two outputs is better?), then uses RL (typically PPO) to fine-tune a policy (e.g., a language model) to maximize that learned reward while staying close to the original supervised model via a KL penalty.

This pipeline was central to ChatGPT, Claude, and other aligned LLMs. Variants include **DPO** (Direct Preference Optimization), which eliminates the explicit reward model and RL step by directly optimizing the policy from preference data via a classification loss.

**Strengths:** Aligns AI behavior with human preferences without hand-crafted reward functions, scales to complex generative tasks.
**Limitations:** Reward model can be gamed (reward hacking), preference data is expensive and noisy, KL constraint limits improvement.
**Use cases:** LLM alignment, text generation, image generation (DALL-E), any domain where human preferences define quality.

### 11.3 DPO (Direct Preference Optimization)

**Paper:** Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS.

DPO reformulates RLHF as a simple classification problem. By deriving the closed-form solution for the optimal policy under the RLHF objective, DPO shows that the reward model is implicit in the policy itself. Training reduces to a binary cross-entropy loss on preference pairs:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

where $y_w$ and $y_l$ are the preferred and dispreferred responses respectively.

**Strengths:** No reward model training, no RL loop, simple to implement, stable training.
**Limitations:** Less flexible than RLHF (no iterative reward model improvement), can underperform RLHF at scale.

### 11.4 GRPO (Group Relative Policy Optimization)

**Paper:** Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models."

GRPO eliminates the critic (value network) from PPO by estimating advantages **relative to a group of sampled responses**. For each prompt, multiple completions are generated, their rewards are computed, and the advantage of each completion is its reward minus the group mean, normalized by the group standard deviation. This avoids the complexity and instability of training a value function for language model optimization.

**Strengths:** Simpler than PPO-based RLHF (no critic), memory-efficient, effective for LLM reasoning tasks.
**Limitations:** Requires generating multiple completions per prompt (increased inference cost), advantage estimates can be noisy with small groups.
**Use cases:** LLM fine-tuning for mathematical reasoning, code generation, and other verifiable domains.

### 11.5 Online DPO and Iterative RLHF

Recent work explores **online** variants of DPO where the policy generates its own completions, collects new preferences (from a reward model or human), and iterates — bridging the gap between static DPO and full RLHF. **SPIN** (Self-Play Fine-Tuning), **IPO** (Identity Preference Optimization), and **KTO** (Kahneman-Tversky Optimization) are related methods that modify the preference optimization objective for improved stability or theoretical properties.

### 11.6 Foundation Models for Decision-Making

A growing body of work applies pretrained foundation models (LLMs, vision-language models) to RL and decision-making:

**Gato** (Reed et al., 2022) trained a single transformer as a generalist agent across Atari, robotics, captioning, and dialogue, treating all modalities as token sequences.

**RT-2** (Brohan et al., 2023) fine-tuned a vision-language model to directly output robot actions as text tokens, transferring web-scale knowledge to physical manipulation.

**SayCan** (Ahn et al., 2022) combined an LLM's semantic knowledge ("what to do") with learned affordance functions ("what is physically possible") to ground language instructions in robotic capabilities.

This paradigm represents a shift from training RL agents from scratch to leveraging pretrained models as priors or planners, with RL fine-tuning for specific embodiments and tasks.

## Algorithm Selection Guide

| Setting | Recommended Algorithms |
||-|
| Discrete actions, single agent | DQN, Rainbow, PPO |
| Continuous actions, sample-efficient | SAC, TD3 |
| Continuous actions, general purpose | PPO |
| Large-scale parallel training | PPO, IMPALA |
| Fixed offline dataset | CQL, IQL, Decision Transformer |
| Sparse rewards with goals | HER + SAC/TD3 |
| Expert demonstrations available | GAIL, BC, Diffusion Policy |
| Multi-agent cooperative | MAPPO, QMIX |
| Board games / planning | AlphaZero, MuZero |
| Sample-constrained visual tasks | DreamerV3 |
| LLM alignment | RLHF (PPO), DPO, GRPO |
| Few-shot task adaptation | MAML, PEARL |

## References

- Andrychowicz et al. (2017). "Hindsight Experience Replay." NeurIPS.
- Bacon et al. (2017). "The Option-Critic Architecture." AAAI.
- Bellemare et al. (2017). "A Distributional Perspective on Reinforcement Learning." ICML.
- Berner et al. (2019). "Dota 2 with Large Scale Deep Reinforcement Learning." arXiv:1912.06680.
- Brohan et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control."
- Burda et al. (2019). "Exploration by Random Network Distillation." ICLR.
- Chen et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling." NeurIPS.
- Chi et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS.
- Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences." NeurIPS.
- Dabney et al. (2018). "Distributional Reinforcement Learning with Quantile Regression." AAAI.
- Dabney et al. (2018). "Implicit Quantile Networks for Distributional Reinforcement Learning." ICML.
- Duan et al. (2016). "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning." arXiv:1611.02779.
- Ecoffet et al. (2021). "First return, then explore." Nature.
- Espeholt et al. (2018). "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures." ICML.
- Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML.
- Fujimoto et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." ICML.
- Fujimoto et al. (2019). "Off-Policy Deep Reinforcement Learning without Exploration." ICML.
- Ha & Schmidhuber (2018). "World Models." NeurIPS.
- Haarnoja et al. (2018). "Soft Actor-Critic." ICML.
- Hafner et al. (2020). "Dream to Control: Learning Behaviors by Latent Imagination." ICLR.
- Hafner et al. (2021). "Mastering Atari with Discrete World Models." ICLR.
- Hafner et al. (2023). "Mastering Diverse Domains through World Models." arXiv:2301.04104.
- Hessel et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning." AAAI.
- Ho & Ermon (2016). "Generative Adversarial Imitation Learning." NeurIPS.
- Janner et al. (2019). "When to Trust Your Model: Model-Based Policy Optimization." NeurIPS.
- Janner et al. (2021). "Offline Reinforcement Learning as One Big Sequence Modeling Problem." NeurIPS.
- Kapturowski et al. (2019). "Recurrent Experience Replay in Distributed Reinforcement Learning." ICLR.
- Kostrikov et al. (2022). "Offline Reinforcement Learning with Implicit Q-Learning." ICLR.
- Kumar et al. (2020). "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS.
- Lillicrap et al. (2016). "Continuous control with deep reinforcement learning." ICLR.
- Lowe et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NeurIPS.
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.
- Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." ICML.
- Ng & Russell (2000). "Algorithms for Inverse Reinforcement Learning." ICML.
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS.
- Pathak et al. (2017). "Curiosity-driven Exploration by Self-Supervised Prediction." ICML.
- Rafailov et al. (2023). "Direct Preference Optimization." NeurIPS.
- Rakelly et al. (2019). "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables." ICML.
- Rashid et al. (2018). "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." ICML.
- Reed et al. (2022). "A Generalist Agent." arXiv:2205.06175.
- Ross et al. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." AISTATS.
- Schaul et al. (2016). "Prioritized Experience Replay." ICLR.
- Schrittwieser et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." Nature.
- Schulman et al. (2015). "Trust Region Policy Optimization." ICML.
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
- Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models."
- Silver et al. (2016). "Mastering the game of Go with deep neural networks and tree search." Nature.
- Silver et al. (2017). "Mastering the game of Go without human knowledge." Nature.
- van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
- Vezhnevets et al. (2017). "FeUdal Networks for Hierarchical Reinforcement Learning." ICML.
- Wang et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML.
- Williams (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine Learning.
- Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS.

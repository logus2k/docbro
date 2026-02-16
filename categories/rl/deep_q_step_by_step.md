# Deep Q-Networks - Step by Step

## The Two Networks Architecture

Deep Q-Learning uses two neural networks with identical architecture:

**Q-Network (Frank's conscience):** Takes a state as input. The output layer has one neuron per possible action, and each neuron outputs a Q-value — a number that quantifies how good that action is expected to be from this state. For LunarLander-v3 with 4 actions:

```
State (8) → Hidden Layers → [Q(do nothing), Q(fire left), Q(fire main), Q(fire right)]
```

This is the network that actively learns. Its weights are updated by backpropagation at every training step.

**Target Network (the ideal conscience):** Has the exact same architecture as the Q-Network. It serves as a stable reference point — a frozen snapshot of what the Q-Network looked like a few thousand steps ago. It is not trained directly. Its weights are periodically copied from the Q-Network.

---

## From Q-Table to Q-Network

In simple environments with few states (like a 9-square grid), the agent can use a **Q-table** — a lookup table with one row per state and one column per action, where each cell holds a Q-value. The agent just looks up its current state and picks the action with the highest value.

But when the number of states is very large or continuous (like LunarLander's 8-dimensional floating-point state), the table becomes impossibly large. It wouldn't fit in memory, and most states would never be visited exactly twice.

The solution: replace the table with a **neural network** that takes any state as input and outputs Q-values for all actions. The network learns a smooth function that generalizes — nearby states produce similar Q-values, so the agent doesn't need to visit every possible state to make good decisions.

Q-learning learns values in a Q-table. **Deep Q-learning** learns the parameters (weights) of a Q-network that approximates those values.

---

## Phase 1: Data Collection

Before training can begin, the agent needs experience. It interacts with the environment and stores what happens.

**Step 1:** The Q-Network is **randomly initialized**. Its Q-values are meaningless at this point — just random numbers. The target network receives an identical copy of these random weights.

**Step 2:** The agent observes an initial state $s$, for example $[0.12, 0.85, -0.03, -0.15, 0.08, 0.02, 0.0, 0.0]$.

**Step 3:** The state is passed into the **Q-Network**. It outputs one Q-value per action, for example:

| Action | Q-value |
|--------|---------|
| Do nothing | 0.3 |
| Fire left | -0.1 |
| Fire main | 0.5 |
| Fire right | 0.2 |

**Step 4:** The agent chooses an action using **epsilon-greedy** selection. With probability $\varepsilon$, it ignores the Q-values and picks a random action (exploration). With probability $1 - \varepsilon$, it picks the action with the highest Q-value (exploitation). Early in training, $\varepsilon$ is high (near 1.0), so most actions are random.

**Step 5:** The agent takes the action in the environment. The environment returns a reward $r$ (e.g., +0.3) and the agent arrives at a new state $s'$.

**Step 6:** The agent stores a **quadruple** of four values into the **experience replay buffer**:

$$(s, \; a, \; r, \; s')$$

That is: the state it was in, the action it took, the reward it received, and the new state it arrived at.

**Step 7:** The agent moves to the new state $s'$ and repeats from Step 3. This continues for many steps, filling the replay buffer with quadruples of experience.

The experience replay buffer accumulates independent snapshots of what happened during interaction. This data will be used to train the Q-Network.

---

## Phase 2: Training the Q-Network

Once the replay buffer has enough data, training begins. At every environment step, a training update is performed.

**Step 1:** Sample a **batch** of quadruples randomly from the experience replay buffer. For clarity, let's trace through a single quadruple $(s, a, r, s')$:

- State $s$: the agent was at position $[0.12, 0.85, -0.03, ...]$
- Action $a$: fire main engine (action 2)
- Reward $r$: +0.3
- Next state $s'$: the agent moved to $[0.11, 0.80, -0.02, ...]$

**Step 2:** Pass the current state $s$ into the **Q-Network**. It outputs Q-values for all actions. We take the Q-value corresponding to the action in our quadruple (action 2, fire main). Say it outputs $Q(s, a) = 3.1$.

This is **Frank's current conscience** — what the Q-Network currently believes about this (state, action) pair.

**Step 3:** Pass the next state $s'$ into the **Target Network**. It outputs Q-values for all actions. We take the **highest** Q-value:

| Action | Q-value from Target Network |
|--------|---------------------------|
| Do nothing | 2.1 |
| Fire left | 1.3 |
| Fire main | 4.7 |
| Fire right | 0.9 |

The highest Q-value is 4.7.

**Step 4:** Compute the **target Q-value** — the ideal conscience — by adding the actual reward to the discounted maximum Q-value from the target network:

$$y = r + \gamma \cdot \max_{a'} Q(s', a'; \theta^-) = 0.3 + 0.99 \times 4.7 = 4.953$$

This represents: "you got +0.3 now, and the best possible future from where you landed is worth about 4.7."

**Step 5:** Now we have two values:

| | Value | Meaning |
|--|-------|---------|
| Q-Network output | 3.1 | Frank's current conscience — what it believes |
| Target value | 4.953 | The ideal conscience — what it should believe |

**Step 6:** Compute the **mean squared loss** between these two values:

$$\mathcal{L} = (y - Q(s, a; \theta))^2 = (4.953 - 3.1)^2 = 3.43$$

In practice, this is averaged across all quadruples in the batch.

**Step 7:** **Backpropagate** this loss through the **Q-Network only**. The optimizer adjusts the Q-Network's weights so that next time it sees this state and action, it will output a value closer to 4.953 instead of 3.1.

The **target network remains unchanged**. No gradients flow through it. It just provided a stable reference number.

**Step 8:** Repeat for the next batch. The Q-Network's weights shift slightly with every update. Frank's conscience improves step by step.

---

## Phase 3: Updating the Target Network

After a fixed number of training steps (e.g., every 10,000 steps), the target network is updated:

$$\theta^- \leftarrow \theta$$

The Q-Network's current weights are copied into the target network. The target network — the ideal conscience — is now refreshed with the Q-Network's improved knowledge.

Between updates, the target network stays frozen. This is what provides training stability: the Q-Network learns against a fixed reference, not a moving target.

The cycle then continues:

```
Steps 1–10,000:       Target network frozen
                       Q-Network learns against stable targets
                       Frank's conscience improves gradually

Step 10,001:           Target network updated ← Q-Network weights
                       New, better reference point established

Steps 10,001–20,000:  Target network frozen again
                       Q-Network learns against improved targets
                       Frank's conscience improves further

Step 20,001:           Target network updated again
...
```

Each update cycle, the ideal conscience gets better because it inherits the Q-Network's improved understanding. Better targets lead to better learning, which leads to better targets at the next sync. The system bootstraps itself into increasingly accurate Q-values.

---

## Why Experience Replay?

The replay buffer serves a critical purpose: it **breaks temporal correlation**.

Without replay, the agent would train on consecutive transitions — step 1, step 2, step 3 — which are highly correlated (each state is almost identical to the previous one). Training a neural network on correlated sequential data causes it to overfit to recent experience and forget earlier knowledge.

By sampling randomly from the buffer, each training batch contains transitions from different episodes, different times, and different situations. This mimics the independent and identically distributed (i.i.d.) data assumption that makes neural network training stable.

A secondary benefit is **data reuse**: each quadruple in the buffer can be sampled many times across different training batches, extracting more learning from each unit of experience.

---

## The Complete Cycle

```
1. DATA COLLECTION
   Q-Network (randomly initialized) produces Q-values
   Agent selects actions via epsilon-greedy
   Stores (state, action, reward, next state) in replay buffer
   Repeat for many steps

2. TRAINING (at every step after buffer has enough data)
   Sample a random batch from the replay buffer
   Q-Network produces Q-value for the stored (state, action) → current conscience
   Target Network produces max Q-value for the next state → ideal conscience
   Target = reward + γ × max Q-value from target network
   Loss = (target − Q-Network prediction)²
   Backpropagate through Q-Network only → Frank learns
   Target Network remains frozen

3. TARGET UPDATE (every N steps)
   Copy Q-Network weights → Target Network
   Ideal conscience is refreshed

4. REPEAT
   Epsilon decays over time (less random, more greedy)
   Q-Network improves → better actions → better data → better learning
   Frank becomes a champ
```

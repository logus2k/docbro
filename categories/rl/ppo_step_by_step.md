# PPO Step by Step

## The Two Networks Architecture

PPO uses two separate neural networks that learn together:

**Policy Network (Actor):** Takes a state as input. The output layer has one neuron per possible action, and each neuron outputs the probability of taking that action. For LunarLander-v3 with 4 actions:

```
State (8) → Hidden Layers → [P(do nothing), P(fire left), P(fire main), P(fire right)]
```

The output is a probability distribution — the four values sum to 1.

**Value Function Network (Critic):** Takes a state as input. The output layer also has one neuron per action, and each neuron outputs a Q-value — a number quantifying how good that action is expected to be from this state. For LunarLander-v3:

```
State (8) → Hidden Layers → [Q(do nothing), Q(fire left), Q(fire main), Q(fire right)]
```

---

## Phase 1: Collecting Data

The agent interacts with the environment for a full episode (or a fixed number of steps), storing everything it experiences. No learning happens yet.

**Step 1:** The agent observes the current state $s$, for example $[0.12, 0.85, -0.03, -0.15, 0.08, 0.02, 0.0, 0.0]$.

**Step 2:** The state is passed into the **policy network**. It outputs a probability distribution over actions, for example:

| Action | Probability |
|--------|------------|
| Do nothing | 0.10 |
| Fire left | 0.15 |
| Fire main | 0.60 |
| Fire right | 0.15 |

**Step 3:** The agent **samples** from this distribution to determine the actual action. With these probabilities, "fire main" is most likely but any action could be picked. Say it samples action 2 (fire main).

**Step 4:** The agent takes the action. The environment returns a reward $r$ (e.g., +0.3) and the next state $s'$.

**Step 5:** The agent stores a tuple of four values:

$$(s, \; a, \; r, \; \pi_{old}(a|s))$$

That is: the state, the action taken, the reward received, and the probability the policy network assigned to that action (0.60 in this case).

**Step 6:** Repeat from Step 1 for the entire episode or for a fixed number of steps.

At the end, we have a **batch** of stored tuples — a complete record of one rollout. We could collect multiple batches (multiple episodes) before training if desired.

---

## Phase 2: Training the Value Function Network

We now use the collected batch to train both networks. Let's start with the value function network.

**Step 1:** For each timestep in the batch, compute the **actual future reward**. This is the sum of discounted future rewards from that timestep onward:

$$G_t = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \dots$$

This tells us: "how good did things actually turn out from this point?"

**Step 2:** For each timestep, pass the stored state $s_t$ into the **value function network**. It outputs Q-values for all actions. We look at the Q-value for the specific action that was taken in our stored tuple. This tells us: "how good did the network expect things to be?"

**Step 3:** Compute the **advantage** for each timestep — the difference between actual and expected:

$$A_t = G_t - Q(s_t, a_t)$$

If $A_t$ is positive, the action turned out better than expected. If negative, worse than expected.

**Step 4:** Square the advantage for each timestep:

$$A_t^2$$

**Step 5:** Take the average of the squared advantages across the entire batch. This single number is the **value function loss**:

$$L^V = \frac{1}{N} \sum_{t=1}^{N} A_t^2$$

**Step 6:** Backpropagate this loss through the value function network and update its weights. The network learns to produce Q-values that better match the actual returns it observes.

---

## Phase 3: Training the Policy Network

At the same time, we train the policy network using the same batch of data and the same advantages.

**Step 1:** Pass the batch of states into the **current** policy network. For each state, it outputs updated action probabilities. We take the probability it now assigns to the action that was taken: $\pi_\theta(a_t|s_t)$.

**Step 2:** Compute the **probability ratio** for each timestep:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

The numerator is the probability the current (possibly updated) policy assigns to the action. The denominator is the probability stored from the collection phase. If the ratio is above 1, the current policy finds this action more likely than before. Below 1, less likely.

**Step 3:** Multiply the ratio by the advantage:

$$r_t(\theta) \cdot A_t$$

This produces one value per timestep. Hold on to these values.

**Step 4:** Separately, **clip** the probability ratio to prevent large policy changes:

$$\text{clip}(r_t(\theta), \; 1 - \epsilon, \; 1 + \epsilon)$$

This constrains the ratio to the range $[1 - \epsilon, \; 1 + \epsilon]$ (typically $[0.8, \; 1.2]$ with $\epsilon = 0.2$).

**Step 5:** Multiply the clipped ratio by the advantage:

$$\text{clip}(r_t(\theta), \; 1 - \epsilon, \; 1 + \epsilon) \cdot A_t$$

Now for each timestep we have two values: the unclipped version (from Step 3) and the clipped version (from Step 5).

**Step 6:** Take the **minimum** of the two values for each timestep:

$$\min\left(r_t(\theta) \cdot A_t, \;\; \text{clip}(r_t(\theta), \; 1 - \epsilon, \; 1 + \epsilon) \cdot A_t\right)$$

The minimum ensures the more conservative (cautious) update is used. This is the core of PPO — it balances effective updates with stability.

**Step 7:** Take the average across the batch to get a single number — the **policy loss**:

$$L^{CLIP} = \frac{1}{N} \sum_{t=1}^{N} \min\left(r_t(\theta) \cdot A_t, \;\; \text{clip}(r_t(\theta), \; 1 - \epsilon, \; 1 + \epsilon) \cdot A_t\right)$$

**Step 8:** Backpropagate this loss through the policy network and update its weights. The network learns to make good actions more probable and bad actions less probable, within safe bounds.

---

## Phase 4: Repeat

Both networks have been updated. The agent returns to Phase 1 — collects a new batch of data using the now-improved policy network, trains both networks again, and repeats.

Over many iterations, the policy network produces better and better action probabilities, and the value function network produces more accurate Q-value estimates. The two networks improve together: better Q-values produce better advantage estimates, which produce better policy updates, which produce better data, which produce better Q-values.

---

## Summary

The PPO training cycle:

```
1. COLLECT DATA
   Policy network outputs action probabilities
   Agent samples actions from the distribution
   Store (state, action, reward, probability) tuples

2. TRAIN VALUE FUNCTION NETWORK
   Compute actual future rewards from the batch
   Compare with the network's Q-value predictions
   Advantage = actual - predicted
   Loss = average of squared advantages
   Backpropagate → value function network improves

3. TRAIN POLICY NETWORK
   Compute probability ratios (new / old)
   Multiply by advantages
   Clip the ratios to prevent large changes
   Take the minimum (conservative update)
   Loss = average across batch
   Backpropagate → policy network improves

4. DISCARD BATCH AND REPEAT
```

Both networks learn together, iteration after iteration, until the agent masters the task.

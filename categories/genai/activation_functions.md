# Common Activation Functions

## Practical workflow

1. Start with ReLU for all hidden layers
2. Use sigmoid for binary output, softmax for multi-class output
3. Switch to Leaky ReLU if you see dying neuron problem
4. Try ELU/Swish if you need maximum performance and have time

## Red flags to switch

1. Dying neurons: >70% of layer outputs zero → try Leaky ReLU
2. Slow training: Consider ELU for faster convergence
3. Binary classification: Always sigmoid output
4. Multi-class: Always softmax output

> The key is starting simple (ReLU + appropriate output function) and only adding complexity when needed.


| Function | Formula | Range | Description | Distinctive Aspects | Current Use Cases | Neural Network Usage |
|----------|---------|-------|-------------|-------------------|-------------------|----------------------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $(0, 1)$ | S-shaped curve that maps any input to probability-like values between 0 and 1 | - Outputs are always positive<br>- Smooth gradient<br>- Prone to vanishing gradients<br>- Used in binary classification output | - Binary classification output layers<br>- LSTM/GRU gates<br>- Anywhere probability-like outputs needed<br>- Binary logistic regression | - Output layer: Binary classification<br>- Hidden layer: Rarely (vanishing gradients)<br>- Gates: LSTM/GRU internal gates |
| **Tanh** | $\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1, 1)$ | Hyperbolic tangent, similar to sigmoid but centered at zero | - Zero-centered output<br>- Steeper gradient than sigmoid<br>- Still suffers from vanishing gradients<br>- Better than sigmoid for hidden layers | - LSTM/GRU gates<br>- Output layers requiring centered values<br>- Some RNN architectures<br>- Specific regression tasks | - Hidden layers: Legacy networks<br>- Output layer: When centered output needed<br>- Gates: LSTM/GRU internal mechanisms |
| **ReLU** | $f(x) = \max(0, x)$ | $[0, \infty)$ | Returns input if positive, zero otherwise | - Computationally efficient<br>- Mitigates vanishing gradient<br>- Can cause "dying ReLU" problem<br>- Most popular for hidden layers | - Hidden layers in most deep networks<br>- CNN architectures<br>- General-purpose activation<br>- Fast training scenarios | - Hidden layers: Primary choice for most networks<br>- Output: Rarely used<br>- Best for: Dense/CNN hidden layers |
| **Softmax** | $\sigma(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)$, sums to 1 | Normalizes inputs into probability distribution | - Used for multi-class classification<br>- All outputs sum to 1<br>- Emphasizes largest input<br>- Maintains relative differences | - Multi-class classification output<br>- Attention mechanisms<br>- Probability distributions<br>- Language model outputs | - Output layer: Multi-class classification only<br>- Hidden layer: Never used<br>- Attention: Soft attention mechanisms |
| **Leaky ReLU** | $f(x) = x$ if $x > 0$, else $\alpha x$ ($\alpha$ small) | $(-\infty, \infty)$ | Like ReLU but allows small negative values | - Solves "dying ReLU" problem<br>- Small negative slope (e.g., 0.01x)<br>- Maintains ReLU benefits | - Hidden layers when dying ReLU is concern<br>- GAN training<br>- Networks with negative inputs<br>- Experimental architectures | - Hidden layers: When ReLU causes dying neurons<br>- Output: Rarely used<br>- Best for: Networks with dying neuron problem |
| **ELU** | $f(x) = x$ if $x > 0$, else $\alpha(e^x-1)$ | $(-\alpha, \infty)$ | Exponential linear unit with smooth negative values | - Smooth negative region<br>- Mean activation closer to zero<br>- Reduces vanishing gradients<br>- More computationally expensive | - Deep networks where vanishing gradients are critical<br>- High-accuracy applications<br>- Research experiments<br>- When faster convergence is needed | - Hidden layers: High-accuracy applications<br>- Output: Rarely used<br>- Best for: Deep networks when accuracy matters more than speed |
| **Swish** | $f(x) = x \times \sigma(\beta x)$ | $(-\infty, \infty)$ | Self-gated activation function discovered by Google | - Smooth, non-monotonic<br>- Often outperforms ReLU<br>- Learned parameter $\beta$ (or fixed)<br>- Biologically motivated | - Research applications<br>- High-performance models<br>- When optimal accuracy is priority<br>- Self-gated neural networks | - Hidden layers: Research/optimization experiments<br>- Output: Rarely used<br>- Best for: When maximum accuracy is priority |

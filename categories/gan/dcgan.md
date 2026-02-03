# Deep Convolutional Generative Adversarial Network (DCGAN)

A DCGAN is a GAN architecture that replaces fully connected layers with convolutional layers, enabling stable training on image data.

The generator takes a latent vector z (typically 100 dimensions sampled from a normal distribution) and upsamples it through transposed convolutions to produce an image. The discriminator does the reverse — it takes an image and downsamples through strided convolutions to output a real/fake probability.

Key architectural guidelines from the original paper: no pooling layers (use strided convolutions instead), batch normalization in both networks (except the generator output and discriminator input), ReLU activations in the generator (tanh on output), LeakyReLU in the discriminator, and no fully connected hidden layers.

The two networks train adversarially in a minimax game — the discriminator learns to distinguish real images from generated ones, while the generator learns to fool the discriminator. The discriminator acts as a learned, adaptive loss function for the generator.

## Stable DCGAN Guidelines

- Replace pooling layers with strided convolutions (discriminator) and transposed convolutions (generator)
- Use batch normalization in both networks, except the generator output layer and discriminator input layer
- Remove fully connected hidden layers
- Generator: ReLU activations, tanh on output
- Discriminator: LeakyReLU activations (slope 0.2)

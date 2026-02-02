# FID & KID Evaluation Metrics

## FID (Frechet Inception Distance)

Measures the distance between the distribution of real images and generated images. It works by passing both sets through a pretrained Inception v3 network, extracting feature vectors, fitting a multivariate Gaussian to each set, and computing the Frechet distance between them. Lower is better. It captures both image quality and diversity — a generator that produces sharp but repetitive images will still score poorly.

## KID (Kernel Inception Distance)

It is similar to FID in spirit but uses the squared Maximum Mean Discrepancy (MMD) with a polynomial kernel instead of fitting Gaussians. The main practical advantage over FID is that KID gives an unbiased estimate regardless of sample size, whereas FID is biased with small sample counts. This makes KID more reliable when you can't afford to generate tens of thousands of images for evaluation.

---

**NOTES:**

**[1]** Both metrics use Inception v3 features as the representation space, so they're measuring perceptual similarity rather than pixel-level similarity.

**[2]** The Frechet Distance between two multivariate Gaussians $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$ is defined as:

$$d^2 = \|\mu_1 - \mu_2\|^2 + \mathrm{Tr}\left(\Sigma_1 + \Sigma_2 - 2\left(\Sigma_1 \Sigma_2\right)^{1/2}\right)$$

The first term measures how far apart the means are (i.e., the average features differ). The second term (the trace component) captures differences in the covariance structure - how the spread and correlations of features differ between real and generated distributions.

It's essentially the Wasserstein-2 distance between two Gaussians, which gives it a nice interpretation as the minimum "cost" of transporting one distribution to the other.

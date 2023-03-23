# MatrixFactorization
Latent factor models via matrix factorization

LFM learns the following function: $\hat{r_{ui}} = \mu + b_i + b_u + q_{i}^{T}p_u$
  - or if "Latent" mode is not selected, it will only use bias terms: $\hat{r}_{ui} = \mu + b_i + b_u$

LFMpp (also known as SVD++) utilizes implicit information: $\hat{r_{ui}} = \mu + b_i + b_u + q_{i}^{T} (p_u *|N(u)|^{-.5} \sum_{i\in N(u)} x_i)$


For more details, see: [Y. Koren, R. Bell and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," in Computer, vol. 42, no. 8, pp. 30-37, Aug. 2009, doi: 10.1109/MC.2009.263.](https://ieeexplore.ieee.org/document/5197422)



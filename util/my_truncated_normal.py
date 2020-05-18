import torch
import numpy as np

class TruncatedNormal:
    def __init__(self, mu, sigma, a, b):
        self.normal = torch.distributions.Normal(0, 1)

        self.mu = mu
        self.sigma = sigma

        self.a = a
        self.b = b

        self.alpha = (a - mu) / sigma
        self.beta  = (b - mu) / sigma
        self.Z     = self.normal.cdf(self.beta) - self.normal.cdf(self.alpha)

        self.eps = np.finfo(np.float32).eps

    def log_prob(self, x):
        return self.normal.log_prob((x - self.mu) / self.sigma) - torch.log(self.sigma * self.Z)

    def sample(self):
        alpha = (self.a - self.mu) / self.sigma
        beta  = (self.b - self.mu) / self.sigma

        alpha_normal_cdf = self.normal.cdf(alpha)
        p = alpha_normal_cdf + (self.normal.cdf(beta) - alpha_normal_cdf) * torch.cuda.FloatTensor(self.mu.shape).uniform_()

        v = (2 * p - 1).clamp(-1 + self.eps, 1 - self.eps)
        x = self.mu + self.sigma * 1.4142135623730951 * torch.erfinv(v)
        return x.clamp(self.a, self.b)

    def entropy(self):
        #experiments required here
        return torch.log(4.132731354122493 * self.sigma * self.Z) \
             + (self.alpha * torch.exp(self.normal.log_prob(self.alpha)) - self.beta * torch.exp(self.normal.log_prob(self.beta))) / (2 * self.Z)
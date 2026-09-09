#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iDDPM Gaussian diffusion (Nichol & Dhariwal 2021) — the exact training/sampling
recipe the DiT paper (Peebles & Xie 2023) uses: learned variance (v-interp between
beta_t and beta_tilde_t), hybrid loss L_simple + lambda*L_vlb, and ancestral
(p_sample) sampling. Linear beta schedule, T=1000, epsilon-prediction.

Model contract: model(x_t, t, **kw) -> [eps(C), v(C)] concatenated on channel dim
(learn_sigma). Conditioning (source latent) is passed via channel-concat in the
caller, so this module stays generic.
"""
import math
import numpy as np
import torch


def linear_betas(T):
    return torch.linspace(1e-4, 0.02, T, dtype=torch.float64)


def _extract(arr, t, shape):
    out = arr.to(t.device)[t].float()
    return out.view(t.shape[0], *([1] * (len(shape) - 1)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1
                  + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def _approx_std_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    centered = x - means
    inv_stdv = torch.exp(-log_scales)
    plus = _approx_std_normal_cdf(inv_stdv * (centered + 1.0 / 255.0))
    minus = _approx_std_normal_cdf(inv_stdv * (centered - 1.0 / 255.0))
    log_cdf_plus = torch.log(plus.clamp(min=1e-12))
    log_one_minus_cdf_minus = torch.log((1.0 - minus).clamp(min=1e-12))
    cdf_delta = plus - minus
    log_probs = torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_minus,
                    torch.log(cdf_delta.clamp(min=1e-12))))
    return log_probs


class IDDPM:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        betas = linear_betas(T)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, 0)
        acp_prev = torch.cat([torch.ones(1, dtype=torch.float64), acp[:-1]])
        self.betas = betas
        self.alphas_cumprod = acp
        self.alphas_cumprod_prev = acp_prev
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp)
        self.sqrt_recip_acp = torch.sqrt(1.0 / acp)
        self.sqrt_recipm1_acp = torch.sqrt(1.0 / acp - 1.0)
        # posterior q(x_{t-1}|x_t,x_0)
        self.post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.post_log_var_clipped = torch.log(
            torch.cat([self.post_var[1:2], self.post_var[1:]]))
        self.post_mean_c1 = betas * torch.sqrt(acp_prev) / (1.0 - acp)
        self.post_mean_c2 = (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp)
        self.log_betas = torch.log(betas)
        self.device = device

    def q_sample(self, x0, t, noise):
        return (_extract(self.sqrt_acp, t, x0.shape) * x0
                + _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    def q_posterior(self, x0, x_t, t):
        mean = (_extract(self.post_mean_c1, t, x_t.shape) * x0
                + _extract(self.post_mean_c2, t, x_t.shape) * x_t)
        return mean, _extract(self.post_log_var_clipped, t, x_t.shape)

    def _predict_x0(self, x_t, t, eps):
        return (_extract(self.sqrt_recip_acp, t, x_t.shape) * x_t
                - _extract(self.sqrt_recipm1_acp, t, x_t.shape) * eps)

    def p_mean_variance(self, model_out, x_t, t, clip=True):
        C = x_t.shape[1]
        eps, v = model_out[:, :C], model_out[:, C:]
        # learned variance: interpolate in log-space between beta_tilde and beta
        min_lv = _extract(self.post_log_var_clipped, t, x_t.shape)
        max_lv = _extract(self.log_betas, t, x_t.shape)
        frac = (v + 1) / 2
        log_var = frac * max_lv + (1 - frac) * min_lv
        x0 = self._predict_x0(x_t, t, eps)
        if clip:
            x0 = x0.clamp(-3, 3)   # latent space; loose clip
        mean, _ = self.q_posterior(x0, x_t, t)
        return mean, log_var, x0, eps

    # ---------------- training ----------------
    def training_losses(self, model, x0, t, model_kwargs, lam_vlb=0.001):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        out = model(x_t, t, **model_kwargs)
        C = x0.shape[1]
        eps_pred, v = out[:, :C], out[:, C:]
        # L_simple
        mse = ((noise - eps_pred) ** 2).mean()
        # L_vlb (use stop-grad eps so vlb only trains the variance v)
        frozen = torch.cat([eps_pred.detach(), v], dim=1)
        mean_q, logvar_q = self.q_posterior(x0, x_t, t)
        mean_p, logvar_p, _, _ = self.p_mean_variance(frozen, x_t, t, clip=False)
        kl = normal_kl(mean_q, logvar_q, mean_p, logvar_p).flatten(1).mean(1) / math.log(2.0)
        # decoder nll at t=0
        nll = -discretized_gaussian_log_likelihood(
            x0, mean_p, 0.5 * logvar_p).flatten(1).mean(1) / math.log(2.0)
        vlb = torch.where(t == 0, nll, kl).mean()
        return mse + lam_vlb * vlb, mse.detach(), vlb.detach()

    # ---------------- sampling ----------------
    @torch.no_grad()
    def p_sample_loop(self, model, shape, model_kwargs, cfg_scale=1.0,
                      null_kwargs=None, progress=False):
        dev = self.device
        x = torch.randn(shape, device=dev)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=dev, dtype=torch.long)
            if cfg_scale != 1.0 and null_kwargs is not None:
                out_c = model(x, t, **model_kwargs)
                out_u = model(x, t, **null_kwargs)
                C = shape[1]
                eps = out_u[:, :C] + cfg_scale * (out_c[:, :C] - out_u[:, :C])
                out = torch.cat([eps, out_c[:, C:]], dim=1)
            else:
                out = model(x, t, **model_kwargs)
            mean, log_var, _, _ = self.p_mean_variance(out, x, t, clip=True)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + torch.exp(0.5 * log_var) * noise
        return x

from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import datasets.image_util as util
import torch.nn.init as init
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        self.latent_size = opt.noiseSize
        self.embed_type = opt.embed_type
        input_size = opt.resSize
        if self.embed_type == "VA":
            input_size += opt.attSize
        self.fc1 = nn.Linear(input_size, layer_sizes[-1])
        self.fc2 = nn.Linear(layer_sizes[-1], self.latent_size * 2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.linear_means = nn.Linear(self.latent_size * 2, self.latent_size)
        self.linear_log_var = nn.Linear(self.latent_size * 2, self.latent_size)
        self.apply(weights_init)

    def forward(self, x, att=None):
        batch_size = x.shape[0]
        if self.embed_type == "VA":
            x = torch.cat((x, att), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        std = torch.exp(0.5 * log_vars)
        eps = torch.randn([batch_size, self.latent_size]).to(x.device)
        eps = Variable(eps)
        z = eps * std + means  # torch.Size([64, 312])
        return z, means, log_vars

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class TimeEmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeEmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

# Decoder/Generator
class DRG_Generator(nn.Module):
    # for AWA and CUB
    def __init__(self, opt):
        super(DRG_Generator, self).__init__()
        layer_sizes = opt.decoder_layer_sizes
        latent_size = opt.noiseSize

        self.dim_t = opt.dim_t
        self.n_T = opt.n_T
        if opt.dataset=="FLO":
            input_size = latent_size + opt.attSize + opt.resSize + 64
            self.time_embed = TimeEmbedFC(self.dim_t, 64)
            self.con_emb_layers = TimeEmbedFC(2048, 64)
        else:
            input_size = latent_size + opt.attSize + opt.resSize + opt.attSize
            self.time_embed = TimeEmbedFC(self.dim_t, opt.attSize)
            self.con_emb_layers = TimeEmbedFC(2048, opt.attSize)

        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2)
        self.act = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, z, c_t, att, t):
        t_emb_ = timestep_embedding(t, self.dim_t, repeat_only=False)
        t_emb = self.time_embed(t_emb_)
        # t_rep = F.normalize(t_emb, p=2, dim=1)

        z = torch.cat((z, att, c_t, t_emb), dim=-1)

        x1 = self.lrelu(self.fc1(z))
        x = self.act(self.fc2(x1))
        return x

# Decoder/Generator
class DFG_Generator(nn.Module):
    # for AWA and CUB
    def __init__(self, opt):
        super(DFG_Generator, self).__init__()
        layer_sizes = opt.decoder_layer_sizes
        latent_size = opt.noiseSize

        self.dim_t = opt.dim_t
        self.n_T = opt.n_T
        if opt.dataset=="FLO":
            input_size = latent_size + opt.attSize + opt.resSize + 64 + 64
            self.time_embed = TimeEmbedFC(self.dim_t, 64)
            self.con_emb_layers = TimeEmbedFC(2048, 64)
        else:
            input_size = latent_size + opt.attSize + opt.resSize + opt.attSize + opt.attSize
            self.time_embed = TimeEmbedFC(self.dim_t, opt.attSize)
            self.con_emb_layers = TimeEmbedFC(2048, opt.attSize)

        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2)
        self.act = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, z, att, con, x_t, t):
        t_emb_ = timestep_embedding(t, self.dim_t, repeat_only=False)
        t_emb = self.time_embed(t_emb_)
        # t_rep = F.normalize(t_emb, p=2, dim=1)
        # z = torch.cat((z, att, x_t, t_rep), dim=-1)

        con_emb = self.con_emb_layers(con)
        # con_rep = F.normalize(con_emb, p=2, dim=1)
        z = torch.cat((z, att, con_emb, x_t, t_emb), dim=-1)

        x1 = self.lrelu(self.fc1(z))
        x = self.act(self.fc2(x1))
        return x

class DFG_Discriminator_xc(nn.Module):
    def __init__(self, opt):
        super(DFG_Discriminator_xc, self).__init__()
        self.fc1 = nn.Linear(opt.resSize +opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.con_emb_layers = TimeEmbedFC(2048, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, con):
        con_emb = self.con_emb_layers(con)
        # con_rep = F.normalize(con_emb, p=2, dim=1)

        h = torch.cat((x, con_emb), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h

    def calc_gradient_penalty(self, real_data, fake_data, input_con, lambda1):
        batch_size = real_data.shape[0]
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(real_data.device)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, Variable(input_con))
        ones = torch.ones(disc_interpolates.size())
        ones = ones.to(real_data.device)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
        return gradient_penalty

class DFG_Discriminator_x0(nn.Module):
    def __init__(self, opt):
        super(DFG_Discriminator_x0, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h

    def calc_gradient_penalty(self, real_data, fake_data, input_att, lambda1):
        batch_size = real_data.shape[0]
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(real_data.device)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, Variable(input_att))
        ones = torch.ones(disc_interpolates.size())
        ones = ones.to(real_data.device)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
        return gradient_penalty

class DRG_Discriminator_ct(nn.Module):
    def __init__(self, opt):
        super(DRG_Discriminator_ct, self).__init__()
        self.n_T = opt.n_T
        self.dim_v = opt.resSize
        self.dim_s = opt.attSize
        self.dim_noise = opt.noiseSize
        self.dim_t = opt.dim_t
        self.time_embed = TimeEmbedFC(self.dim_t, opt.attSize)

        self.fc1 = nn.Linear(self.dim_v+self.dim_v + opt.attSize + self.dim_t, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2)

        self.apply(weights_init)

    def forward(self, x_t, x_tp1, att, t):
        t_emb_ = timestep_embedding(t, self.dim_t, repeat_only=False)
        t_emb = self.time_embed(t_emb_)

        h = torch.cat((x_t, x_tp1, att, t_emb), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h

    def calc_gradient_penalty(self, real_x_t, fake_x_t, real_x_tp1, input_att, t, lambda1):
        batch_size = real_x_t.shape[0]
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_x_t.size())
        alpha = alpha.to(real_x_t.device)
        interpolates = alpha * real_x_t + ((1 - alpha) * fake_x_t)
        interpolates = interpolates.to(real_x_t.device)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, real_x_tp1, Variable(input_att), t)
        ones = torch.ones(disc_interpolates.size())
        ones = ones.to(real_x_t.device)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
        return gradient_penalty

# conditional discriminator for inductive
class DFG_Discriminator_xt(nn.Module):
    def __init__(self, opt):
        super(DFG_Discriminator_xt, self).__init__()
        self.n_T = opt.n_T
        self.dim_v = opt.resSize
        self.dim_s = opt.attSize
        self.dim_noise = opt.noiseSize
        self.dim_t = opt.dim_t
        self.time_embed = TimeEmbedFC(self.dim_t, opt.attSize)
        self.con_emb_layers = TimeEmbedFC(2048, opt.attSize)

        self.fc1 = nn.Linear(self.dim_v+self.dim_v + opt.attSize + opt.attSize + self.dim_t, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2)

        self.apply(weights_init)

    def forward(self, x_t, x_tp1, att, con, t):
        t_emb_ = timestep_embedding(t, self.dim_t, repeat_only=False)
        t_emb = self.time_embed(t_emb_)
        # t_rep = F.normalize(t_emb, p=2, dim=1)

        con_emb = self.con_emb_layers(con)
        # con_rep = F.normalize(con_emb, p=2, dim=1)

        h = torch.cat((x_t, x_tp1, att, con_emb, t_emb), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h

    def calc_gradient_penalty(self, real_x_t, fake_x_t, real_x_tp1, input_att, input_con, t, lambda1):
        batch_size = real_x_t.shape[0]
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(real_x_t.size())
        alpha = alpha.to(real_x_t.device)
        interpolates = alpha * real_x_t + ((1 - alpha) * fake_x_t)
        interpolates = interpolates.to(real_x_t.device)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates, real_x_tp1, Variable(input_att), Variable(input_con), t)
        ones = torch.ones(disc_interpolates.size())
        ones = ones.to(real_x_t.device)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
        return gradient_penalty

class V2S_mapping(nn.Module):
    def __init__(self, opt, attSize):
        super(V2S_mapping, self).__init__()
        self.embedSz = 0
        self.class_embedding_norm = opt.class_embedding_norm
        self.fc1 = nn.Linear(opt.resSize + self.embedSz, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.hidden = None
        self.sigmoid = None
        self.apply(weights_init)

    def forward(self, feat, att=None):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc3(self.hidden)
        if self.class_embedding_norm:
            h = F.normalize(h, p=2, dim=1)
        self.out = h
        return h

    def getLayersOutDet(self):
        return self.hidden.detach()

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def ddpmgan_sigma_schedule(beta1, beta2, n_timestep, device, use_linear_betas):
    if use_linear_betas:
        first = torch.tensor(1e-8).to(device)
        betas = (beta2 - beta1) * torch.arange(0, n_timestep + 1, dtype=torch.float32) / n_timestep + beta1
        betas = betas.to(device)
        betas = torch.cat((first[None], betas))[:-1]
        sigmas = betas ** 0.5
        sqrt_alphas = torch.sqrt(1 - betas)
    else:
        eps_small = 1e-3
        t = np.arange(0, n_timestep + 1, dtype=np.float32)
        t = t / n_timestep
        t = torch.from_numpy(t).to(device) * (1. - eps_small) + eps_small

        var = var_func_vp(t, beta1, beta2).to(device)
        alphas_bar = 1.0 - var
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]

        first = torch.tensor(1e-8).to(device)
        betas = torch.cat((first[None], betas))
        sigmas = betas ** 0.5
        sqrt_alphas = torch.sqrt(1 - betas)

    return sigmas, sqrt_alphas, betas, var

class ddpmgan_prior_coefficients():
    def __init__(self, beta1, beta2, n_timestep, device, use_linear_betas):
        if use_linear_betas:
            assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1) if linear betas"
        self.sigmas, self.sqrt_alphas, self.betas, self.vars = ddpmgan_sigma_schedule(beta1, beta2, n_timestep, device, use_linear_betas)
        self.cross_entropy_const = 0.5 * (1.0 + torch.log(2.0 * np.pi * self.vars[0])).to(device)
        self.sqrt_alphas_bar = torch.cumprod(self.sqrt_alphas, dim=0).to(device)
        self.sigmas_bar = torch.sqrt(1 - self.sqrt_alphas_bar ** 2).to(device)
        self.sqrt_alphas_prev = self.sqrt_alphas.clone().to(device)
        self.sqrt_alphas_prev[-1] = 1
        self.g2 = beta1 + (beta2 - beta1) * torch.arange(0, n_timestep + 1, dtype=torch.float32).to(device)

        self.oneover_sqrta = 1 / self.sqrt_alphas
        log_alpha_t = torch.log(self.sqrt_alphas**2)
        alphabar_t =  torch.cumsum(log_alpha_t, dim=0).exp()
        sqrtmab = torch.sqrt(1 - alphabar_t)
        self.mab_over_sqrtmab = (1 - self.sqrt_alphas**2) / sqrtmab
        self.sqrt_betas = torch.sqrt(self.betas)

class ddpmgan_posterior_coefficients():
    def __init__(self, beta1, beta2, n_timestep, device, use_linear_betas):
        _, _, self.betas, _ = ddpmgan_sigma_schedule(beta1, beta2, n_timestep, device, use_linear_betas)
        self.betas = self.betas[1:].to(device)
        self.alphas = 1 - self.betas.to(device)
        self.alphas_bar = torch.cumprod(self.alphas, 0).to(device)
        self.alphas_bar_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_bar[:-1]), 0
        ).to(device)
        self.posterior_variance = (self.betas * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)).to(device)

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar).to(device)
        self.sqrt_recip_alphas_bar = torch.rsqrt(self.alphas_bar).to(device)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1 / self.alphas_bar - 1).to(device)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_bar_prev) / (1 - self.alphas_bar)).to(device)
        self.posterior_mean_coef2 = ((1 - self.alphas_bar_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_bar)).to(
            device)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20)).to(device)

        self.mean_coef_xt = torch.sqrt(1./self.alphas).to(device)
        self.mean_coef_eps = (self.mean_coef_xt*(1. - self.alphas) / torch.sqrt(1. - self.alphas_bar)).to(device)

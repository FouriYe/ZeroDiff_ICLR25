# author: ZihanYe
# ZeroDiff (ICLR25)
from __future__ import print_function
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# import functions
import datasets.image_util as util
import classifiers.classifier_images as classifier
from config_zerodiff import opt
import zerodiff_tools
import torch.nn.functional as F
from sklearn import preprocessing
import numpy as np
import torch.nn as nn
import os

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "w")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()

folder_name = "./log"
os.makedirs(folder_name, exist_ok=True)
folder_name = "./out"
os.makedirs(folder_name, exist_ok=True)

logger_name = "./log/%s/train_zerodiff_DFG_%dpercent_att:%s_b:%d_lr:%s_n_T:%d_betas:%s,%s_gamma:ADV:%.1f_VAE:%.1f_x0:%.1f_xt:%.1f_dist:%.1f_f:%.1f_num:%s" % (
    opt.dataset, opt.split_percent, opt.class_embedding, opt.batch_size, str(opt.lr), opt.n_T, str(opt.ddpmbeta1),
    str(opt.ddpmbeta2), opt.gamma_ADV, opt.gamma_VAE, opt.gamma_x0, opt.gamma_xt, opt.gamma_dist, opt.factor_dist, opt.syn_num)
logger = Logger(logger_name)
model_save_name = "./out/%s/train_zerodiff_DFG_%dpercent_att:%s_b:%d_lr:%s_n_T:%d_betas:%s,%s_gamma:ADV:%.1f_VAE:%.1f_x0:%.1f_xt:%.1f_dist:%.1f_f:%.1f_num:%d" % (
    opt.dataset, opt.split_percent, opt.class_embedding, opt.batch_size, str(opt.lr), opt.n_T, str(opt.ddpmbeta1),
    str(opt.ddpmbeta2), opt.gamma_ADV, opt.gamma_VAE, opt.gamma_x0, opt.gamma_xt, opt.gamma_dist, opt.factor_dist, opt.syn_num)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_con = torch.FloatTensor(opt.batch_size, 2048)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)  # attSize class-embedding size
input_label = torch.LongTensor(opt.batch_size)  # attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.noiseSize)
input_test_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_test_con = torch.FloatTensor(opt.batch_size, opt.resSize)
input_test_att = torch.FloatTensor(opt.batch_size, opt.attSize)
##########
# Cuda
if opt.cuda:
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_label = input_label.cuda()
    input_con = input_con.cuda()
    input_test_res, input_test_con, input_test_att = input_test_res.cuda(), input_test_con.cuda(), input_test_att.cuda()


def loss_fn(recon_x, x, mean, log_var):
    Recon = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), size_average=False)
    Recon = Recon.sum() / x.size(0)
    # Recon = torch.nn.functional.mse_loss(recon_x, x.detach(), size_average=False)
    # Recon = Recon.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (Recon + KLD)


def sample(batch_size):
    batch_feature, batch_con, batch_att, batch_label = data.next_seen_batch(batch_size)
    input_res.copy_(batch_feature)
    input_con.copy_(batch_con)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)
    return input_res, input_con, input_att, input_label

def sample_con(batch_size):
    idx = torch.randperm(data.ntrain)[0:batch_size]
    batch_feature = data.train_feature[idx]
    batch_label = data.train_label[idx]
    batch_att = data.attribute[batch_label]

    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)

    return input_res, input_att, input_label

def sampleTestSeen():
    batch_feature, batch_con, batch_att, _ = data.next_test_seen_batch(opt.batch_size)
    input_test_res.copy_(batch_feature)
    input_test_con.copy_(batch_con)
    input_test_att.copy_(batch_att)

    return Variable(input_test_res), Variable(input_test_con), Variable(input_test_att)

def WeightedL14att(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)

def generate_syn_feature(zerodiff, classes, attribute, num, progressive=False):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_con = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    if opt.cuda:
        syn_att = syn_att.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        att = Variable(syn_att, volatile=True)

        fake, fake_con = zerodiff.sample_from_model(att, progressive=progressive)

        output = fake
        output_con = fake_con
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_con.narrow(0, i * num, num).copy_(output_con.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_con, syn_label


def save_zerodiff(zerodiff, save_name, post):
    torch.save({'state_dict_G': zerodiff.netG.state_dict(),
                'state_dict_Dec': zerodiff.netDec.state_dict(),
                'state_dict_D_x0': zerodiff.netD_x0.state_dict(),
                'state_dict_D_xt': zerodiff.netD_xt.state_dict(),
                'state_dict_D_xc': zerodiff.netD_xc.state_dict(),
                }, save_name + post + '.tar')


class ZERODIFF(torch.nn.Module):
    def __init__(self, data, n_T, betas, seenclasses, unseenclasses, attribute,  netR_model_path, device='cuda'):
        super(ZERODIFF, self).__init__()
        self.n_T = n_T
        self.dim_v = opt.resSize
        self.dim_s = opt.attSize
        self.dim_noise = opt.noiseSize
        self.device = device
        self.seenclasses = seenclasses
        self.unseenclasses = unseenclasses
        self.attribute = attribute
        self.attribute_seen = attribute[data.seenclasses]

        self.prior_coefficients = zerodiff_tools.ddpmgan_prior_coefficients(betas[0], betas[1], n_T, device, False)
        self.posterior_coefficients = zerodiff_tools.ddpmgan_posterior_coefficients(betas[0], betas[1], n_T, device, False)

        self.netE = zerodiff_tools.Encoder(opt).to(self.device)
        self.netG = zerodiff_tools.DFG_Generator(opt).to(self.device)
        self.netD_x0 = zerodiff_tools.DFG_Discriminator_x0(opt).to(self.device)
        self.netD_xt = zerodiff_tools.DFG_Discriminator_xt(opt).to(self.device)
        self.netD_xc = zerodiff_tools.DFG_Discriminator_xc(opt).to(self.device)
        self.netDec = zerodiff_tools.V2S_mapping(opt, opt.attSize).to(self.device)

        self.optimizerE = optim.Adam(self.netE.parameters(), lr=opt.lr)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerDec = optim.Adam(self.netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
        self.optimizerD_x0 = optim.Adam(self.netD_x0.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD_xt = optim.Adam(self.netD_xt.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD_xc = optim.Adam(self.netD_xc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.gamma_ADV = opt.gamma_ADV
        self.gamma_VAE = opt.gamma_VAE
        self.gamma_x0 = opt.gamma_x0
        self.gamma_xt = opt.gamma_xt
        self.gamma_recons = opt.gamma_recons
        self.lambda1 = opt.lambda1
        self.gamma_dist = opt.gamma_dist
        self.factor_dist = opt.factor_dist

        self.loss_mse = torch.nn.MSELoss(reduce=False)

        self.batch_size = opt.batch_size
        self.data = data

        self.netR = zerodiff_tools.DRG_Generator(opt).to(self.device)
        netR_state_dict = torch.load(netR_model_path)
        self.netR.load_state_dict(netR_state_dict['state_dict_G_con'])
        self.netR.eval()

        self.interval_recorder_sum = {}
        self.init_recorder()

    def init_recorder(self):
        self.interval_recorder_sum['criticD_train_real_x0'] = 0.0
        self.interval_recorder_sum['criticD_train_real_xt'] = 0.0
        self.interval_recorder_sum['criticD_train_real_xc'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_x0'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_xt'] = 0.0
        self.interval_recorder_sum['criticD_train_fake_xc'] = 0.0
        self.interval_recorder_sum['criticD_test_real_x0'] = 0.0
        self.interval_recorder_sum['criticD_test_real_xt'] = 0.0
        self.interval_recorder_sum['criticD_test_real_xc'] = 0.0

    def forward(self):
        gp_sum = 0  # lAMBDA VARIABLE
        for iter_d in range(opt.critic_iter):
            x_0_real, con_0_real, att_0_real, label = sample(self.batch_size)
            D_cost, Wasserstein_D, gp_sum, distill_loss = self.update_D(x_0_real, con_0_real, att_0_real, gp_sum, label)

        gp_sum /= (self.gamma_ADV * self.lambda1 * opt.critic_iter)
        if (gp_sum > 1.05).sum() > 0:
            self.lambda1 *= 1.1
        elif (gp_sum < 1.001).sum() > 0:
            self.lambda1 /= 1.1
        G_cost, vae_loss_seen = self.update_G(x_0_real, con_0_real, att_0_real, label)
        return D_cost, Wasserstein_D, distill_loss, G_cost, vae_loss_seen

    def update_D(self, x_0_real, con_0_real, att_0_real, gp_sum, label):
        for p in self.netE.parameters():
            p.requires_grad = False
        for p in self.netG.parameters():
            p.requires_grad = False
        for p in self.netD_x0.parameters():
            p.requires_grad = True
        for p in self.netD_xt.parameters():
            p.requires_grad = True
        for p in self.netD_xc.parameters():
            p.requires_grad = True
        for p in self.netDec.parameters():
            p.requires_grad = True

        z, means, log_var = self.netE(x_0_real, att_0_real)

        _ts_feat = torch.randint(0, self.n_T, (self.batch_size,), dtype=torch.int64).to(self.device)
        x_t_real, x_tp1_real, ratio_x0 = self.q_sample_pairs(x_0_real, _ts_feat)

        x_0_fake = self.netG(z, att_0_real, con_0_real, x_tp1_real.detach(), _ts_feat)
        x_t_fake = self.sample_posterior(x_0_fake, x_tp1_real, _ts_feat)

        self.netD_x0.zero_grad()
        self.netD_xt.zero_grad()
        self.netD_xc.zero_grad()
        self.netDec.zero_grad()

        att_0_recons = self.netDec(x_0_real)
        R_cost = self.gamma_recons * WeightedL14att(att_0_recons, att_0_real)
        R_cost.backward()

        criticD_real_x0 = -self.netD_x0(x_0_real, att_0_real).mean() if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)
        criticG_real_xt = -self.netD_xt(x_t_real, x_tp1_real, att_0_real, con_0_real, _ts_feat).mean() if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)
        criticD_real_xc = -self.netD_xc(x_0_real, con_0_real).mean()
        criticD_real = self.gamma_x0 * criticD_real_x0 + self.gamma_xt * criticG_real_xt + criticD_real_xc
        criticD_real = self.gamma_ADV * criticD_real
        criticD_real.backward()

        criticD_fake_x0 = self.netD_x0(x_0_fake.detach(), att_0_real).mean() if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)
        criticG_fake_xt = self.netD_xt(x_t_fake.detach(), x_tp1_real, att_0_real, con_0_real, _ts_feat).mean() if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticD_fake_xc = self.netD_xc(x_0_fake.detach(), con_0_real).mean()
        criticD_fake = self.gamma_x0 * criticD_fake_x0 + self.gamma_xt * criticG_fake_xt + criticD_fake_xc
        criticD_fake = self.gamma_ADV * criticD_fake
        criticD_fake.backward()

        # gradient penalty
        gp_x0 = self.netD_x0.calc_gradient_penalty(x_0_real, x_0_fake.data, att_0_real, self.lambda1)
        gp_xt = self.netD_xt.calc_gradient_penalty(x_t_real, x_t_fake.data, x_tp1_real, att_0_real, con_0_real, _ts_feat, self.lambda1)
        gp_xc = self.netD_xc.calc_gradient_penalty(x_0_real, x_0_fake.data, con_0_real, self.lambda1)
        gp = self.gamma_ADV * (self.gamma_x0 * gp_x0 + self.gamma_xt * gp_xt + gp_xc)
        gp.backward()
        gp_sum += gp.data
        Wasserstein_D = criticD_real - criticD_fake

        # distill
        factor = ratio_x0**self.factor_dist
        # print(factor)
        criticD_real_x0 = self.netD_x0(x_0_real, att_0_real) if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)
        criticD_fake_x0 = self.netD_x0(x_0_fake.detach(), att_0_real) if self.gamma_x0 > 0 else torch.tensor(0.0).to(self.device)

        criticD_real_xt = self.netD_xt(x_t_real, x_tp1_real, att_0_real, con_0_real, _ts_feat) if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticD_fake_xt = self.netD_xt(x_t_fake.detach(), x_tp1_real, att_0_real, con_0_real, _ts_feat) if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)

        criticD_real_xc = self.netD_xc(x_0_real, con_0_real)
        criticD_fake_xc = self.netD_xc(x_0_fake.detach(), con_0_real)

        Wasserstein_D_x0 = criticD_real_x0 - criticD_fake_x0
        Wasserstein_D_xt = criticD_real_xt - criticD_fake_xt
        Wasserstein_D_xc = criticD_real_xc - criticD_fake_xc

        distill_loss = (factor * self.loss_mse(Wasserstein_D_x0, Wasserstein_D_xt.detach())).mean() + (factor * self.loss_mse(Wasserstein_D_xt, Wasserstein_D_x0.detach())).mean() + \
                       (factor * self.loss_mse(Wasserstein_D_xc, Wasserstein_D_xt.detach())).mean() + (factor * self.loss_mse(Wasserstein_D_xt, Wasserstein_D_xc.detach())).mean() + \
                       self.loss_mse(Wasserstein_D_x0, Wasserstein_D_xc.detach()).mean() + self.loss_mse(Wasserstein_D_xc, Wasserstein_D_x0.detach()).mean()

        distill_loss = self.gamma_dist * distill_loss
        distill_loss.backward()

        D_cost = criticD_fake - criticD_real + gp  # add Y here and #add vae reconstruction loss
        self.optimizerDec.step()
        self.optimizerD_x0.step()
        self.optimizerD_xt.step()
        self.optimizerD_xc.step()

        with torch.no_grad():
            test_seen_x_0_real, test_seen_con_0_real, test_seen_att_0_real = sampleTestSeen()
            test_seen_x_t_real, test_seen_x_tp1_real, ratio_x0 = self.q_sample_pairs(test_seen_x_0_real, _ts_feat)
            criticD_test_real_x0 = self.netD_x0(test_seen_x_0_real, test_seen_att_0_real)
            criticD_test_real_xt = self.netD_xt(test_seen_x_t_real, test_seen_x_tp1_real, test_seen_att_0_real, test_seen_con_0_real, _ts_feat)
            criticD_test_real_xc = self.netD_xc(test_seen_x_0_real, test_seen_con_0_real)

        self.interval_recorder_sum['criticD_train_real_x0'] += criticD_real_x0.mean()
        self.interval_recorder_sum['criticD_train_real_xt'] += criticD_real_xt.mean()
        self.interval_recorder_sum['criticD_train_real_xc'] += criticD_real_xc.mean()
        self.interval_recorder_sum['criticD_train_fake_x0'] += criticD_fake_x0.mean()
        self.interval_recorder_sum['criticD_train_fake_xt'] += criticD_fake_xt.mean()
        self.interval_recorder_sum['criticD_train_fake_xc'] += criticD_fake_xc.mean()
        self.interval_recorder_sum['criticD_test_real_x0'] += criticD_test_real_x0.mean()
        self.interval_recorder_sum['criticD_test_real_xt'] += criticD_test_real_xt.mean()
        self.interval_recorder_sum['criticD_test_real_xc'] += criticD_test_real_xc.mean()

        return D_cost, Wasserstein_D, gp_sum, distill_loss

    def update_G(self, x_0_real, con_0_real, att_0_real, label):
        for p in self.netE.parameters():
            p.requires_grad = True
        for p in self.netG.parameters():
            p.requires_grad = True
        for p in self.netD_x0.parameters():  # freeze discrimator
            p.requires_grad = False
        for p in self.netD_xt.parameters():
            p.requires_grad = False
        for p in self.netD_xc.parameters():
            p.requires_grad = False
        if opt.gamma_recons > 0 and opt.freeze_dec:
            for p in self.netDec.parameters():  # freeze decoder
                p.requires_grad = False

        self.netE.zero_grad()
        self.netG.zero_grad()

        z, means, log_var = self.netE(x_0_real, att_0_real)

        _ts_feat = torch.randint(0, self.n_T, (self.batch_size,), dtype=torch.int64).to(self.device)
        x_t_real, x_tp1_real, _ = self.q_sample_pairs(x_0_real, _ts_feat)
        x_0_fake = self.netG(z, att_0_real, con_0_real, x_tp1_real.detach(), _ts_feat)
        x_t_fake = self.sample_posterior(x_0_fake, x_tp1_real, _ts_feat)

        errG = 0.0
        vae_loss_seen = loss_fn(x_0_fake, x_0_real, means, log_var) if self.gamma_VAE > 0 else torch.tensor(0.0).to(self.device)
        errG += self.gamma_VAE * vae_loss_seen

        criticG_fake_x0 = -self.netD_x0(x_0_fake, att_0_real).mean() if self.gamma_x0 > 0 else torch.tensor(0.0).to( self.device)
        criticG_fake_xt = -self.netD_xt(x_t_fake, x_tp1_real, att_0_real, con_0_real, _ts_feat).mean() if self.gamma_xt > 0 else torch.tensor(0.0).to(self.device)
        criticG_fake_xc = -self.netD_xc(x_0_fake, con_0_real).mean()
        criticG_fake = self.gamma_x0 * criticG_fake_x0 + self.gamma_xt * criticG_fake_xt + criticG_fake_xc
        G_cost = criticG_fake

        errG += self.gamma_ADV * G_cost

        self.netDec.zero_grad()
        att_0_recons = self.netDec(x_0_fake)
        R_cost = WeightedL14att(att_0_recons, att_0_real)
        errG += self.gamma_recons * R_cost

        errG.backward()
        # write a condition here
        self.optimizerE.step()
        self.optimizerG.step()
        if self.gamma_recons > 0 and not opt.freeze_dec:  # not train decoder at feedback time
            self.optimizerDec.step()
        return G_cost, vae_loss_seen

    def sample_from_model(self, att, progressive=False):
        n_sample = att.shape[0]
        with torch.no_grad():
            x_t = torch.randn(n_sample, self.dim_v).to(self.device)
            z = torch.randn(n_sample, self.dim_noise).to(self.device)

            z_con = torch.randn(n_sample, self.dim_noise).to(self.device)
            _ts_con = (self.n_T - 1) + torch.zeros((n_sample,), dtype=torch.int64).to(self.device) # torch.randint(0, self.n_T + 1, (n_sample,), dtype=torch.int64).to(self.device)
            con_t = torch.randn(n_sample, 2048).to(self.device)
            con_0_fake = self.netR(z_con, att, con_t, _ts_con)
            con_0_fake = con_0_fake.detach()

            if progressive:
                for i in reversed(range(self.n_T)):
                    _ts = torch.full((n_sample,), i, dtype=torch.int64).to(x_t.device)
                    x_0_pred = self.netG(z, att, con_0_fake, x_t, _ts)
                    x_t_sub_one = self.sample_posterior(x_0_pred, x_t, _ts.long())
                    x_t = x_t_sub_one.detach()
                x_0_fake = x_0_pred.detach()
            else:
                _ts = (self.n_T - 1) + torch.zeros((n_sample,), dtype=torch.int64).to(self.device)
                x_0_fake = self.netG(z, att, con_0_fake, x_t, _ts)
                x_0_fake = x_0_fake.detach()

        return x_0_fake, con_0_fake

    def q_sample_pairs(self, x_0, t):
        """
        Generate a pair of disturbed images for training, use prior_coefficients
        :param x_0: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        t = t.long()
        noise = torch.randn_like(x_0)
        x_t, ratio_x0 = self.q_sample(x_0, t)

        ratio_xt2xtp1 = zerodiff_tools.extract(self.prior_coefficients.sqrt_alphas, t + 1, x_0.shape)
        ratio_noise = zerodiff_tools.extract(self.prior_coefficients.sigmas, t + 1, x_0.shape)

        x_t_plus_one = ratio_xt2xtp1 * x_t + ratio_noise * noise

        return x_t, x_t_plus_one, ratio_x0

    def q_sample(self, x_0, t):
        """
        use prior_coefficients
        q(x_{t}|x_0,t)
        Diffuse the data (t == 0 means diffused for t step)
        """
        t = t.long()
        noise = torch.randn_like(x_0)

        ratio_x0 = zerodiff_tools.extract(self.prior_coefficients.sqrt_alphas_bar, t, x_0.shape)
        ratio_noise = zerodiff_tools.extract(self.prior_coefficients.sigmas_bar, t, x_0.shape)

        x_t = ratio_x0 * x_0 + ratio_noise * noise

        return x_t, ratio_x0

    def sample_posterior(self, x_0, x_t, t):
        """
        use posterior_coefficients
        q(x_{t-1}|x_0,x_t,t)
        """
        t = t.long()
        posterior_mean_coef1 = zerodiff_tools.extract(self.posterior_coefficients.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2 = zerodiff_tools.extract(self.posterior_coefficients.posterior_mean_coef2, t, x_t.shape)

        mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        log_var_clipped = zerodiff_tools.extract(self.posterior_coefficients.posterior_log_variance_clipped, t, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        sample_x_pos = mean + nonzero_mask[:, None] * torch.exp(0.5 * log_var_clipped) * noise

        return sample_x_pos


zerodiff = ZERODIFF(data, n_T=opt.n_T, betas=(opt.ddpmbeta1, opt.ddpmbeta2), seenclasses=data.seenclasses,
                  unseenclasses=data.unseenclasses, attribute=data.attribute,
                  netR_model_path=opt.netR_model_path, device='cuda')
zerodiff.train()

best_gzsl_acc_V = 0
best_acc_seen_V = 0
best_acc_unseen_V = 0
best_zsl_acc_V = 0

best_gzsl_acc_VS = 0
best_acc_seen_VS = 0
best_acc_unseen_VS = 0
best_zsl_acc_VS = 0

best_gzsl_acc_C = 0
best_acc_seen_C = 0
best_acc_unseen_C = 0
best_zsl_acc_C = 0

best_gzsl_acc_VC = 0
best_acc_seen_VC = 0
best_acc_unseen_VC = 0
best_zsl_acc_VC = 0

best_gzsl_acc_VCS = 0
best_acc_seen_VCS = 0
best_acc_unseen_VCS = 0
best_zsl_acc_VCS = 0

best_seen_acc_V = 0

best_acc_seen_list_V, best_acc_unseen_list_V, best_acc_zsl_list_V = [], [], []
best_acc_seen_list_C, best_acc_unseen_list_C, best_acc_zsl_list_C = [], [], []
best_acc_seen_list_VC, best_acc_unseen_list_VC, best_acc_zsl_list_VC = [], [], []
best_acc_seen_list_VS, best_acc_unseen_list_VS, best_acc_zsl_list_VS = [], [], []
best_acc_seen_list_VCS, best_acc_unseen_list_VCS, best_acc_zsl_list_VCS = [], [], []


n_iter=data.ntrain//opt.batch_size
for epoch in range(0, opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        D_cost, Wasserstein_D, distill_loss, G_cost, vae_loss_seen = zerodiff()

    log_record = '[%d/%d] Loss_D: %.4f, Wasserstein_dist:%.4f, distill_loss:%.4f' % (
        epoch, opt.nepoch, D_cost.item(), Wasserstein_D.item(), distill_loss.item())
    print(log_record)
    logger.write(log_record + '\n')

    log_record = '[%d/%d] Loss_G: %.4f, vae_loss_seen:%.4f' % (
        epoch, opt.nepoch, G_cost.item(), vae_loss_seen.item())
    print(log_record)
    logger.write(log_record + '\n')

    criticD_train_real_x0 = ZERODIFF.interval_recorder_sum['criticD_train_real_x0'].item() / n_iter
    criticD_train_real_xt = zerodiff.interval_recorder_sum['criticD_train_real_xt'].item() / n_iter
    criticD_train_real_xc = zerodiff.interval_recorder_sum['criticD_train_real_xc'].item() / n_iter

    criticD_test_real_x0 = zerodiff.interval_recorder_sum['criticD_test_real_x0'].item() / n_iter
    criticD_test_real_xt = zerodiff.interval_recorder_sum['criticD_test_real_xt'].item() / n_iter
    criticD_test_real_xc = zerodiff.interval_recorder_sum['criticD_test_real_xc'].item() / n_iter

    criticD_train_fake_x0 = zerodiff.interval_recorder_sum['criticD_train_fake_x0'].item() / n_iter
    criticD_train_fake_xt = zerodiff.interval_recorder_sum['criticD_train_fake_xt'].item() / n_iter
    criticD_train_fake_xc = zerodiff.interval_recorder_sum['criticD_train_fake_xc'].item() / n_iter
    zerodiff.init_recorder()

    log_record = '[%d/%d] D_train_real_x0: %.6f, D_train_real_xt: %.6f, D_train_real_xc: %.6f' % (epoch, opt.nepoch, criticD_train_real_x0, criticD_train_real_xt, criticD_train_real_xc)
    print(log_record)
    logger.write(log_record + '\n')

    log_record = '[%d/%d] D_test_real_x0: %.6f, D_test_real_xt: %.6f, D_test_real_xc: %.6f' % (epoch, opt.nepoch, criticD_test_real_x0, criticD_test_real_xt, criticD_test_real_xc)
    print(log_record)
    logger.write(log_record + '\n')

    log_record = '[%d/%d] D_train_fake_x0: %.6f, D_train_fake_xt: %.6f, D_train_fake_xc: %.6f' % (epoch, opt.nepoch, criticD_train_fake_x0, criticD_train_fake_xt, criticD_train_fake_xc)
    print(log_record)
    logger.write(log_record + '\n')

    if epoch % opt.eval_interval == 0 or epoch == (opt.nepoch - 1):
        zerodiff.eval()
        syn_feature, syn_con, syn_label = generate_syn_feature(zerodiff, data.unseenclasses, data.attribute, opt.syn_num)
        syn_feature_pro, syn_con_pro, syn_label_pro = generate_syn_feature(zerodiff, data.unseenclasses, data.attribute, opt.syn_num, progressive=True)
        syn_feature_seen, syn_con_seen, syn_label_seen = generate_syn_feature(zerodiff, data.seenclasses, data.attribute, opt.syn_num)

        # Train Seen classifier in V
        seen_cls_V = classifier.CLASSIFIER(syn_feature_seen, util.map_label(syn_label_seen, data.seenclasses), \
                                           data, data.seenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="seen")
        acc = seen_cls_V.acc
        if best_seen_acc_V < acc:
            best_seen_acc_V = acc
        log_record = 'Seen (V): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Generalized zero-shot learning
        if opt.gzsl:
            # Concatenate real seen features with synthesized unseen features
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_C = torch.cat((data.train_paco, syn_con), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            train_X_pro = torch.cat((data.train_feature, syn_feature_pro), 0)
            train_C_pro = torch.cat((data.train_paco, syn_con_pro), 0)
            train_Y_pro = torch.cat((data.train_label, syn_label_pro), 0)
            nclass = opt.nclass_all

            # Train GZSL classifier in V
            gzsl_cls_V = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5,  25, opt.syn_num, cls_mode="GZSL")
            if best_gzsl_acc_V < gzsl_cls_V.H:
                best_acc_seen_V, best_acc_unseen_V, best_gzsl_acc_V = gzsl_cls_V.acc_seen, gzsl_cls_V.acc_unseen, gzsl_cls_V.H
                best_acc_unseen_list_V, best_acc_seen_list_V = gzsl_cls_V.best_acc_U_list, gzsl_cls_V.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_V")
            log_record = 'GZSL (V): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_V.acc_unseen, gzsl_cls_V.acc_seen, gzsl_cls_V.H)
            print(log_record)
            logger.write(log_record + '\n')

            gzsl_cls_V = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                               25, opt.syn_num, cls_mode="GZSL")
            if best_gzsl_acc_V < gzsl_cls_V.H:
                best_acc_seen_V, best_acc_unseen_V, best_gzsl_acc_V = gzsl_cls_V.acc_seen, gzsl_cls_V.acc_unseen, gzsl_cls_V.H
                best_acc_unseen_list_V, best_acc_seen_list_V = gzsl_cls_V.best_acc_U_list, gzsl_cls_V.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_V")
            log_record = 'GZSL pro (V): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_V.acc_unseen, gzsl_cls_V.acc_seen, gzsl_cls_V.H)
            print(log_record)
            logger.write(log_record + '\n')

            # Train GZSL classifier in VS
            gzsl_cls_VS = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                                25, opt.syn_num, cls_mode="GZSL", netDec=zerodiff.netDec,
                                                dec_size=opt.attSize, dec_hidden_size=4096, useS=True)
            if best_gzsl_acc_VS < gzsl_cls_VS.H:
                best_acc_seen_VS, best_acc_unseen_VS, best_gzsl_acc_VS = gzsl_cls_VS.acc_seen, gzsl_cls_VS.acc_unseen, gzsl_cls_VS.H
                best_acc_unseen_list_VS, best_acc_seen_list_VS = gzsl_cls_VS.best_acc_U_list, gzsl_cls_VS.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VS")
            log_record = 'GZSL (VS): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_VS.acc_unseen, gzsl_cls_VS.acc_seen, gzsl_cls_VS.H)
            print(log_record)
            logger.write(log_record + '\n')

            # Train GZSL classifier in VS
            gzsl_cls_VS = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr,
                                                0.5, 25, opt.syn_num, cls_mode="GZSL", netDec=zerodiff.netDec,
                                                dec_size=opt.attSize,
                                                dec_hidden_size=4096, useS=True)
            if best_gzsl_acc_VS < gzsl_cls_VS.H:
                best_acc_seen_VS, best_acc_unseen_VS, best_gzsl_acc_VS = gzsl_cls_VS.acc_seen, gzsl_cls_VS.acc_unseen, gzsl_cls_VS.H
                best_acc_unseen_list_VS, best_acc_seen_list_VS = gzsl_cls_VS.best_acc_U_list, gzsl_cls_VS.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VS")
            log_record = 'GZSL pro (VS): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_VS.acc_unseen, gzsl_cls_VS.acc_seen, gzsl_cls_VS.H)
            print(log_record)
            logger.write(log_record + '\n')

            # Train GZSL classifier in C
            gzsl_cls_C = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                               25, opt.syn_num, cls_mode="GZSL", con_size=2048, _train_C = train_C, useV=False, useC=True)
            if best_gzsl_acc_C < gzsl_cls_C.H:
                best_acc_seen_C, best_acc_unseen_C, best_gzsl_acc_C = gzsl_cls_C.acc_seen, gzsl_cls_C.acc_unseen, gzsl_cls_C.H
                best_acc_unseen_list_C, best_acc_seen_list_C = gzsl_cls_C.best_acc_U_list, gzsl_cls_C.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_C")
            log_record = 'GZSL (C): U: %.4f, S: %.4f, H: %.4f' % (
                gzsl_cls_C.acc_unseen, gzsl_cls_C.acc_seen, gzsl_cls_C.H)
            print(log_record)
            logger.write(log_record + '\n')

            # Train GZSL classifier in C
            gzsl_cls_C = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                               25, opt.syn_num, cls_mode="GZSL", con_size=2048, _train_C = train_C_pro, useV=False, useC=True)
            if best_gzsl_acc_C < gzsl_cls_C.H:
                best_acc_seen_C, best_acc_unseen_C, best_gzsl_acc_C = gzsl_cls_C.acc_seen, gzsl_cls_C.acc_unseen, gzsl_cls_C.H
                best_acc_unseen_list_C, best_acc_seen_list_C = gzsl_cls_C.best_acc_U_list, gzsl_cls_C.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_C")
            log_record = 'GZSL pro (C): U: %.4f, S: %.4f, H: %.4f' % (
                gzsl_cls_C.acc_unseen, gzsl_cls_C.acc_seen, gzsl_cls_C.H)
            print(log_record)
            logger.write(log_record + '\n')

            # Train GZSL classifier in VC
            gzsl_cls_VC = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                                25, opt.syn_num, cls_mode="GZSL", con_size=2048, _train_C = train_C, useC=True)
            if best_gzsl_acc_VC < gzsl_cls_VC.H:
                best_acc_seen_VC, best_acc_unseen_VC, best_gzsl_acc_VC = gzsl_cls_VC.acc_seen, gzsl_cls_VC.acc_unseen, gzsl_cls_VC.H
                best_acc_unseen_list_VC, best_acc_seen_list_VC = gzsl_cls_VC.best_acc_U_list, gzsl_cls_VC.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VC")
            log_record = 'GZSL (VC): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_VC.acc_unseen, gzsl_cls_VC.acc_seen, gzsl_cls_VC.H)
            print(log_record)
            logger.write(log_record + '\n')

            # Train GZSL classifier in VC
            gzsl_cls_VC = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr,
                                                0.5, 25, opt.syn_num, cls_mode="GZSL", con_size=2048, _train_C = train_C_pro, useC=True)
            if best_gzsl_acc_VC < gzsl_cls_VC.H:
                best_acc_seen_VC, best_acc_unseen_VC, best_gzsl_acc_VC = gzsl_cls_VC.acc_seen, gzsl_cls_VC.acc_unseen, gzsl_cls_VC.H
                best_acc_unseen_list_VC, best_acc_seen_list_VC = gzsl_cls_VC.best_acc_U_list, gzsl_cls_VC.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VC")
            log_record = 'GZSL pro (VC): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_VC.acc_unseen, gzsl_cls_VC.acc_seen, gzsl_cls_VC.H)
            print(log_record)
            logger.write(log_record + '\n')

            # Train GZSL classifier in VCS
            gzsl_cls_VCS = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                                 25, opt.syn_num, cls_mode="GZSL", netDec=zerodiff.netDec,
                                                 dec_size=opt.attSize, dec_hidden_size=4096, useS=True, con_size=2048, _train_C = train_C, useC=True)
            if best_gzsl_acc_VCS < gzsl_cls_VCS.H:
                best_acc_seen_VCS, best_acc_unseen_VCS, best_gzsl_acc_VCS = gzsl_cls_VCS.acc_seen, gzsl_cls_VCS.acc_unseen, gzsl_cls_VCS.H
                best_acc_unseen_list_VCS, best_acc_seen_list_VCS = gzsl_cls_VCS.best_acc_U_list, gzsl_cls_VCS.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VCS")
            log_record = 'GZSL (VCS): U: %.4f, S: %.4f, H: %.4f' % (
                gzsl_cls_VCS.acc_unseen, gzsl_cls_VCS.acc_seen, gzsl_cls_VCS.H)
            print(log_record)
            logger.write(log_record + '\n')

            # Train GZSL classifier in VC
            gzsl_cls_VCS = classifier.CLASSIFIER(train_X_pro, train_Y_pro, data, nclass, opt.cuda, opt.classifier_lr,
                                                 0.5,  25, opt.syn_num, cls_mode="GZSL", netDec=zerodiff.netDec,
                                                 dec_size=opt.attSize, dec_hidden_size=4096, useS=True,
                                                 con_size=2048, _train_C = train_C_pro, useC=True)  #
            if best_gzsl_acc_VCS < gzsl_cls_VCS.H:
                best_acc_seen_VCS, best_acc_unseen_VCS, best_gzsl_acc_VCS = gzsl_cls_VCS.acc_seen, gzsl_cls_VCS.acc_unseen, gzsl_cls_VCS.H
                save_zerodiff(zerodiff, model_save_name, "gzsl")
                best_acc_unseen_list_VCS, best_acc_seen_list_VCS = gzsl_cls_VCS.best_acc_U_list, gzsl_cls_VCS.best_acc_S_list
                save_zerodiff(zerodiff, model_save_name, "gzsl_VCS")

            log_record = 'GZSL pro (VCS): U: %.4f, S: %.4f, H: %.4f' % (
                gzsl_cls_VCS.acc_unseen, gzsl_cls_VCS.acc_seen, gzsl_cls_VCS.H)
            print(log_record)
            logger.write(log_record + '\n')

        # Zero-shot learning
        # Train ZSL classifier in V
        zsl_cls_V = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                          data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                          opt.syn_num, cls_mode="ZSL")
        acc = zsl_cls_V.acc
        if best_zsl_acc_V < acc:
            best_zsl_acc_V = acc
            best_acc_zsl_list_V = zsl_cls_V.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_V")
        log_record = 'ZSL (V): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in V
        zsl_cls_V = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                          data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                          opt.syn_num,  cls_mode="ZSL")

        acc = zsl_cls_V.acc
        if best_zsl_acc_V < acc:
            best_zsl_acc_V = acc
            best_acc_zsl_list_V = zsl_cls_V.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_V")
        log_record = 'ZSL pro (V): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in VS
        zsl_cls_VS = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                           data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="ZSL", netDec=zerodiff.netDec, dec_size=opt.attSize,
                                           dec_hidden_size=4096, useS=True)
        acc = zsl_cls_VS.acc
        if best_zsl_acc_VS < acc:
            best_zsl_acc_VS = acc
            best_acc_zsl_list_VS = zsl_cls_VS.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VS")
        log_record = 'ZSL (VS): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in VS
        zsl_cls_VS = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                           data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="ZSL", netDec=zerodiff.netDec, dec_size=opt.attSize,
                                           dec_hidden_size=4096, useS=True)

        acc = zsl_cls_VS.acc
        if best_zsl_acc_VS < acc:
            best_zsl_acc_VS = acc
            best_acc_zsl_list_VS = zsl_cls_VS.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VS")
        log_record = 'ZSL pro (VS): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in C
        zsl_cls_C = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                          data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                          opt.syn_num, cls_mode="ZSL", useV=False, con_size=2048, _train_C = syn_con, useC=True)
        acc = zsl_cls_C.acc
        if best_zsl_acc_C < acc:
            best_zsl_acc_C = acc
            best_acc_zsl_list_C = zsl_cls_C.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_C")
        log_record = 'ZSL (C): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in C
        zsl_cls_C = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                          data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                          opt.syn_num, cls_mode="ZSL", useV=False, con_size=2048, _train_C = syn_con_pro,  useC=True)
        acc = zsl_cls_C.acc
        if best_zsl_acc_C < acc:
            best_zsl_acc_C = acc
            best_acc_zsl_list_C = zsl_cls_C.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_C")
        log_record = 'ZSL pro (C): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in VC
        zsl_cls_VC = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                           data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="ZSL", con_size=2048, _train_C = syn_con, useC=True)
        acc = zsl_cls_VC.acc
        if best_zsl_acc_VC < acc:
            best_zsl_acc_VC = acc
            best_acc_zsl_list_VC = zsl_cls_VC.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VC")
        log_record = 'ZSL (VC): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in VC
        zsl_cls_VC = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                           data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="ZSL", con_size=2048, _train_C = syn_con_pro, useC=True)
        acc = zsl_cls_VC.acc
        if best_zsl_acc_VC < acc:
            best_zsl_acc_VC = acc
            best_acc_zsl_list_VC = zsl_cls_VC.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VC")
        log_record = 'ZSL pro (VC): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in VCS
        zsl_cls_VCS = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                                            data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                            opt.syn_num, cls_mode="ZSL", netDec=zerodiff.netDec, dec_size=opt.attSize,
                                            dec_hidden_size=4096, useS=True, con_size=2048, _train_C = syn_con,  useC=True)
        acc = zsl_cls_VCS.acc
        if best_zsl_acc_VCS < acc:
            best_zsl_acc_VCS = acc
            best_acc_zsl_list_VCS = zsl_cls_VCS.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VCS")
        log_record = 'ZSL (VCS): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train ZSL classifier in VC
        zsl_cls_VCS = classifier.CLASSIFIER(syn_feature_pro, util.map_label(syn_label_pro, data.unseenclasses), \
                                            data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                            opt.syn_num, cls_mode="ZSL", netDec=zerodiff.netDec, dec_size=opt.attSize,
                                            dec_hidden_size=4096, useS=True, con_size=2048, _train_C = syn_con_pro,
                                            useC=True)
        acc = zsl_cls_VCS.acc
        if best_zsl_acc_VCS < acc:
            best_zsl_acc_VCS = acc
            best_acc_zsl_list_VCS = zsl_cls_VCS.best_acc_zsl_list
            save_zerodiff(zerodiff, model_save_name, "zsl_VCS")
        log_record = 'ZSL pro (VCS): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train Seen classifier in V
        seen_cls_V = classifier.CLASSIFIER(syn_feature_seen, util.map_label(syn_label_seen, data.seenclasses), \
                                           data, data.seenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                           opt.syn_num, cls_mode="seen")
        acc = seen_cls_V.acc
        if best_seen_acc_V < acc:
            best_seen_acc_V = acc
        log_record = 'Seen (V): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # reset G to training mode
        zerodiff.train()

        if opt.gzsl:
            log_record = "best GZSL (V): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_V, best_acc_seen_V, best_gzsl_acc_V)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (V): " + str(best_acc_seen_list_V)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (V): " + str(best_acc_unseen_list_V)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best GZSL (VS): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_VS, best_acc_seen_VS, best_gzsl_acc_VS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (VS): " + str(best_acc_seen_list_VS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (VS): " + str(best_acc_unseen_list_VS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best GZSL (C): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_C, best_acc_seen_C, best_gzsl_acc_C)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (C): " + str(best_acc_seen_list_C)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (C): " + str(best_acc_unseen_list_C)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best GZSL (VC): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_VC, best_acc_seen_VC, best_gzsl_acc_VC)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (VC): " + str(best_acc_seen_list_VC)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (VC): " + str(best_acc_unseen_list_VC)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best GZSL (VCS): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_VCS, best_acc_seen_VCS, best_gzsl_acc_VCS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_seen_list (VCS): " + str(best_acc_seen_list_VCS)
            print(log_record)
            logger.write(log_record + '\n')

            log_record = "best_acc_unseen_list (VCS): " + str(best_acc_unseen_list_VCS)
            print(log_record)
            logger.write(log_record + '\n')

        log_record = 'best ZSL (V): %.4f' % (best_zsl_acc_V.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (V): " + str(best_acc_zsl_list_V)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best ZSL (VS): %.4f' % (best_zsl_acc_VS.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (VS): " + str(best_acc_zsl_list_VS)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best ZSL (C): %.4f' % (best_zsl_acc_C.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (C): " + str(best_acc_zsl_list_C)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best ZSL (VC): %.4f' % (best_zsl_acc_VC.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (VC): " + str(best_acc_zsl_list_VC)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best ZSL (VCS): %.4f' % (best_zsl_acc_VCS.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = "best_acc_zsl_list (VCS): " + str(best_acc_zsl_list_VCS)
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best seen (V): %.4f' % (best_seen_acc_V.item())
        print(log_record)
        logger.write(log_record + '\n')


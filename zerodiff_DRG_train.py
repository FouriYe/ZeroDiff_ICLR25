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
from config_zerodiff import opt
import zerodiff_tools
import torch.nn.functional as F
import classifiers.classifier_images as classifier

def save_model(netG_con, model_save_name, post):
    torch.save({'state_dict_G_con': netG_con.state_dict(),
                }, model_save_name + post + '.tar')


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "w")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()

logger_name = "./log/%s/zerodiff_DRG_%dpercent_att:%s_b:%d_lr:%s_n_T:%d_betas:%s,%s_gamma:ADV:%.1f_VAE:%.1f_x0:%.1f_xt:%.1f_dist:%.1f_num:%s" % (
    opt.dataset, opt.split_percent, opt.class_embedding, opt.batch_size, str(opt.lr), opt.n_T, str(opt.ddpmbeta1),
    str(opt.ddpmbeta2), opt.gamma_ADV, opt.gamma_VAE, opt.gamma_x0, opt.gamma_xt, opt.gamma_dist, opt.syn_num)
logger = Logger(logger_name)
model_save_name = "./out/%s/zerodiff_DRG_%dpercent_att:%s_b:%d_lr:%s_n_T:%d_betas:%s,%s_gamma:ADV:%.1f_VAE:%.1f_x0:%.1f_xt:%.1f_dist:%.1f_num:%d" % (
    opt.dataset, opt.split_percent, opt.class_embedding, opt.batch_size, str(opt.lr), opt.n_T, str(opt.ddpmbeta1),
    str(opt.ddpmbeta2), opt.gamma_ADV, opt.gamma_VAE, opt.gamma_x0, opt.gamma_xt, opt.gamma_dist, opt.syn_num)

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
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.noiseSize)
##########
# Cuda
if opt.cuda:
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_label = input_label.cuda()
    input_con = input_con.cuda()


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), size_average=False)
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return (BCE + KLD)


def sample(batch_size):
    batch_feature, batch_con, batch_att, batch_label = data.next_seen_batch(batch_size)
    input_res.copy_(batch_feature)
    input_con.copy_(batch_con)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)
    return input_res, input_con, input_att, input_label


def WeightedL14att(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)


def generate_syn_feature(ddpmgan, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_feature_con = torch.FloatTensor(nclass * num, 2048)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    if opt.cuda:
        syn_att = syn_att.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        att = Variable(syn_att, volatile=True)

        fake_con = ddpmgan.sample_from_model(att)
        syn_feature.narrow(0, i * num, num).copy_(fake_con.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


class ZERODIFF_DRG(torch.nn.Module):
    def __init__(self, data, n_T, betas, seenclasses, unseenclasses, attribute, device='cuda'):
        super(ZERODIFF_DRG, self).__init__()
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

        self.netG_con = zerodiff_tools.DRG_Generator(opt).to(self.device)
        self.netD_c0 = zerodiff_tools.DFG_Discriminator_x0(opt).to(self.device)
        self.netD_ct = zerodiff_tools.DRG_Discriminator_ct(opt).to(self.device)
        self.netDec = zerodiff_tools.V2S_mapping(opt, opt.attSize).to(self.device)

        self.optimizerG_con = optim.Adam(self.netG_con.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD_c0 = optim.Adam(self.netD_c0.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD_ct = optim.Adam(self.netD_ct.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerDec = optim.Adam(self.netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))

        self.gamma_x0 = opt.gamma_x0
        self.gamma_xt = opt.gamma_xt
        self.gamma_recons = opt.gamma_recons
        self.lambda1 = opt.lambda1

        self.loss_mse = torch.nn.MSELoss()

        self.batch_size = opt.batch_size
        self.data = data

    def forward(self):
        for iter_d in range(opt.critic_iter):
            _, con_0_real, att_0_real, label = sample(self.batch_size)
            D_cost, Wasserstein_D = self.update_D(con_0_real, att_0_real, label)
        G_cost, loss_att_mse = self.update_G(con_0_real, att_0_real, label)

        return D_cost, Wasserstein_D, G_cost, loss_att_mse

    def update_D(self, con_0_real, att_0_real, label):
        for p in self.netD_c0.parameters():
            p.requires_grad = True
        for p in self.netD_ct.parameters():
            p.requires_grad = True
        for p in self.netDec.parameters():
            p.requires_grad = True
        for p in self.netG_con.parameters():
            p.requires_grad = False

        self.netD_c0.zero_grad()
        self.netD_ct.zero_grad()
        self.netDec.zero_grad()

        att_0_recons = self.netDec(con_0_real)
        R_cost = self.loss_mse(att_0_recons, att_0_real)
        R_cost.backward()

        # contrastive reconstruction seen
        z = torch.randn(self.batch_size, self.dim_noise).to(self.device)
        _ts_con = torch.randint(0, self.n_T, (self.batch_size,), dtype=torch.int64).to(self.device)
        con_t_real, con_tp1_real = self.q_sample_pairs(con_0_real, _ts_con)

        con_0_fake = self.netG_con(z, att_0_real, con_t_real, _ts_con)
        # NOTE!!!:
        # The true code should be "con_0_fake = self.netG_con(z, att_0_real, con_tp1_real, _ts_con)"
        # If that, however, the performance deceases significantly.
        # The reason is unknown.

        con_t_fake = self.sample_posterior(con_0_fake, con_tp1_real, _ts_con)

        criticD_real_c0 = -self.netD_c0(con_0_real, att_0_real).mean()
        criticD_real_ct = -self.netD_ct(con_t_real, con_tp1_real, att_0_real, _ts_con).mean()
        criticD_real = self.gamma_x0 * criticD_real_c0 + self.gamma_xt * criticD_real_ct
        criticD_real.backward()

        criticD_fake_c0 = self.netD_c0(con_0_fake.detach(), att_0_real).mean()
        criticG_fake_ct = self.netD_ct(con_t_fake.detach(), con_tp1_real, att_0_real, _ts_con).mean()
        criticD_fake = self.gamma_x0 * criticD_fake_c0 + self.gamma_xt * criticG_fake_ct
        criticD_fake.backward()

        Wasserstein_D = criticD_real - criticD_fake

        gp_c0 = self.netD_c0.calc_gradient_penalty(con_0_real, con_0_fake.data, att_0_real, self.lambda1)
        gp_ct = self.netD_ct.calc_gradient_penalty(con_t_real, con_t_fake.data, con_tp1_real, att_0_real, _ts_con, self.lambda1)
        gp = self.gamma_x0 * gp_c0 + self.gamma_xt * gp_ct
        gp.backward()

        D_cost = criticD_fake - criticD_real + gp

        self.optimizerDec.step()
        self.optimizerD_c0.step()
        self.optimizerD_ct.step()

        return D_cost, Wasserstein_D

    def update_G(self, con_0_real, att_0_real, label):
        for p in self.netG_con.parameters():
            p.requires_grad = True
        for p in self.netD_c0.parameters():
            p.requires_grad = False
        for p in self.netD_ct.parameters():
            p.requires_grad = False
        for p in self.netDec.parameters():  # freeze decoder
            p.requires_grad = False

        self.netG_con.zero_grad()

        # contrastive reconstruction seen
        z = torch.randn(self.batch_size, self.dim_noise).to(self.device)
        _ts_con = torch.randint(0, self.n_T, (self.batch_size,), dtype=torch.int64).to(self.device)
        con_t_real, con_tp1_real = self.q_sample_pairs(con_0_real, _ts_con)

        con_0_fake = self.netG_con(z, att_0_real, con_t_real, _ts_con)
        # NOTE!!!:
        # The true code should be "con_0_fake = self.netG_con(z, att_0_real, con_tp1_real, _ts_con)"
        # If that, however, the performance deceases significantly.
        # The reason is unknown.

        con_t_fake = self.sample_posterior(con_0_fake, con_tp1_real, _ts_con)

        errG = 0.0

        att_0_recons = self.netDec(con_0_fake)
        loss_att_mse = self.loss_mse(att_0_recons, att_0_real)

        criticG_fake_c0 = -self.netD_c0(con_0_fake, att_0_real).mean()
        criticG_fake_ct = -self.netD_ct(con_t_fake, con_tp1_real, att_0_real, _ts_con).mean()
        criticG_fake = self.gamma_x0 * criticG_fake_c0 + self.gamma_xt * criticG_fake_ct
        G_cost = criticG_fake
        errG += G_cost

        errG.backward()
        self.optimizerG_con.step()
        return G_cost, loss_att_mse

    def sample_from_model(self, att):
        n_sample = att.shape[0]
        with torch.no_grad():
            z_con = torch.randn(n_sample, self.dim_noise).to(self.device)
            _ts_con = (self.n_T - 1) + torch.zeros((n_sample,), dtype=torch.int64).to(self.device)
            con_t_real = torch.randn(n_sample, 2048).to(self.device)
            con_0_fake = self.netG_con(z_con, att, con_t_real, _ts_con)
        return con_0_fake

    def q_sample_pairs(self, x_0, t):
        """
        Generate a pair of disturbed images for training, use prior_coefficients
        :param x_0: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        t = t.long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t)
        x_t_plus_one = zerodiff_tools.extract(self.prior_coefficients.sqrt_alphas, t + 1, x_0.shape) * x_t + \
                       zerodiff_tools.extract(self.prior_coefficients.sigmas, t + 1, x_0.shape) * noise
        return x_t, x_t_plus_one

    def q_sample(self, x_0, t):
        """
        use prior_coefficients
        q(x_{t}|x_0,t)
        Diffuse the data (t == 0 means diffused for t step)
        """
        t = t.long()
        noise = torch.randn_like(x_0)
        x_t = zerodiff_tools.extract(self.prior_coefficients.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              zerodiff_tools.extract(self.prior_coefficients.sigmas_bar, t, x_0.shape) * noise
        return x_t

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


zerodiff_drg = ZERODIFF_DRG(data, n_T=opt.n_T, betas=(opt.ddpmbeta1, opt.ddpmbeta2), seenclasses=data.seenclasses,
                  unseenclasses=data.unseenclasses, attribute=data.attribute, device='cuda')

best_gzsl_acc_C = 0
best_acc_seen_C = 0
best_acc_unseen_C = 0
best_zsl_acc_C = 0
best_seen_acc_C = 0

best_gzsl_acc_C = 0
best_acc_seen_C = 0
best_acc_unseen_C = 0
best_zsl_acc_C = 0
best_seen_acc_C = 0

nclass = opt.nclass_all

data.train_feature, data.test_seen_feature, data.test_unseen_feature = data.train_paco, data.test_seen_paco, data.test_unseen_paco
for epoch in range(0, opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        D_cost, Wasserstein_D, G_cost, loss_att_mse = zerodiff_drg()

    log_record = '[%d/%d] D_cost:%.4f, Wasserstein_D:%.4f' % (epoch, opt.nepoch, D_cost.item(), Wasserstein_D.item())
    print(log_record)
    logger.write(log_record + '\n')

    log_record = '[%d/%d] G_cost:%.4f, loss_att_mse:%.4f' % (
    epoch, opt.nepoch, G_cost.item(), loss_att_mse.item())
    print(log_record)
    logger.write(log_record + '\n')

    if epoch % opt.eval_interval == 0 or epoch == (opt.nepoch - 1):
        zerodiff_drg.eval()
        syn_unseen_feat, syn_unseen_label = generate_syn_feature(zerodiff_drg, data.unseenclasses, data.attribute, opt.syn_num)
        syn_seen_feat, syn_seen_label = generate_syn_feature(zerodiff_drg, data.seenclasses, data.attribute, opt.syn_num)

        # Generalized zero-shot learning
        if opt.gzsl:
            # Concatenate real seen features with synthesized unseen features
            train_X = torch.cat((data.train_feature.cpu(), syn_unseen_feat.cpu()), 0)
            train_Y = torch.cat((data.train_label, syn_unseen_label), 0)
            # Train GZSL classifier
            gzsl_cls_C = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                                               25, opt.syn_num, cls_mode="GZSL")
            if best_gzsl_acc_C < gzsl_cls_C.H:
                best_acc_seen_C, best_acc_unseen_C, best_gzsl_acc_C = gzsl_cls_C.acc_seen, gzsl_cls_C.acc_unseen, gzsl_cls_C.H
                save_model(zerodiff_drg.netG_con, model_save_name, '_gzsl')
            log_record = 'GZSL (C): U: %.4f, S: %.4f, H: %.4f' % (
            gzsl_cls_C.acc_unseen, gzsl_cls_C.acc_seen, gzsl_cls_C.H)
            print(log_record)
            logger.write(log_record + '\n')

        # Zero-shot learning
        # Train ZSL classifier
        zsl_cls_C = classifier.CLASSIFIER(syn_unseen_feat.cpu(), util.map_label(syn_unseen_label, data.unseenclasses),
                                          data,
                                          data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, \
                                          25, opt.syn_num, cls_mode="ZSL")
        acc = zsl_cls_C.acc
        if best_zsl_acc_C < acc:
            best_zsl_acc_C = acc
            save_model(zerodiff_drg.netG_con, model_save_name, '_zsl')
        log_record = 'ZSL (C): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # Train Seen classifier
        seen_cls_C = classifier.CLASSIFIER(syn_seen_feat.cpu(), util.map_label(syn_seen_label, data.seenclasses), data,
                                           data.seenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, \
                                           25, opt.syn_num, cls_mode="seen")
        acc = seen_cls_C.acc
        if best_seen_acc_C < acc:
            best_seen_acc_C = acc
        log_record = 'Seen (C): %.4f' % (acc)
        print(log_record)
        logger.write(log_record + '\n')

        # reset G to training mode
        zerodiff_drg.train()

        if opt.gzsl:
            log_record = "best GZSL (C): U: %.4f, S: %.4f, H: %.4f" % \
                         (best_acc_unseen_C, best_acc_seen_C, best_gzsl_acc_C)
            print(log_record)
            logger.write(log_record + '\n')

        log_record = 'best ZSL (C): %.4f' % (best_zsl_acc_C.item())
        print(log_record)
        logger.write(log_record + '\n')

        log_record = 'best seen (C): %.4f' % (best_seen_acc_C.item())
        print(log_record)
        logger.write(log_record + '\n')



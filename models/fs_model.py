import time
import torch
import torch.nn as nn
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
import itertools
import os
import cv2
import numpy as np
from .Ms_ssim import MS_SSIM
from neural_style.vgg import Vgg16
from models.fs_networks import Generator_Adain_Upsample, Discriminator
from torch.nn import functional as F
from models.ranger2020 import Ranger
import lpips
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class FsModel(object):
    def __init__(self, args):

        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.start_iter = args.start_iter
        self.img_size = args.img_size
        self.feat_match_loss = args.use_feat_matching
        self.lr = args.lr

        """ Weight """
        self.rec_weight = args.rec_weight
        self.id_weight = args.faceid_weight
        self.adv_weight = args.adv_weight
        self.fm_weight = args.fm_weight
        self.identity_weight = args.identity_weight

        self.device = "cuda"
        self.gpu_ids = args.gpu_ids
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.pretrained_weights = args.pretrained_weights

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print("##### Information #####")
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Weight #####")
        print("# rec_weight : ", self.rec_weight)
        print("# faceid_weight : ", self.id_weight)
        print("# adv_weight : ", self.adv_weight)
        print("# fm_weight : ", self.fm_weight)

    # *****************************Build Model****************************

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 20, self.img_size + 20)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor()])

        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])

        transformer_Arcface = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.train_src = ImageFolder(os.path.join('dataset', self.dataset, 'trainset'), train_transform)
        self.train_tar = ImageFolder(os.path.join('dataset', self.dataset, 'trainset'), train_transform)
        self.test_src = ImageFolder(os.path.join('dataset', self.dataset, 'testset'), test_transform)
        self.test_tar = ImageFolder(os.path.join('dataset', self.dataset, 'testset'), test_transform)

        self.train_src_loader = DataLoader(self.train_src, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.train_tar_loader = DataLoader(self.train_tar, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.test_src_loader = DataLoader(self.test_src, batch_size=1, shuffle=True, drop_last=True)
        self.test_tar_loader = DataLoader(self.test_tar, batch_size=1, shuffle=True, drop_last=True)

        """ Define discriminators and Generator """
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False).to(self.device)
        self.netD1 = Discriminator(input_nc=3, use_sigmoid=False).to(self.device)
        self.netD2 = Discriminator(input_nc=3, use_sigmoid=False).to(self.device)

        # Id network
        netArc_checkpoint = './pretrained_models/arcface_checkpoint.tar'
        netArc_checkpoint = torch.load(netArc_checkpoint)
        self.netArc = netArc_checkpoint['model'].module
        self.netArc = self.netArc.to(self.device)
        self.netArc.eval()

        # self.vgg = Vgg16(requires_grad=False).to(self.device)
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
        """ Define Loss """
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.L1_loss = nn.SmoothL1Loss().to(self.device)
        self.MS_ssim_loss = MS_SSIM(max_val=1)
        self.MS_ssim_w = 0.84
        # self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        """ Trainer """
        # self.E_optim = Ranger(self.encoder.parameters(), lr=self.lr)
        self.G_optim = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.D_optim = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)

    def train(self):
        self.netG.train(), self.netD1.train(), self.netD2.train()
        if self.pretrained_weights:
            params = torch.load('./pretrained_models/swapper.pt', map_location=self.device)
            self.netG.load_state_dict(params['netG'])
            self.netD1.load_state_dict(params['netD1'])
            self.netD2.load_state_dict(params['netD2'])
            print(" [*] Load {} Success".format(self.pretrained_weights))
            
        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(self.start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
            try:
                src_im, _ = train_src_iter.next()
            except:
                train_src_iter = iter(self.train_src_loader)
                src_im, _ = train_src_iter.next()

            try:
                tar_im, _ = train_tar_iter.next()
            except:
                train_tar_iter = iter(self.train_tar_loader)
                tar_im, _ = train_tar_iter.next()

            src_im, tar_im = src_im.to(self.device), tar_im.to(self.device)
            src_arc_im, tar_arc_im = self.detransform(src_im), self.detransform(tar_im)
            tar_img_id = F.interpolate(tar_arc_im, scale_factor=0.5)
            tar_latent_id = self.netArc(tar_img_id)
            tar_latent_id = F.normalize(tar_latent_id, p=2, dim=1)
            src_img_id = F.interpolate(src_arc_im, scale_factor=0.5)
            src_latent_id = self.netArc(src_img_id)
            src_latent_id = F.normalize(src_latent_id, p=2, dim=1)

            # Update Discriminator
            self.D_optim.zero_grad()

            fake_face = self.netG(src_im, tar_latent_id)
            real_D1_logit = self.netD1(src_im)
            real_D2_logit = self.netD2(F.interpolate(src_im, scale_factor=0.5))
            fake_D1_logit = self.netD1(fake_face.detach())
            fake_D2_logit = self.netD2(F.interpolate(fake_face, scale_factor=0.5).detach())

            D1_ad_loss = self.MSE_loss(real_D1_logit[5], torch.ones_like(real_D1_logit[5]).to(self.device)) + \
                           self.MSE_loss(fake_D1_logit[5], torch.zeros_like(fake_D1_logit[5]).to(self.device))
            D2_ad_loss = self.MSE_loss(real_D2_logit[5], torch.ones_like(real_D2_logit[5]).to(self.device)) + \
                           self.MSE_loss(fake_D2_logit[5], torch.zeros_like(fake_D2_logit[5]).to(self.device))

            Grad_penalty_D1 = self.gradient_penalty(self.netD1, src_im, fake_face.detach())
            Grad_penalty_D2 = self.gradient_penalty(self.netD2, F.interpolate(src_im, scale_factor=0.5), F.interpolate(fake_face, scale_factor=0.5).detach())

            D_loss = self.adv_weight * (D1_ad_loss + D2_ad_loss) + 0.00001 * (Grad_penalty_D1 + Grad_penalty_D2)
            D_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()
            fake_face = self.netG(src_im, tar_latent_id)
            real_D1_logit = self.netD1(src_im)
            real_D2_logit = self.netD2(F.interpolate(src_im, scale_factor=0.5))
            fake_D1_logit = self.netD1(fake_face)
            fake_D2_logit = self.netD2(F.interpolate(fake_face, scale_factor=0.5))
            feat_real = [real_D1_logit, real_D2_logit]
            feat_fake = [fake_D1_logit, fake_D2_logit]

            # adv loss
            D_ad_loss = self.MSE_loss(fake_D1_logit[5].clone().detach(), torch.ones_like(fake_D1_logit[5]).to(self.device)) + \
                        self.MSE_loss(fake_D2_logit[5].clone().detach(), torch.ones_like(fake_D2_logit[5]).to(self.device))

            # ID loss
            fake_arc_face = self.detransform(fake_face)
            fake_face_down = F.interpolate(fake_arc_face, scale_factor=0.5)
            latent_fake = self.netArc(fake_face_down)
            latent_fake = F.normalize(latent_fake, p=2, dim=1)
            id_loss = (1 - self.cosin_metric(latent_fake, tar_latent_id))

            #rec loss
            rec_face = self.netG(fake_face, src_latent_id)
            rec_loss = self.MS_ssim_w * (1 - self.MS_ssim_loss(rec_face, src_im)) + (1 - self.MS_ssim_w) * self.L1_loss(rec_face, src_im)

            # LPIPS
            # feat_match_loss = self.lpips_loss(fake_face, src_im)


            # identity loss
            # identity_face = self.netG(src_im, src_latent_id)
            # identity_loss = self.MS_ssim_w * (1 - self.MS_ssim_loss(identity_face, src_im)) + (1 - self.MS_ssim_w) * self.L1_loss(identity_face, src_im)

            # Feature matching loss
            feat_match_loss = 0
            n_layers_D = 4
            num_D = 2
            if self.feat_match_loss:
                feat_weights = 4.0 / (n_layers_D + 1)
                D_weights = 1.0 / num_D
                for i in range(num_D):
                    for j in range(2, len(feat_fake[i]) - 1):  # five layers
                        feat_match_loss += D_weights * feat_weights * self.L1_loss(feat_fake[i][j], feat_real[i][j].detach())

            G_loss = self.adv_weight * D_ad_loss + self.id_weight * id_loss + self.fm_weight * feat_match_loss + self.rec_weight * rec_loss
            G_loss = torch.mean(G_loss)
            G_loss.backward()
            self.G_optim.step()


            if step % 10 == 0:
                print("[%5d/%5d] time: %4.4f dis_loss: %.8f, gen_loss: %.8f" % (step, self.iteration, time.time() - start_time, D_loss, G_loss.item()))
            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 4, 0, 3))

                self.netG.eval(), self.netD1.eval(), self.netD2.eval()
                with torch.no_grad():
                    for _ in range(train_sample_num):
                        try:
                            src_im, _ = train_src_iter.next()
                        except:
                            train_src_iter = iter(self.train_src_loader)
                            src_im, _ = train_src_iter.next()

                        try:
                            tar_im, _ = train_tar_iter.next()
                        except:
                            train_tar_iter = iter(self.train_tar_loader)
                            tar_im, _ = train_tar_iter.next()

                        src_im, tar_im = src_im.to(self.device), tar_im.to(self.device)
                        src_arc_im, tar_arc_im = self.detransform(src_im), self.detransform(tar_im)
                        tar_img_id = F.interpolate(tar_arc_im, scale_factor=0.5)
                        tar_latent_id = self.netArc(tar_img_id)
                        tar_latent_id = F.normalize(tar_latent_id, p=2, dim=1)
                        src_img_id = F.interpolate(src_arc_im, scale_factor=0.5)
                        src_latent_id = self.netArc(src_img_id)
                        src_latent_id = F.normalize(src_latent_id, p=2, dim=1)
                        gen_face = self.netG(src_im, tar_latent_id)
                        src_id_face = self.netG(src_im, src_latent_id)


                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(src_im[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(tar_im[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(gen_face[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(src_id_face[0])))), 0)),1)



                    for _ in range(test_sample_num):
                        try:
                            src_im, _ = test_src_iter.next()
                        except:
                            test_src_iter = iter(self.test_src_loader)
                            src_im, _ = test_src_iter.next()

                        try:
                            tar_im, _ = test_tar_iter.next()
                        except:
                            test_tar_iter = iter(self.test_tar_loader)
                            tar_im, _ = test_tar_iter.next()

                        src_im, tar_im = src_im.to(self.device), tar_im.to(self.device)
                        src_arc_im, tar_arc_im = self.detransform(src_im), self.detransform(tar_im)
                        tar_img_id = F.interpolate(tar_arc_im, scale_factor=0.5)
                        tar_latent_id = self.netArc(tar_img_id)
                        tar_latent_id = F.normalize(tar_latent_id, p=2, dim=1)
                        src_img_id = F.interpolate(src_arc_im, scale_factor=0.5)
                        src_latent_id = self.netArc(src_img_id)
                        src_latent_id = F.normalize(src_latent_id, p=2, dim=1)
                        gen_face = self.netG(src_im, tar_latent_id)
                        src_id_face = self.netG(src_im, src_latent_id)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(src_im[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(tar_im[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(gen_face[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(src_id_face[0])))), 0)),1)


                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'enc_%07d.png' % step), A2B * 255.0)

                self.netG.train(), self.netD1.train(), self.netD2.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % 3000 == 0:
                params = {}
                params['netG'] = self.netG.state_dict()
                params['netD1'] = self.netD1.state_dict()
                params['netD2'] = self.netD2.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['netG'] = self.netG.state_dict()
        params['netD1'] = self.netD1.state_dict()
        params['netD2'] = self.netD2.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.netG.load_state_dict(params['netG'])
        self.netG.load_state_dict(params['netD1'])
        self.netG.load_state_dict(params['netD2'])

    def gradient_penalty(self, net_D, img_att, img_fake):
        # interpolate sample
        bs = img_fake.shape[0]
        alpha = torch.rand(bs, 1, 1, 1).expand_as(img_fake).to(self.device)
        interpolated = (alpha * img_att + (1 - alpha) * img_fake).requires_grad_(True)
        pred_interpolated = net_D(interpolated)
        pred_interpolated = pred_interpolated[-1]

        # compute gradients
        grad = torch.autograd.grad(outputs=pred_interpolated,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(pred_interpolated.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        loss_d_gp = torch.mean((grad_l2norm) ** 2)
        return loss_d_gp


    def cosin_metric(self, x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def detransform(self, src):
        src[:, 0, :, :] = (src[:, 0, :, :] - 0.485) / 0.229
        src[:, 1, :, :] = (src[:, 1, :, :] - 0.456) / 0.224
        src[:, 2, :, :] = (src[:, 2, :, :] - 0.406) / 0.225
        return src
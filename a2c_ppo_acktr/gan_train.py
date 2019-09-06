#############################################################################
# Import                                                                    #
#############################################################################
import os
import math
import random
import argparse

import itertools

import numpy as np
from scipy import io
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

import a2c_ppo_acktr.gan_models as models 



class Gan_segmenter:
    def __init__(self,opt):
        self.opt = opt
        
        opt.device = "cuda:0"
        self.device = torch.device(opt.device)
        cudnn.benchmark = True
            
        def weights_init_ortho(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                # nn.init.zeros_(m.weight) 
                nn.init.orthogonal_(m.weight, opt.initOrthoGain)
                
        self.netEncM = models._netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(self.device)
        self.netGenX = models._netGenX(sizex=opt.sizex, nOut=opt.nx, nc=opt.nz, nf=opt.nfX, nMasks=opt.nMasks, selfAtt=opt.useSelfAttG).to(self.device)
        self.netDX = models._resDiscriminator128(nIn=opt.nx, nf=opt.nfD, selfAtt=opt.useSelfAttD).to(self.device)

        self.netEncM.apply(weights_init_ortho)
        self.netGenX.apply(weights_init_ortho)
        self.netDX.apply(weights_init_ortho)
        print(self.netEncM.cnn[0].mods[1].weight.data[0,0,:,:])
        self.optimizerEncM = torch.optim.Adam(self.netEncM.parameters(), lr=opt.lrM, betas=(0, 0.9), weight_decay=opt.wdecay, amsgrad=False)
        self.optimizerGenX = torch.optim.Adam(self.netGenX.parameters(), lr=opt.lrG, betas=(0, 0.9), amsgrad=False)
        self.optimizerDX = torch.optim.Adam(self.netDX.parameters(), lr=opt.lrD, betas=(0, 0.9), amsgrad=False)

        if opt.wrecZ > 0:
            self.netRecZ = models._netRecZ(sizex=opt.sizex, nIn=opt.nx, nc=opt.nz, nf=opt.nfZ, nMasks=opt.nMasks).to(self.device)
            self.netRecZ.apply(weights_init_ortho)
            self.optimizerRecZ = torch.optim.Adam(self.netRecZ.parameters(), lr=opt.lrZ, betas=(0, 0.9), amsgrad=False)

            self.mask_enc = models.mask_encoder().to(self.device)
            # self.mask_enc.apply(weights_init_ortho)
            self.optimizermask_enc = torch.optim.Adam(self.mask_enc.parameters(), lr=opt.lrM, betas=(0, 0.9), amsgrad=False)
        
  
    def train_segmenter(self,data_g):
        # print('TRAIN')
        
        xData = data_g.to(self.device)
        xReal = xData
        zData = torch.randn((xData.size(0), self.opt.nMasks, self.opt.nz, 1, 1), device=self.device)
        self.netEncM.zero_grad()
        self.netGenX.zero_grad()
        self.netDX.zero_grad()
        self.netEncM.train()
        self.netGenX.train()
        self.netDX.train()
        
      
        self.mask_enc.zero_grad()
        self.mask_enc.train()
        if self.opt.wrecZ > 0:
            self.netRecZ.zero_grad()
            self.netRecZ.train()


        dStep = (self.opt.iteration % self.opt.dStepFreq == 0)
        gStep = (self.opt.iteration % self.opt.gStepFreq == 0)
        #########################  AutoEncode X #########################
        if gStep:
            mEnc = self.netEncM(xData)
            mask_loss = torch.mean(0.5- torch.abs(mEnc-0.5))

            hGen = self.netGenX(mEnc, zData)

            xGen = (hGen + ((1 - mEnc.unsqueeze(2)) * xData.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))
            dGen = self.netDX(xGen)
            lossG = - dGen.mean()
            if self.opt.wrecZ > 0:
                zRec = self.netRecZ(hGen.sum(1))
                err_recZ = ((zData - zRec) * (zData - zRec)).mean()
                lossG += err_recZ * self.opt.wrecZ + mask_loss
            lossG.backward()
            self.optimizerEncM.step()
            self.optimizerGenX.step()
            if self.opt.wrecZ > 0:
                self.optimizerRecZ.step()
        if dStep:
            self.netDX.zero_grad()
            with torch.no_grad():
                mEnc = self.netEncM(xData)
                hGen = self.netGenX(mEnc, zData)
                xGen = (hGen + ((1 - mEnc.unsqueeze(2)) * xData.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))
            dPosX = self.netDX(xReal)
            dNegX = self.netDX(xGen)
            err_dPosX = (-1 + dPosX)
            err_dNegX = (-1 - dNegX)
            err_dPosX = ((err_dPosX < 0).float() * err_dPosX).mean()
            err_dNegX = ((err_dNegX < 0).float() * err_dNegX).mean()
            (-err_dPosX - err_dNegX).backward()
            self.optimizerDX.step()
            del mEnc
            del hGen 
            del xGen
        self.opt.iteration += 1
        # print('ITER: ',self.opt.iteration)    
        if self.opt.iteration % self.opt.checkpointFreq == 0:
            
            self.test(xData)
            self.save_models()

            # if self.opt.iteration % 250000 == 0:
            #     self.save_models()
           
            self.netEncM.train()
            self.netGenX.train()
            self.netDX.train()
            if self.opt.wrecZ > 0:
                self.netRecZ.train()


    def save_models(self):
        self.netEncM.zero_grad()
        try:
            torch.save(self.netEncM,  'ASRL_models/'+self.opt.env_name[:4]+'_'+self.opt.log_dir[-5:]+'_'+str(self.opt.iteration)+'.pth')
        except:
            print("Cannot save checkpoint")
        # self.netEncM.zero_grad()
        # self.netGenX.zero_grad()
        # self.netDX.zero_grad()
        # stateDic = {
        #     'netEncM': self.netEncM.state_dict(),
        #     'netGenX': self.netGenX.state_dict(),
        #     'netDX': self.netDX.state_dict(),
        #     'optimizerEncM': self.optimizerEncM.state_dict(),
        #     'optimizerGenX': self.optimizerGenX.state_dict(),
        #     'optimizerDX': self.optimizerDX.state_dict(),
        #     'options': self.opt,
        # }
        # if self.opt.wrecZ > 0:
        #     self.netRecZ.zero_grad()
        #     stateDic['netRecZ'] = self.netRecZ.state_dict()
        #     stateDic['optimizerRecZ'] = self.optimizerRecZ.state_dict(),
        # try:
        #     torch.save(stateDic, os.path.join(self.opt.outf, 'test_state_%05d.pth' % self.opt.iteration))
        #     torch.save(opt, os.path.join(self.opt.outf, "test_options.pth"))
        # except:
        #     print("Cannot save checkpoint")

        # if self.opt.clean and self.opt.iteration > self.opt.checkpointFreq:
        #     try:
        #         os.remove(os.path.join(self.opt.outf,  'test_state_%05d.pth' % (self.opt.iteration - self.opt.checkpointFreq)))
        #     except:
        #         pass
    
    def test(self,data):
        z_test = torch.randn((self.opt.nTest, self.opt.nMasks, self.opt.nz, 1, 1), device=self.device)
        zn_test = torch.randn((self.opt.nTest, self.opt.nz, 1, 1), device=self.device)

        test_data = data[:5]
        self.netEncM.eval()
        self.netGenX.eval()
        self.netDX.eval()
        if self.opt.wrecZ > 0:
            self.netRecZ.eval()
        # print(self.netEncM.cnn[0].mods[1].weight.data[0,0,:,:])


        out_X = torch.full((self.opt.nMasks, self.opt.nTest+1, self.opt.nTest+4, self.opt.nx, self.opt.sizex, self.opt.sizex), -1).to(self.device)
        
        out_X[:,1:,0] = test_data
        with torch.no_grad():
            mEnc_test = self.netEncM(test_data)
            out_X[:,1:,2] = mEnc_test.transpose(0,1).unsqueeze(2)*2-1
            out_X[:,1:,1] = ((out_X[:,1:,2] < 0).float() * -1) + (out_X[:,1:,2] > 0).float()
            out_X[:,1:,3] = (self.netGenX(mEnc_test, z_test) + ((1 - mEnc_test.unsqueeze(2)) * test_data.unsqueeze(1))).transpose(0,1)
            for k in range(self.opt.nMasks):
                for i in range(self.opt.nTest):
                    zx_test = z_test.clone()
                    zx_test[:, k] = zn_test[i]
                    out_X[k, 1:, i+4] = self.netGenX(mEnc_test, zx_test)[:,k] + ((1 - mEnc_test[:,k:k+1]) * test_data)
           
    
            try:
                vutils.save_image(out_X.view(-1,self.opt.nx,self.opt.sizex, self.opt.sizex), os.path.join(self.opt.outf, "loss2_out_%05d.png" % self.opt.iteration), normalize=True, range=(-1,1), nrow=self.opt.nTest+4)
            except:
                print("Cannot save output")







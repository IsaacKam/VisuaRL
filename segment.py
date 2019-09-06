from pca_project import PCAProjectNetALL
import os
from vgg import *
from PIL import Image
import torchvision.transforms as tvt
import torchvision.transforms.functional as TF
import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import visualpriors
import torchvision.utils as vutils
import a2c_ppo_acktr.gan_models as models 
from a2c_ppo_acktr.arguments import get_args
args = get_args()
# cv2.ocl.setUseOpenCL(False)
# if torch.cuda.is_available():
#         model = vgg11(pretrained=True).cuda()

#         pca = PCAProjectNetALL().cuda()

# else:
#         model = vgg11(pretrained=True)

#         pca = PCAProjectNetALL()

if args.priors == 3:

  dict = torch.load('seg_models/'+args.game+'.pth')
  netEncM = models._netEncM(sizex=128, nIn=3, nMasks=2, nRes=1,nf=64, temperature=1).cuda()
  netEncM.load_state_dict(dict,strict= False)
  for param in netEncM.parameters():
      param.requires_grad = False

def mask_maker(image_list,priors,device):
  
    init  =image_list
    proces = image_list.shape[0]

    num_of_images = image_list.shape[1]



#  DDT - Colour
    # image_list = F.interpolate(image_list.unsqueeze(1).repeat(1,3,1,1,1).view(1,-1,proces,84,84).squeeze().transpose(0,1), size=(224, 224), mode='bilinear', align_corners=False)
   
    # image_list = image_list.view(proces*num_of_images, 3,224, 224)/255 
    # features, _ = model(image_list)
    # features = features.to(device)
   
    # segmented = pca(features)
    # enlarged = init[:,6:9,:,:]/255
    # enlarged= enlarged* 2 - 1
    
    # # segment2D = visualpriors.feature_readout(enlarged, 'segment_unsup2d', device='cuda')
    # segmented = pca(segment2D)

    # first_project_map = torch.clamp(segmented[0], min=0)
    # first_maxv = first_project_map.view(first_project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    # first_project_map /= first_maxv
    # first_project_map = (F.interpolate(first_project_map.unsqueeze(1), size=(84, 84), mode='bilinear', align_corners=False) ).view(proces,1,84,84)

    # second_project_map = torch.clamp(segmented[1], min=0)
    # second_maxv = second_project_map.view(second_project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    # second_project_map /= second_maxv
    # second_project_map = (F.interpolate(second_project_map.unsqueeze(1), size=(84, 84), mode='bilinear', align_corners=False) ).view(proces,1,84,84)

    # third_project_map = torch.clamp(segmented[2], min=0)
    # third_maxv = third_project_map.view(third_project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    # third_project_map /= third_maxv
    # third_project_map = (F.interpolate(third_project_map.unsqueeze(1), size=(84, 84), mode='bilinear', align_corners=False) ).view(proces,1,84,84)

    # check = first_project_map[0,0,0,0]
    
    # if check != check:
    #     print("NAN Error, First Mask")
    #     first_project_map = init[:,6:9,:,:]

    # check = second_project_map[0,0,0,0]
    # if check != check:
    #     print("NAN Error, Second Mask")
    #     second_project_map = init[:,6:9,:,:]
    
    # check = third_project_map[0,0,0,0]

    # if check != check:
    #     print("NAN Error, Third Mask")
    #     third_project_map = init[:,6:9,:,:]






    # Graduated Canny - Colour Prior 2 is same intensity 3 different images, prior 1 - is different intensities
    segs = np.zeros((proces, 3, 84, 84))
    numpy_obs = np.uint8(init.permute(0,2, 3, 1).cpu().detach().numpy())  
    if priors == 2:
        
#         #     # Original Canny
#         # for i in range(proces):
#         #     segs[i,0,:,:] = cv2.Canny(numpy_obs[i,0,:,:],400,600)
#         #     segs[i,1,:,:] = cv2.Canny(numpy_obs[i,1,:,:],400,600)
#         #     segs[i,2,:,:] = cv2.Canny(numpy_obs[i,2,:,:],400,600)

    elif priors ==1:    
#         # #      Graduated Canny
        for i in range(proces):
            segs[i,0,:,:] = cv2.Canny(numpy_obs[i,:,:,6:9],100,150)
            segs[i,1,:,:] = cv2.Canny(numpy_obs[i,:,:,6:9],300,500)
            segs[i,2,:,:] = cv2.Canny(numpy_obs[i,:,:,6:9],540,550)

        # vutils.save_image( torch.from_numpy(segs[2]).float().unsqueeze(1),"goo.png", normalize=True, nrow=7)

        all_images = torch.cat([init,  torch.from_numpy(segs).float().cuda()],1).cuda()




# # #  Random 
# #     all_images = torch.cat([init, torch.rand(proces,3,84,84)*180],1).cuda()
    



    # ASRL SEGMENTS - prior 1 is mask multiplied with input prior 2 is just masks 0
    if priors == 3:
      input_extended = F.interpolate( init[:,6:9,:,:]/255, size=(128, 128), mode='bilinear', align_corners=False) 
      netEncM.eval()
      masks =  F.interpolate( netEncM(input_extended), size=(84, 84), mode='bilinear', align_corners=False) 
      greyscale_input =  (0.07 * init[:,8,:,:] + 0.72 * init[:,7,:,:] + 0.21 * init[:,6,:,:]).unsqueeze(1)
      
      features = masks

     

    # all_images = (torch.cat([init,features],1).cuda())

 





# #     # edges canny keypoints - Colour
#         enlarged = F.interpolate(init[:,6:9,:,:]/255, size=(256, 256), mode='bilinear', align_corners=False)
#         enlarged= enlarged* 2  -1
        # reshading = visualpriors.feature_readout(enlarged, 'reshading', device='cuda')
#         edges = visualpriors.feature_readout(enlarged, 'edge_texture', device='cuda')
#         keypoints = visualpriors.feature_readout(enlarged, 'keypoints2d', device='cuda')
#         features =  (F.interpolate(torch.cat([edges,keypoints,reshading],1), size=(84, 84), mode='bilinear', align_corners=False)/ 2. + 0.5)*255
#         # vutils.save_image(features[4].unsqueeze(1),"goo1.png", normalize=True, nrow=64)
#         all_images = (torch.cat([init,features],1).cuda())

    return all_images

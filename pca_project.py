import torch.nn as nn
import torch
import numpy as np
import time


class PCAProjectNet(nn.Module):
    def __init__(self):
        super(PCAProjectNet, self).__init__()

    def forward(self, features):     # features: NCWH
        k = features.size(0) * features.size(2) * features.size(3)
        x_mean = (features.sum(dim=2).sum(dim=2).sum(dim=0) / k).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        features = features - x_mean

        reshaped_features = features.view(features.size(0), features.size(1), -1)\
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)

        cov = torch.matmul(reshaped_features, reshaped_features.t()) / k
        # eigval, eigvec = torch.eig(cov, eigenvectors=True)
        _,_,eigvec = torch.svd(cov)


        first_compo = eigvec[:, 0]

        projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)\
            .view(features.size(0), features.size(2), features.size(3))

        maxv = projected_map.max()
        minv = projected_map.min()

        projected_map *= (maxv + minv) / torch.abs(maxv + minv)
        
        return projected_map

class PCAProjectNetALL(nn.Module):
    def __init__(self):
        super(PCAProjectNetALL, self).__init__()
    
    def forward(self, features):
        # start_time = time.time()

        # features: NCWH
        k = features.size(0) * features.size(2) * features.size(3)
        x_mean = (features.sum(dim=2).sum(dim=2).sum(dim=0) / k).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        
        # print("1--- %s seconds ---" % (time.time() - start_time))
        # d = time.time()
        features = features - x_mean
        # print("2--- %s seconds ---" % (time.time() - d))
        # d = time.time()
        reshaped_features = features.view(features.size(0), features.size(1), -1)\
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)
        
        cov = torch.matmul(reshaped_features, reshaped_features.t()) / k
        # print("3--- %s seconds ---" % (time.time() - d))
        # d = time.time()
        # print('cov',cov.shape)
        # eigval, eigvec = torch.eig(cov, eigenvectors=True)
        # print("4--- %s seconds ---" % (time.time() - d))
        d = time.time()
        _,_,eigvec = torch.svd(cov)
        # print("4s--- %s seconds ---" % (time.time() - d))
        # d = time.time()
        first_compo = eigvec[:, 0]
        
        first_projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)\
            .view(features.size(0), features.size(2), features.size(3))
        # print("5--- %s seconds ---" % (time.time() - d))
        # d = time.time()

        first_maxv = first_projected_map.max()
        first_minv = first_projected_map.min()
            
        first_projected_map *= (first_maxv + first_minv) / torch.abs(first_maxv + first_minv)
        # print("6--- %s seconds ---" % (time.time() - d))
        # d = time.time()  
        second_compo = eigvec[:, 1]
            
        second_projected_map = torch.matmul(second_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)\
            .view(features.size(0), features.size(2), features.size(3))
                
        second_maxv = second_projected_map.max()
        second_minv = second_projected_map.min()
                
        second_projected_map *= (second_maxv + second_minv) / torch.abs(second_maxv + second_minv)

        third_compo = eigvec[:, 2]
    
        third_projected_map = torch.matmul(third_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)\
        .view(features.size(0), features.size(2), features.size(3))
        
        third_maxv = third_projected_map.max()
        third_minv = third_projected_map.min()
        
        third_projected_map *= (third_maxv + third_minv) / torch.abs(third_maxv + third_minv)
        # print("7--- %s seconds ---" % (time.time() - d))
        # d = time.time()    
        return [first_projected_map,second_projected_map,third_projected_map]

if __name__ == '__main__':
    img = torch.randn(6, 512, 14, 14)
    pca = PCAProjectNet()
    pca(img)

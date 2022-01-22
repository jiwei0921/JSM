import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
from skimage import io
import os



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_contrastive



def loss_weight(input, target):

    _,c,w,h = target.size()
    loss_w = F.binary_cross_entropy_with_logits(input.clone().detach(), target.clone().detach(), reduction='none')

    loss_sample_tensor = torch.tensor(loss_w.data).mean(dim=1).mean(dim=1).mean(dim=1)
    loss_sample_weight = torch.softmax(1- loss_sample_tensor,dim=0)

    weight = loss_sample_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,c,w,h)

    return weight,loss_sample_weight



def update_pseudoLabel(ppath, depth_semantic,new_gt,epoch_num):

    batch_length = new_gt.size(0)
    [images_path, gts_path] = ppath
    images_path, gts_path = list(images_path), list(gts_path)


    # Backup pseudo label
    if not os.path.exists('./update'):
        os.mkdir('./update')
    dirname = './update/{}'.format(str(epoch_num))
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        os.system("cp -r {} {}".format(str(gts_path[0][:-len(gts_path[0].split('/')[-1])]), dirname))


    # Update fake label
    if not os.path.exists('./update/temp'):
        os.mkdir('./update/temp')
    for i in range(batch_length):
        res = new_gt[i,:,:,:]
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        Sal_name = './update/temp/' + gts_path[i].split('/')[-1]
        io.imsave(Sal_name, np.uint8(res * 255))
        os.system('python ../DenseCRF/examples/dense_hsal.py {} {} {}'.format(images_path[i], Sal_name, gts_path[i]))
    return None


import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from skimage import io
from tqdm import trange
# from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from model.TextNN import TextNN
from evaluateSOD.main import evalateSOD
from data import test_dataset



def eval_data(dataset_path, test_datasets, ckpt_name):

    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--snapshot', type=str, default=ckpt_name, help='checkpoint name')
    cfg = parser.parse_args()


    model_rgb = CPD_ResNet()
    model_depth = CPD_ResNet()
    model_text = TextNN()
    model_rgb.load_state_dict(torch.load('./ckpt/JSM_ResNet/'+'JSM_rgb.pth' +cfg.snapshot))
    model_depth.load_state_dict(torch.load('./ckpt/JSM_ResNet/' + 'JSM_depth.pth' +cfg.snapshot))
    model_text.load_state_dict(torch.load('./ckpt/JSM_ResNet/' +'JSM_text.pth' + cfg.snapshot))


    cuda = torch.cuda.is_available()
    if cuda:
        model_rgb.cuda()
        model_depth.cuda()
        model_text.cuda()
    model_rgb.eval()
    model_depth.eval()
    model_text.eval()



    for dataset in test_datasets:
        save_path = './results/JSM_ResNet/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/test_images/'
        gt_root = dataset_path + dataset + '/test_masks/'
        depth_root = dataset_path + dataset + '/test_depth/'
        test_loader = test_dataset(image_root, gt_root, depth_root, cfg.testsize)
        print('Evaluating dataset: %s' %(dataset))


        '''~~~ YOUR FRAMEWORK~~~'''
        for i in trange(test_loader.size):
            image, gt, _, name = test_loader.load_data()
            if cuda:
                image = image.cuda()

            # Only RGB Stream for predicting saliency
            _, res_r, _ = model_rgb(image)
            res = res_r.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            io.imsave(save_path+name, np.uint8(res * 255))


        # Calculate Metrics
        _ = evalateSOD(save_path, gt_root, dataset,ckpt_name,switch=False)
        '''switch is True: four metrics are evaluated, E,S,F,MAE'''
        '''switch is False: only MAE is measured, saving test cost'''
    return



if __name__ == '__main__':

    dataset_path = '../Dataset/test_data/'
    test_datasets=['NLPR']
    # test_datasets = ['NJUD', 'NLPR','STERE1000', 'SIP','LFSD', 'RGBD135','SSD','DUT']

    ckpt_name = '.48'

    eval_data(dataset_path,test_datasets,ckpt_name)

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from tqdm import tqdm
from datetime import datetime
from model.CPD_ResNet_models import CPD_ResNet
# from model.CPD_models import CPD_VGG
from model.Sal_depth import Saldepth
from model.TextNN import TextNN
from data import get_loader
from utils import clip_gradient, adjust_lr
from demo_test import eval_data
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from model.HolisticAttention import min_max_norm
from joint_semantic import loss_weight, update_pseudoLabel,ContrastiveLoss


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cudnn.benchmark = True


writer = SummaryWriter()
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_load', type=bool, default=False, help='whether load checkpoint or not')
parser.add_argument('--snapshot', type=int, default=20, help='load checkpoint number')
parser.add_argument('--tau', type=int, default=5, help='Update_Interval')

parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
opt = parser.parse_args()


image_root = './data/images/'
gt_root = './data/fake_mask/'
depth_root = './data/depth/'
caption_path = './data/caption/'

val_root = '../Dataset/test_data/'
validation = ['NJUD']


train_loader = get_loader(image_root, gt_root, depth_root, num_workers=12,caption=caption_path,batchsize=opt.batchsize, trainsize=opt.trainsize)

# build models

model_rgb = CPD_ResNet()        # Using CPD as feature-extraction Backbone for saliency network
model_depth = CPD_ResNet()      # Using CPD as feature-extraction Backbone for depth network
model_text = TextNN()           # TSM modeling
model = Saldepth()              # Smooth Operation for Spatial Supervision Generation


if opt.ckpt_load:
    model_rgb.load_state_dict(torch.load('./ckpt/JSM_ResNet/' + 'JSM_rgb.pth.' + str(opt.snapshot)))
    model_depth.load_state_dict(torch.load('./ckpt/JSM_ResNet/' + 'JSM_depth.pth.' + str(opt.snapshot)))
    model_text.load_state_dict(torch.load('./ckpt/JSM_ResNet/' + 'JSM_text.pth.' + str(opt.snapshot)))
    model.load_state_dict(torch.load('./ckpt/JSM_ResNet/' + 'JSM.pth.' + str(opt.snapshot)))


cuda = torch.cuda.is_available()
if cuda:
    model_rgb.cuda()
    model_depth.cuda()
    model_text.cuda()
    model.cuda()


params_rgb = model_rgb.parameters()
params_depth = model_depth.parameters()
params_text = model_text.parameters()
params = model.parameters()

optimizer_rgb = torch.optim.Adam(params_rgb, opt.lr)
optimizer_depth = torch.optim.Adam(params_depth, opt.lr)
optimizer_text = torch.optim.Adam(params_text, opt.lr)
optimizer = torch.optim.Adam(params, opt.lr)


total_step = len(train_loader)
BCE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
Distance = ContrastiveLoss()
nll_loss= torch.nn.NLLLoss()





def train(train_loader, model_rgb, model_depth, model_text, model,
          optimizer_rgb, optimizer_depth, optimizer_text, optimizer, epoch):
    model_rgb.train()
    model_depth.train()
    model_text.train()
    model.train()


    for i, pack in enumerate(tqdm(train_loader), start=1):
        iteration = i + epoch*len(train_loader)

        optimizer_rgb.zero_grad()
        optimizer_depth.zero_grad()
        optimizer_text.zero_grad()
        optimizer.zero_grad()

        # images, gts, depths,ppath,ori_data = pack
        images, gts, depths, ppath, ori_data, weakly_data = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        ori_data = [Variable(i) for i in ori_data]
        weakly_data = [Variable(i) for i in weakly_data]
        if cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            ori_data = [i.cuda() for i in ori_data]
            weakly_data = [i.cuda() for i in weakly_data]
        [cap_fea, semantic_fea, pos, cls_target, neg_target] = weakly_data # Dim: [[10,20,300],[10,300],[10],[10],[10]]


        '''~~~The Overall Framework~~~'''
        # RGB Visual Semantic Stream
        atts_rgb,dets_rgb,visual_feature= model_rgb(images)
        pred_sal = dets_rgb.detach()

        loss_rgb1 = BCE(atts_rgb, gts)
        loss_rgb2 = BCE(dets_rgb, gts)
        loss_rgb = (loss_rgb1 + loss_rgb2) / 2.0

        loss_rgb.backward()
        clip_gradient(optimizer_rgb, opt.clip)
        optimizer_rgb.step()


        # Depth Spatial Semantic Stream
        depth_pos_gt = model(images,depths,pred_sal)

        atts_depth, dets_depth, _ = model_depth(images)
        loss_depth1 = MSE(atts_depth, depth_pos_gt)
        loss_depth2 = MSE(dets_depth, depth_pos_gt)
        loss_depth = (loss_depth1 + loss_depth2) / 2.0

        loss_depth.backward()
        clip_gradient(optimizer_depth, opt.clip)
        optimizer_depth.step()


        # Caption Textual Semantic Stream
        visual_feat = visual_feature.detach()
        visual_pos_feat = torch.mul(visual_feat, pred_sal.sigmoid())
        out_logits_pos = model_text(visual_pos_feat, cap_fea, pos)
        out_logits_pos = F.log_softmax(out_logits_pos,dim=-1)
        loss_text_pos = nll_loss(out_logits_pos, cls_target)
        # loss_text = nll_loss(out_logits_pos, cls_target)

        visual_neg_feat = torch.mul(visual_feat, (1-pred_sal.sigmoid()))
        out_logits_neg = model_text(visual_neg_feat, cap_fea, pos)
        out_logits_neg = F.log_softmax(out_logits_neg, dim=-1)
        loss_text_neg = nll_loss(out_logits_neg, neg_target)    # background / non-salient class

        loss_text = (loss_text_pos + loss_text_neg) / 2.0

        loss_text.backward()
        clip_gradient(optimizer_text, opt.clip)
        optimizer_text.step()



        Update_Interval = opt.tau
        if (epoch + 1) % Update_Interval == 0:
            '''Update pseudo Label'''
            # Note that: we need to obtain original data with no augmentation to replace the fake label
            with torch.no_grad():
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                img_ori,gt_ori,depth_ori,caption_ori,pos_ori = ori_data
                _, det_rgb, vis_feature = model_rgb(img_ori)
                _, dets_depth, _ = model_depth(img_ori)
                S_pred = det_rgb.detach()
                D_pred = dets_depth.detach()


                '''Joint Semantic Mining'''
                vis_fea = vis_feature.detach()
                # Current GT visual feature
                vis_gt_feat = torch.mul(vis_fea, gt_ori)
                rate_gt = model_text(vis_gt_feat, caption_ori, pos_ori)
                rate_gt = F.log_softmax(rate_gt, dim=-1)

                # Depth Semantic Enhancement
                res = S_pred + S_pred - (1 - D_pred.sigmoid())
                zero = torch.zeros_like(res)
                depth_s = torch.tensor(torch.where(res <= 0.0, zero, res)).sigmoid()
                depth_semantic = min_max_norm(depth_s)
                vis_dep_feat = torch.mul(vis_fea, depth_semantic)
                rate_depth = model_text(vis_dep_feat, caption_ori, pos_ori)
                rate_depth = F.log_softmax(rate_depth, dim=-1)

                # Joint Semantic Generation
                bsz = depth_semantic.size(0)
                new_gt = gt_ori.detach()
                for jj in range(bsz):
                    # Obtaining the probability of predicted semantic word.
                    g_rate = rate_gt[jj,:][cls_target[jj]]
                    d_rate = rate_depth[jj,:][cls_target[jj]]
                    g_rate,d_rate = F.softmax(torch.stack([g_rate,d_rate]),dim=-1)

                    # Generating the updated label via joint semantic weighting.
                    # print('ori_fea',new_gt[jj,:,:,:])
                    new_gt[jj,:,:,:] = g_rate * gt_ori[jj,:,:,:] + d_rate * depth_semantic[jj,:,:,:]
                    # print('new_fea', new_gt[jj, :, :, :])

                update_pseudoLabel(ppath,depth_semantic,new_gt,int(epoch+1))


        '''~~~END~~~'''


        if i % 400 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_rgb: {:.4f} Loss_depth_sal: {:0.4f} '
                  'Loss_text: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_rgb.data, loss_depth.data, loss_text.data))
        writer.add_scalar('Loss/rgb', loss_rgb.item(), iteration)
        writer.add_scalar('Loss/depth', loss_depth.item(), iteration)
        writer.add_scalar('Loss/text', loss_text.item(), iteration)
        # writer.add_scalar('Loss/Sal_depth', loss.item(), iteration)
        # writer.add_images('Results/rgb', dets_rgb.sigmoid(), iteration)
        # writer.add_images('Results/pred_depth', dets_depth, iteration)
        # writer.add_images('Results/Sal_depth', depth_pos, iteration)


    save_path = 'ckpt/JSM_ResNet/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 2 == 0:
        torch.save(model_rgb.state_dict(), save_path + 'JSM_rgb.pth' + '.%d' % (epoch+1))
        torch.save(model_depth.state_dict(), save_path + 'JSM_depth.pth' + '.%d' % (epoch + 1))
        torch.save(model_text.state_dict(), save_path + 'JSM_text.pth' + '.%d' % (epoch + 1))
        torch.save(model.state_dict(), save_path + 'JSM.pth' + '.%d' % (epoch + 1))


print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer_rgb, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(optimizer_depth, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(optimizer_text, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model_rgb, model_depth, model_text, model,
          optimizer_rgb, optimizer_depth, optimizer_text, optimizer, epoch)
    if (epoch+1) % 2 == 0:
        ckpt_name = '.' + str(epoch+1)
        eval_data(val_root, validation,ckpt_name)
    if epoch >= opt.epoch -1:
        writer.close()

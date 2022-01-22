import os, sys
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch
import pandas as pd


#several data augumentation strategies
def cv_random_flip(img, label,depth):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth
def randomCrop(image, label,depth):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region)
def randomRotation(image,label,depth):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
    return image,label,depth
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)

        randY=random.randint(0,img.shape[1]-1)

        if random.randint(0,1)==0:

            img[randX,randY]=0

        else:

            img[randX,randY]=255
    return Image.fromarray(img)

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, caption_root, trainsize):
        self.caption_root = caption_root
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        # 'negative' means background class.
        # We don't distinguish between singular and plural forms, e.g. car and cars (the same class).
        self.target_dict = {'negative': 0, 'people': 1, 'man': 2, 'girl': 3, 'dress': 4, 'owl': 5, 'towel': 6,
                       'woman': 7, 'dog': 8, 'truck': 9, 'plant': 10, 'car': 11, 'chicken': 12, 'train': 13,
                       'paper': 14, 'statue': 15, 'bird': 16, 'vases': 17, 'vase': 18, 'bench': 19,
                       'horse': 20, 'sofa': 21, 'fish': 22, 'callboard': 23, 'book': 24, 'horses': 25,
                       'papers': 26, 'stone': 27, 'pigeon': 28, 'airplane': 29, 'pot': 30, 'butterfly': 31,
                       'sign': 32, 'butterflies': 33, 'lamp': 34, 'men': 35, 'cat': 36, 'zebra': 37, 'oranges': 38,
                       'poster': 39, 'motorcycle': 40, 'bowl': 41, 'bottle': 42, 'boat': 43, 'pots': 44, 'hydrant': 45,
                       'birds': 46, 'sticks': 47, 'fan': 48, 'building': 49, 'airplanes': 50, 'shoes': 51, 'boxes': 52,
                       'picture': 53, 'bus': 54, 'tree': 55, 'cars': 56, 'bear': 57, 'flower': 58, 'bag': 59,
                       'basketball': 60, 'cow': 61, 'duck': 62, 'bee': 63, 'fruit': 64, 'tortoise': 65, 'house': 66,
                       'door': 67, 'bike': 68, 'masks': 69, 'dinosaur': 70, 'bowls': 71, 'statues': 72, 'child': 73,
                       'phone': 74, 'flowers': 75, 'mask': 76, 'helicopter': 77, 'pillow': 78, 'bushes': 79,
                       'dresses': 80, 'box': 81, 'hat': 82, 'monkey': 83, 'piano': 84, 'plants': 85, 'chair': 86,
                       'dustbin': 87, 'leaf': 88, 'women': 89, 'clock': 90, 'books': 91, 'bush': 92, 'stick': 93,
                       'lion': 94, 'knife': 95, 'boy': 96, 'snails': 97, 'snail': 98, 'lanterns': 99, 'deer': 100,
                       'chairs': 101, 'computer': 102, 'cats': 103, 'signs': 104, 'wheel': 105, 'window': 106,
                       'lamps': 107, 'tower': 108, 'fans': 109, 'benches': 110, 'tv': 111, 'bikes': 112,
                       'giraffes': 113, 'machine': 114, 'boats': 115, 'cup': 116, 'apple': 117, 'leopard': 118,
                       'pictures': 119, 'suit': 120, 'tiger': 121, 'deers': 122, 'monitor': 123, 'bridge': 124,
                       'coral': 125, 'cows': 126, 'dogs': 127, 'cake': 128, 'sofas': 129, 'stones': 130, 'bottles': 131,
                       'telescope': 132, 'road': 133, 'pig': 134, 'crocodile': 135, 'panda': 136, 'toy': 137,
                       'plates': 138, 'hats': 139, 'lantern': 140, 'plate': 141, 'zebras': 142, 'boys': 143,
                       'girls': 144, 'handle': 145, 'tank': 146, 'corals': 147, 'footballs': 148, 'handles': 149}

        self.DF_cap = pd.DataFrame(columns=['Index','Image','Caption','Semantic',
                                            'Position','SubCategory','Category','Source'])
        DF_this = pd.read_csv(caption_root+'CapS.csv',encoding='gbk')
        DF_this.reset_index(drop=True)
        #DF_this = DF_this.drop('Unnamed: 0', 1)
        self.DF_cap = pd.concat([self.DF_cap, DF_this])


        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        name = self.images[index]
        name_index = name.split('/')[-1]
        Caption = self.DF_cap.loc[self.DF_cap['Image']==name_index, 'Caption'].values[0]
        Semantic_word = self.DF_cap.loc[self.DF_cap['Image']==name_index, 'Semantic'].values[0]
        Position = self.DF_cap.loc[self.DF_cap['Image']==name_index, 'Position'].values[0]

        '''word embedding'''
        max_length = 20
        Dict_vocab = torch.load(self.caption_root+'CapS.pt')
        words = [w for w in Caption.strip().split()]
        words_id = [Dict_vocab['word2id'][w] for w in words]
        sen_length = len(words_id)
        if sen_length > max_length:
            words_id = words_id[:max_length]
        else:
            supp_empty = max_length-sen_length
            words_id.extend([0]*supp_empty)     # id 0 means the unknown word, whose embedding is zero.
        # caption_feat_list = [Dict_vocab['id2vec'][id] for id in words_id]                    # Dim: 20 * 300
        # caption_feat = torch.stack(caption_feat_list,dim=0)
        '''Add start point'''
        caption_feat =  [Dict_vocab['id2vec'][0]]
        caption_feat.extend([Dict_vocab['id2vec'][id] for id in words_id])              # Dim: 1+20 * 300
        caption_feat = torch.stack(caption_feat,dim=0)

        semantic_feat = Dict_vocab['id2vec'][Dict_vocab['word2id'][Semantic_word]]      # Dim: 1 * 300

        idx = self.target_dict[Semantic_word]
        cls_target = torch.tensor(idx).long()
        neg_target = torch.tensor(0).long()

        Weakly_Info = [caption_feat, semantic_feat, Position, cls_target, neg_target]   # Dim: [21 * 300, 300, 1, 1, 1]


        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])
        img_ori = self.img_transform(image)
        gt_ori = self.gt_transform(gt)
        depth_ori = self.depths_transform(depth)
        #
        # data augment
        image, gt, depth = cv_random_flip(image, gt, depth)
        image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)
        #gt = randomPeper(gt)
        #
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        return image, gt, depth, [self.images[index],self.gts[index]], \
               [img_ori,gt_ori,depth_ori,caption_feat,Position], Weakly_Info

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w,h),Image.BILINEAR),gt.resize((w,h),Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, depth_root, caption, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, depth_root, caption, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt).unsqueeze(0)
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt,depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size




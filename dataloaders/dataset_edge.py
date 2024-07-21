import cv2
import h5py
import numpy as np
import torch
import os
import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


def pseudo_label_generator_cell(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    def rgb_to_gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    if 1 not in np.unique(seed) or 2 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 3] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        
        gray_data = rgb_to_gray(data)  # 将RGB图像转换为灰度图像
        sigma = 0.35
        gray_data = rescale_intensity(gray_data, in_range=(-sigma, 1 + sigma),
                                      out_range=(-1, 1))
        segmentation = random_walker(gray_data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label



class EdgeBaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label", edge_paras=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.edge_paras = edge_paras
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.sample_list = train_ids
        elif self.split == 'val':
            self.sample_list = test_ids
        
        print("total {} samples".format(len(self.sample_list)))
        
    
    def _get_fold_ids(self, fold):
        
        all_cases_set = list(range(1, 2632))
        fold1_testing_set = [i for i in range(1, 527)]
        fold1_training_set = [i for i in all_cases_set if i not in fold1_testing_set]
        
        fold2_testing_set = [i for i in range(527, 1053)]
        fold2_training_set = [i for i in all_cases_set if i not in fold2_testing_set]
        
        fold3_testing_set = [i for i in range(1053, 1579)]
        fold3_training_set = [i for i in all_cases_set if i not in fold3_testing_set]
        
        fold4_testing_set = [i for i in range(1579, 2105)]
        fold4_training_set = [i for i in all_cases_set if i not in fold4_testing_set]
        
        fold5_testing_set = [i for i in range(2105, 2632)]
        fold5_training_set = [i for i in all_cases_set if i not in fold5_testing_set]
        
        
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"
        
    def __len__(self):
        return len(self.sample_list)
        
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, str(case) + '.h5'), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        edge = h5f[self.edge_paras][:]
        sample = {'image': image, 'label': label, 'edge': edge}
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_cell(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label, 'edge': edge}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label, 'edge': edge}
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample
        
def random_rot_flip(image, label,edge):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    edge = np.flip(edge, axis=axis).copy()
    return image, label, edge


def random_rotate(image, label, edge, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    edge = ndimage.rotate(edge, angle, order=0,
                           reshape=False, mode="constant", cval=0)
    return image, label, edge

class EdgeRandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, edge = sample['image'], sample['label'], sample['edge']
        if random.random() > 0.5:
            image, label, edge = random_rot_flip(image, label, edge)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label, edge = random_rotate(image, label, edge, cval=4)
            else:
                image, label, edge = random_rotate(image, label, edge, cval=0)
        x, y, _ = image.shape
        zoom_factors = (self.output_size[0] / x, self.output_size[1] / y, 1)  # 不缩放通道维度
        image = zoom(image, zoom_factors, order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 假设标签是单通道的
        edge = zoom(edge, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # 将numpy数组转换为torch张量
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # 调整通道顺序为(C, H, W)
        label = torch.from_numpy(label.astype(np.uint8))
        edge = torch.from_numpy(edge.astype(np.uint8))
        # convert 255 to 1
        edge[edge == 255] = 1

        # 创建包含处理后图像和标签的字典
        sample = {'image': image, 'label': label, 'edge': edge}

        return sample
        
        
        
if __name__ == "__main__":
    from torchvision.transforms import transforms
    train_dataset = BaseDataSets(base_dir="data/SegPC_2021", split="train", edge_paras="30_40_0", transform=transforms.Compose([
        RandomGenerator([256, 256])
    ]), fold='fold1', sup_type='scribble')
    print(train_dataset[0])

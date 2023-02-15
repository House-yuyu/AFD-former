import os
import os.path
import random
import torch

import torch.utils.data as udata  #
import torchvision.transforms.functional as TF
import cv2
from natsort import natsorted


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['BMP', 'jpg', 'png', 'JPG', 'PNG', 'bmp'])


class Dataset_train(udata.Dataset):
    def __init__(self, rgb_dir):
        super(Dataset_train, self).__init__()

        noisy_files = natsorted(os.listdir(os.path.join(rgb_dir, 'distortion')))
        clean_files = natsorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.noisy_filenames = [os.path.join(rgb_dir, 'distortion',  x) for x in noisy_files if is_image_file(x)]
        self.clean_filenames = [os.path.join(rgb_dir, 'gt',          x) for x in clean_files if is_image_file(x)]

        self.sizex = len(self.clean_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index = index % self.sizex
        clean = TF.to_tensor(load_img(self.clean_filenames[index]))  # c,h,w [0-1]range,RGB
        noisy = TF.to_tensor(load_img(self.noisy_filenames[index]))

        # aug = random.randint(0, 8)
        #
        # # Data Augmentations
        # if aug == 1:
        #     inp_img = inp_img.flip(1)
        #     tar_img = tar_img.flip(1)
        # elif aug == 2:
        #     inp_img = inp_img.flip(2)
        #     tar_img = tar_img.flip(2)
        # elif aug == 3:
        #     inp_img = torch.rot90(inp_img, dims=(1, 2))
        #     tar_img = torch.rot90(tar_img, dims=(1, 2))
        # elif aug == 4:
        #     inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
        #     tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        # elif aug == 5:
        #     inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
        #     tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        # elif aug == 6:
        #     inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
        #     tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        # elif aug == 7:
        #     inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
        #     tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        return clean, noisy


class Dataset_val(udata.Dataset):
    def __init__(self, rgb_dir):
        super(Dataset_val, self).__init__()
        noisy_files = natsorted(os.listdir(os.path.join(rgb_dir, 'distortion')))
        clean_files = natsorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.noisy_filenames = [os.path.join(rgb_dir, 'distortion',  x) for x in noisy_files if is_image_file(x)]
        self.clean_filenames = [os.path.join(rgb_dir, 'gt',          x) for x in clean_files if is_image_file(x)]

        self.sizex = len(self.clean_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index = index % self.sizex
        clean = TF.to_tensor(load_img(self.clean_filenames[index]))
        noisy = TF.to_tensor(load_img(self.noisy_filenames[index]))

        return clean, noisy


class Dataset_test(udata.Dataset):
    def __init__(self, rgb_dir, noisy_level):
        super(Dataset_test, self).__init__()

        noisy_files = natsorted(os.listdir(os.path.join(rgb_dir, noisy_level)))
        clean_files = natsorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.noisy_filenames = [os.path.join(rgb_dir, noisy_level,  x) for x in noisy_files if is_image_file(x)]
        self.clean_filenames = [os.path.join(rgb_dir, 'gt',         x) for x in clean_files if is_image_file(x)]

        self.sizex = len(self.clean_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index = index % self.sizex
        clean = TF.to_tensor(load_img(self.clean_filenames[index]))
        noisy = TF.to_tensor(load_img(self.noisy_filenames[index]))

        return clean, noisy


if __name__ == '__main__':
    dataset_train = Dataset_train('../data/patch_train')
    test = dataset_train.__getitem__(2)
    print(test)
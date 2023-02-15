from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--gt_dir', default='../../data/SIDD_gt', type=str, help='Directory for gt images')
parser.add_argument('--input_dir', default='../../data/SIDD_nosiy', type=str, help='Directory for gt images')
parser.add_argument('--patch_dir', default='../../data/patch_SIDD_train', type=str, help='Directory for image patches')

parser.add_argument('--patchsize', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches_B', default=300, type=int, help='Number of patches per image')

parser.add_argument('--num_patches_S', default=40, type=int, help='Number of patches per image')

parser.add_argument('--num_cores', default=16, type=int, help='Number of CPU Cores')
args = parser.parse_args()

PS        = args.patchsize
NUM_CORES = args.num_cores
NUM_PATCHES_B = args.num_patches_B
NUM_PATCHES_S = args.num_patches_S

noisy_patchDir = os.path.join(args.patch_dir, 'distortion')
clean_patchDir = os.path.join(args.patch_dir, 'gt')


#get sorted folders
files_gt = natsorted(glob(os.path.join(args.gt_dir,  '*.PNG')))  # natsorted会比sorted排序更自然,赋值给了files才生效
files_no = natsorted(glob(os.path.join(args.input_dir, '*.PNG')))

noisy_files, clean_files = [], []
for m in files_gt:
    clean_files.append(m)
for n in files_no:
    noisy_files.append(n)


def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def save_files(index):
    clean_file = clean_files[index]
    clean_img = cv2.imread(clean_file)
    noisy_file = noisy_files[index]
    noisy_img = cv2.imread(noisy_file)

    H = clean_img.shape[0]
    W = clean_img.shape[1]

    for j in range(NUM_PATCHES_B):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]
        # 做一次增广
        idx = np.random.randint(0, 8)
        x_aug = data_aug(noisy_patch, idx)
        y_aug = data_aug(clean_patch, idx)

        cv2.imwrite(os.path.join(noisy_patchDir, 'B_{}_{}_{}.png'.format(index + 1, j + 1, idx)), x_aug)
        cv2.imwrite(os.path.join(clean_patchDir, 'B_{}_{}_{}.png'.format(index + 1, j + 1, idx)), y_aug)



Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(clean_files))))

files_length = natsorted(glob(os.path.join('/home/zhanghuang_701/zx/Restormer-test/data/patch_SIDD_train/distortion',  '*.png')))
print(len(files_length))

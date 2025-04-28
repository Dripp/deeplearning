# data.py
import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import torch
from torchvision import transforms


class AugData:
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        rot = random.randint(0, 3)
        data = np.rot90(data, rot, axes=[1, 2]).copy()
        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()
        new_sample = {'data': data, 'label': label}
        return new_sample


class ToTensor:
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        # 归一化：将像素值从 [0, 255] 映射到 [0, 1]
        data = data / 255.0
        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }
        return new_sample


class MyDataset(Dataset):
    def __init__(self, dataset_dir, mode, transform=None):
        random.seed(1234)
        self.transform = transform
        self.mode = mode

        if mode == 'train':
            self.cover_dir = os.path.join(dataset_dir, 'cover_train')
            self.stego_dir = os.path.join(dataset_dir, 'stego_train')
        elif mode == 'val':
            self.cover_dir = os.path.join(dataset_dir, 'cover_val')
            self.stego_dir = os.path.join(dataset_dir, 'stego_val')
        elif mode == 'test':
            self.cover_dir = os.path.join(dataset_dir, 'cover_test')
            self.stego_dir = os.path.join(dataset_dir, 'stego_test')
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")

        self.cover_list = [x.split(os.sep)[-1] for x in glob(os.path.join(self.cover_dir, '*.pgm'))]
        self.stego_list = [x.split(os.sep)[-1] for x in glob(os.path.join(self.stego_dir, '*.pgm'))]

        print(f"Mode: {self.mode}")
        print(f"Cover directory: {self.cover_dir}, number of files: {len(self.cover_list)}")
        print(f"First 5 cover files: {self.cover_list[:5]}")
        print(f"Stego directory: {self.stego_dir}, number of files: {len(self.stego_list)}")
        print(f"First 5 stego files: {self.stego_list[:5]}")

        assert len(self.cover_list) == len(self.stego_list), \
            f"Cover ({len(self.cover_list)}) and Stego ({len(self.stego_list)}) file counts do not match in {self.mode} dataset"

        sorted_cover = sorted(self.cover_list, key=str.lower)
        sorted_stego = sorted(self.stego_list, key=str.lower)
        if sorted_cover != sorted_stego:
            print("Mismatched files:")
            cover_only = set(sorted_cover) - set(sorted_stego)
            stego_only = set(sorted_stego) - set(sorted_cover)
            if cover_only:
                print(f"Files in cover but not in stego: {cover_only}")
            if stego_only:
                print(f"Files in stego but not in cover: {stego_only}")
            raise AssertionError(f"Cover and Stego file names do not match in {self.mode} dataset")

        self.cover_list = [f.lower() for f in self.cover_list]
        random.shuffle(self.cover_list)
        expected_size = 4000 if mode == 'train' else 500
        assert len(self.cover_list) == expected_size, \
            f"Expected {expected_size} images in {self.cover_dir}, but found {len(self.cover_list)}"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        file_index = int(idx)
        cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[file_index])
        cover_data = cv2.imread(cover_path, -1)
        stego_data = cv2.imread(stego_path, -1)
        if cover_data is None or stego_data is None:
            raise ValueError(f"Failed to load images: {cover_path} or {stego_path}")
        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
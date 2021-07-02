"""
    Data loader and utilities for PatchCamelyon dataset
    For more information, please refer to https://patchcamelyon.grand-challenge.org/
"""
import h5py
from PIL import Image
from torch.utils.data import Dataset


class PatchCamelyonIter(Dataset):
    def __init__(self, img_path, label_path, preprocess):
        self.preprocess = preprocess
        self.img_path = img_path
        self.label_path = label_path
        self.data_size = self.load_data_size()

    def load_data_size(self):
        with h5py.File(self.label_path, 'r') as file_reader:
            label_list = list(file_reader['y'])
        data_size = len(label_list)

        return data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        with h5py.File(self.img_path, 'r') as img_list, h5py.File(self.label_path, 'r') as label_list:
            _img = img_list['x'][idx]
            _img = Image.fromarray(_img)
            _img = self.preprocess(_img)
            _lbl = label_list['y'][idx].item()
        return _img, _lbl


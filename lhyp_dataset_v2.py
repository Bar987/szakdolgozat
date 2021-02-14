from torch.utils.data import Dataset
import torch
import numpy as np
import skimage
from build_dataset_v2 import draw_images

class LhypDataset(Dataset):

    def __init__(self, data, transform=None, noise=False, isRNN = True):
        self.data = data
        self.transform = transform
        self.noise = noise
        self.isRNN = isRNN

    def __len__(self):
        if self.isRNN:
            return len(self.data)
        else:
            return len(self.data)*len(self.data[0]['img'])

    def __getitem__(self, idx):

        if self.isRNN:
            sample = self.data[idx]['img']

            if self.transform:
                new_sample = np.zeros((len(sample), 3, 128, 128))
                for i in range(len(sample)):
                    if self.noise:
                        new_sample[i] = skimage.util.random_noise(self.transform(sample[i]), mode='gaussian', seed=42, clip=True)
                    else:
                        new_sample[i] = self.transform(sample[i])
                return (new_sample.astype(np.float32), torch.tensor(self.data[idx]['label']), self.data[idx]['filename'])

            return (sample, torch.tensor(self.data[idx]['label']), self.data[idx]['filename'])
        else:
            patient_idx = idx // len(self.data[0]['img'])
            img_idx = idx % len(self.data[0]['img'])
            sample = self.data[patient_idx]['img'][img_idx]

            if self.transform:
                if self.noise:
                    new_sample = skimage.util.random_noise(self.transform(sample), mode='gaussian', seed=42, clip=True)
                else:
                    new_sample = self.transform(sample)
                return (new_sample, torch.tensor(self.data[patient_idx]['label']), self.data[patient_idx]['filename'])

            return (sample, torch.tensor(self.data[patient_idx]['label']), self.data[patient_idx]['filename'])
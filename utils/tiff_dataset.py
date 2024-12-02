import torch
import torchvision.transforms as transforms
import numpy as np
import csv

from torch.utils.data import Dataset
from skimage import io
from pathlib import Path
from typing import Optional, Tuple

# Read the CSV and return a list of paths (ignoring the first row)
def get_paths_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        paths = [Path(row[0]) for row in reader]
    return paths


class TiffDataset(Dataset):

    def __init__(self, input_dir: Path, label_classes: Optional[np.ndarray] = None):
        self.label_classes = label_classes
        self.list_img = get_paths_from_csv(input_dir / "images.csv")[:50]
        self.list_label = get_paths_from_csv(input_dir / "labels.csv")[:50]

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx) -> dict:
        img_raw_path = self.list_img[idx]
        label_path = self.list_label[idx]

        image = self.preprocess_image(io.imread(img_raw_path))
        labels = self.label_to_classes(io.imread(label_path))

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(labels.copy()).long().contiguous()
        }

    def label_to_classes(self, labeled_image):
        output = np.zeros((labeled_image.shape[0], labeled_image.shape[1], self.label_classes.shape[0]))

        for c, label_class in enumerate(self.label_classes):
            label = np.nanmin(label_class == labeled_image, axis=2)
            output[:, :, c] = label

        return output.argmax(axis=2)

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        return image.transpose(2, 0, 1) / 255.0


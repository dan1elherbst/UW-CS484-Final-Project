import os
import re
from typing import Union

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class KITTIDataset(Dataset):
    """
    Class that represents (a chunk of) the KITTI dataset for unsupervised/semi-supervised
    monocular depth estimation that can be used by PyTorch. In training mode, two stereo 
    images are provided, while for tests, only one image (the left one) is available.
    Images are loaded from a directory that should have the default folder structure from 
    the official KITTI data source (or installed via the provided downloading script 
    'download_data.sh').
    """

    def __init__(
        self, 
        root: str = 'data', 
        mode: str = 'train',
        width: int = 1242,
        height: int = 375,
        augment: bool = True,
        prob_flip: float = 0.5,
        prob_color_augment: float = 0.5,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root (str): 
                Root path of the KITTI data set.
            mode: 
                (['train'|'test']): Training or test mode.
            width (int): 
                Width the images should have. If provided, all images will be transformed 
                to the same specified dimension.
            height (int): 
                Height of the images.
            augment (bool): 
                Whether or not to perform data augmentation for training.
            prob_flip (float): 
                Probability of a horizontal flip, in which the left and right image are 
                also switched.
            prob_color_augment (float): 
                Probability of performing a range of color augmentations, adjusting the 
                brightness, contrast, saturation, and hue of the images.
        """
        super().__init__()
        self.root = root
        self.mode = mode
        self.width, self.height = width, height
        self.augment = augment
        self.prob_flip = prob_flip
        self.prob_color_augment = prob_color_augment

        # Create sorted list of paths for left images
        self.img_left_paths = sorted([
            os.path.join(path, name) 
            for path, _, files in os.walk(root) 
            for name in files 
            if re.match(r'.*/image_02/data/.*\.png', os.path.join(path, name))
        ])

        if mode == 'train':

            # Create sorted list of paths for right images
            self.img_right_paths = sorted([
                os.path.join(path, name) 
                for path, _, files in os.walk(root) 
                for name in files 
                if re.match(r'.*/image_03/data/.*\.png', os.path.join(path, name))
            ])
            assert len(self.img_right_paths) == len(self.img_left_paths)

    def __len__(self) -> int:
        """
        Return number of samples in the dataset.
        """
        return len(self.img_left_paths)

    def __getitem__(
        self, 
        idx: int,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        Load a specific image or pair of images.
        """

        # Load left image into torch tensor; transform to specified size
        img_left = self._to_tensor(Image.open(self.img_left_paths[idx]))

        if self.mode == 'test':
            return img_left

        elif self.mode == 'train':

            # Load right image into torch tensor; transform to specified size
            img_right = self._to_tensor(Image.open(self.img_right_paths[idx]))

            # Perform data augmentation
            if self.augment:
                img_left, img_right = self._random_flip(img_left, img_right)
                img_left, img_right = self._color_augment(img_left, img_right)

            return img_left, img_right

    def _random_flip(
        self, 
        img_left: Tensor, 
        img_right: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a random horizontal flip of an image pair.
        """
        if torch.rand(1) < self.prob_flip:
            return torch.flip(img_right, (2,)), torch.flip(img_left, (2,))
        else:
            return img_left, img_right

    def _color_augment(
        self, 
        img_left: Tensor, 
        img_right: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform random color augmentation for an image pair. This may deviate slightly 
        from the color augmentation used in the original paper. 
        """
        if torch.rand(1) < self.prob_color_augment:
            color_jitter = transforms.ColorJitter(
                brightness=(0.5, 2.0),
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
            )
            joint_transformed = color_jitter(torch.concat((img_left, img_right), dim=2))
            joint_W = joint_transformed.shape[2]
            return (joint_transformed[:, :, :joint_W//2], 
                    joint_transformed[:, :, -joint_W//2:])
        else:
            return img_left, img_right

    def _to_tensor(self, img: Image.Image) -> Tensor:
        """
        Transform a PIL image to a torch tensor and crop/resize it to the desired size 
        [self.width, self.height].
        """
        if img.height / img.width >= self.height / self.width:
            img.thumbnail((self.width, img.height))
        else:
            img.thumbnail((img.width, self.height))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((self.height, self.width)),
        ])
        return transform(img)

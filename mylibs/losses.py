import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.functional import resize


class MonoDepthLoss:
    """
    Implementation of the losses for self-supervised monocular depth estimation intruduced
    by Clément Godard, Oisin Mac Aodha, Gabriel J. Brostow in the paper

    "Unsupervised Monocular Depth Estimation with Left-Right Consistency" 
    (https://arxiv.org/pdf/1609.03677.pdf).

    Some parts of the implementation are inspired by the reference implementation of the 
    original paper, available at https://github.com/mrharicot/monodepth, which is based 
    on TensorFlow while this project relies on PyTorch.
    """

    def __init__(
        self, 
        device: str,
        alpha_ap: float = 1,
        alpha_ds_std: float = 0.1,
        alpha_ds_scales: tuple[float, float, float, float] = (1, 1/2, 1/4, 1/8),
        alpha_lr: float = 1,
    ) -> None:
        """
        Initialize the loss object and set parameters of the loss function that will 
        remain constant during the training procedure.

        Args:
            device (str):
                Device on which to compute the loss. Input tensors have to be on the same 
                device.
            alpha_ap (float):
                Weight for the appearance matching part of the loss function.
            alpha_ds_std (float):
                Weight for the standard disparity smoothness part of the loss function.
            alpha_ds_scales (tuple of floats):
                Multiplicative factors for the disparity smoothness loss to apply to each 
                of the 4 scales of the disparity map outputs.
            alpha_lr (float):
                Weight for the left-right disparity consistency part of the loss function.
        """
        self.device = device
        self.alpha_ap = alpha_ap
        self.alpha_ds_std = alpha_ds_std
        self.alpha_ds_scales = alpha_ds_scales
        self.alpha_lr = alpha_lr

    def total_loss(
        self,
        img_left: Tensor,
        img_right: Tensor,
        disp_scales: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> Tensor:
        """
        Compute the total loss of the model across all scales for a batch of input images.
        Total loss is the mean of the loss for the individual images. 

        Args:
            img_left (tensor [N, C, H, W]):
                Left input images.
            img_right (tensor [N, C, H, W]):
                Right input images.
            disp_scales (tuple of tensors):
                Previous model output consisting of the downscaled disparity maps. 
        
        Returns:
            tensor [1]:
                Total loss of the model for the given input images 'img_left' and 
                'img_right'.
        """
        loss = 0
        for img_left_scale,\
            img_right_scale,\
            disp,\
            alpha_ds_scale\
        in zip(self.image_scales(img_left), 
            self.image_scales(img_right),
            disp_scales, 
            self.alpha_ds_scales):
            loss += self.total_loss_scaled(img_left_scale, 
                                           img_right_scale, 
                                           disp,
                                           self.alpha_ds_std * alpha_ds_scale)
        return loss

    def total_loss_scaled(
        self,
        img_left: Tensor,
        img_right: Tensor,
        disp: Tensor,
        alpha_ds: float,
    ) -> Tensor:
        """
        Compute total loss for one scaled version of the images and disparity maps.

        Args:
            img_left (tensor [N, C, H, W]):
                Scaled left input images.
            img_right (tensor [N, C, H, W]):
                Scaled right input images.
            disp (tensor [N, 2, H, W]):
                Right and left disparity maps for the input images 'img_left' and 
                'img_right'.
        
        Returns:
            tensor [1]:
                Total loss of the model for the given scaled input images 'img_left' and 
                'img_right'.
        """
        disp_right, disp_left = disp[:, 0, :, :], disp[:, 1, :, :]

        # Appearance matching losses
        loss_ap_left = self.appearance_matching_loss(img_left, 
                                                     img_right, 
                                                     disp_left, 
                                                     'left')
        loss_ap_right = self.appearance_matching_loss(img_right, 
                                                      img_left, 
                                                      disp_right, 
                                                      'right')

        # Disparity smoothness losses
        loss_ds_left = self.disparity_smoothness_loss(img_left, disp_left)
        loss_ds_right = self.disparity_smoothness_loss(img_right, disp_right)

        # Disparity consistency loss
        loss_lr = self.disparity_consistency_loss(disp_left, disp_right)

        # return weighted sum of all losses
        loss = self.alpha_ap * (loss_ap_left + loss_ap_right) + \
               alpha_ds * (loss_ds_left + loss_ds_right) + \
               self.alpha_lr * loss_lr
        return loss

    def appearance_matching_loss(
        self,
        img: Tensor,
        other_img: Tensor,
        disp: Tensor, 
        disp_mode: str,
        alpha: float = 0.85,
    ) -> Tensor:
        """
        Compute appearance matching loss which is a weighted average of the SSIM loss and 
        normal L1 loss between a batch of original images and their recontructed
        approximations.

        Args:
            img (tensor [N, C, H, W]):
                Original images for which the appearance matching loss should be 
                calculated.
            other_img (tensor [N, C, H, W]):
                Other images which are used in combination with the disparity map in order
                to reconstruct the original images.
            disp (tensor [N, H, W]):
                Disparity map which, applied to 'other_img', aims at reconstructing 'img'.
            disp_mode (['right'|'left']):
                Whether to reconstruct the right image or the left image.
            alpha (float between 0 and 1):
                Weight of the SSIM part of the loss.
        
        Returns:
            tensor [1]:
                Appearance matching loss averaged over the batch.
        """
        reconstructed_img = self.generate_images(other_img, disp, disp_mode)
        SSIM_loss = torch.mean(self.SSIM(img, reconstructed_img))
        L1_loss = torch.mean(torch.linalg.norm(img - reconstructed_img, ord=1, dim=1))
        return alpha * SSIM_loss + (1 - alpha) * L1_loss

    def disparity_smoothness_loss(
        self,
        img: Tensor,
        disp: Tensor,
    ) -> Tensor:
        """
        Compute disparity smoothness loss for a batch of images and their related
        estimated disparity maps. 

        Args: 
            img (tensor [N, C, H, W]): 
                Images.
            disp (tensor [N, H, W]):
                Disparity maps which align with the images.
        
        Returns:
            tensor [1]:
                Disparity smoothness loss averaged over the batch of images that was 
                provided. 
        """
        norm_grad_x = torch.linalg.norm(self.d_x(img), ord=1, dim=1)
        norm_grad_y = torch.linalg.norm(self.d_y(img), ord=1, dim=1)
        disp_grad_x, disp_grad_y = self.d_x(disp), self.d_y(disp)
        mean_x = torch.mean(torch.abs(disp_grad_x) * torch.exp(-norm_grad_x))
        mean_y = torch.mean(torch.abs(disp_grad_y) * torch.exp(-norm_grad_y))
        return mean_x + mean_y

    def disparity_consistency_loss(
        self,
        disp_left: Tensor,
        disp_right: Tensor,
    ) -> Tensor:
        """
        Calculate the disparity consistency part of the total loss.

        Args:
            disp_left (tensor [N, H, W]):
                Left disparity maps.
            disp_right (tensor [N, H, W]):
                Right disparity maps.
        
        Returns:
            tensor [1]:
                Disparity consistency loss averaged over the input batch.
        """
        disp_left_to_right = self.generate_images(disp_left[:, None, :, :], 
                                            disp_right, 
                                            disp_mode='right')\
                            .squeeze(1)
        disp_right_to_left = self.generate_images(disp_right[:, None, :, :],
                                            disp_left,
                                            disp_mode='left')\
                            .squeeze(1)
        loss_left = torch.mean(torch.abs(disp_right - disp_left_to_right))
        loss_right = torch.mean(torch.abs(disp_left - disp_right_to_left))
        return loss_left + loss_right

    def generate_images(
        self,
        img: Tensor, 
        disp: Tensor,
        disp_mode: str,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: bool = False,
    ) -> Tensor:
        """
        Apply disparity maps to multiple images.

        Args:
            img (tensor [N, C, H, W]):
                Torch tensor storing all images to which disparity maps should be applied.
            disp (tensor [N, H, W]):
                Torch tensor storing disparity maps for all images (disparity in direction 
                of x).
            disp_mode (['right'|'left']):
                Whether to generate the right image or the left image.
            mode (str):
                Parameter passed to F.grid_sample.
            padding_mode (str):
                Parameter passed to F.grid_sample.
            align_corners (bool):
                Parameter passed to F.grid_sample.
        
        Returns:
            tensor [N, C, H, W]:
                Reconstructed images.
        """
        N, _, H, W = img.shape
        _, meshgrid_H, meshgrid_W = torch.meshgrid(torch.linspace(0, N, N),
                                                torch.linspace(-1, 1, H),
                                                torch.linspace(-1, 1, W),
                                                indexing='ij')
        meshgrid_H, meshgrid_W = meshgrid_H.to(self.device), meshgrid_W.to(self.device)
        if disp_mode == 'right':
            meshgrid_W = meshgrid_W + 2 * disp
        elif disp_mode == 'left':
            meshgrid_W = meshgrid_W - 2 * disp
        grid = torch.concat((meshgrid_W[:, :, :, None], meshgrid_H[:, :, :, None]), 
                            dim=3)
        return F.grid_sample(img, 
                            grid, 
                            mode=mode, 
                            padding_mode=padding_mode, 
                            align_corners=align_corners)

    def SSIM(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute Structural Similarity Index (SSIM) between two images. Code mainly taken 
        from SSIM function in the reference implementation in TensorFlow and adapted to 
        PyTorch:
        https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py

        Args:
            x (tensor [N, C, H, W]):
                First input image.
            y (tensor [N, C, H, W]):
                Second input image.

        Returns:
            tensor [N, H-1, W-1]:
                Pixel-wise SSIM.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x  = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y  = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y , 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, min=0, max=1)

    def d_x(self, img: Tensor) -> Tensor:
        """
        Compute image gradients in direction of x.

        Args:
            img (tensor [N, C, H, W] or [N, H, W]):
                Input images.

        Returns:
            tensor [N, C, H, W-1] or [N, H, W-1]:
                Image gradients in direction of x. 
        """
        if len(img.shape) == 4:
            return img[:, :, :, 1:] - img[:, :, :, :-1]
        elif len(img.shape) == 3:
            return img[:, :, 1:] - img[:, :, :-1]

    def d_y(self, img: Tensor) -> Tensor:
        """
        Compute image gradients in direction of y.

        Args:
            img (tensor [N, C, H, W] or [N, H, W]):
                Input images.

        Returns:
            tensor [N, C, H-1, W] or [N, H-1, W]:
                Image gradients in direction of y. 
        """
        if len(img.shape) == 4:
            return img[:, :, 1:, :] - img[:, :, :-1, :]
        elif len(img.shape) == 3:
            return img[:, 1:, :] - img[:, :-1, :]

    def image_scales(
        self, 
        img: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns 4 scaled version of the input images, dividing height and width by 1, 2, 
        4, and 8.

        Args:
            img (tensor [N, C, H, W]):
                Input images.
        
        Returns:
            tuple of tensors:
                Scaled versions of the images.
        """
        H, W = img.shape[2], img.shape[3]
        return img,\
            resize(img, (H//2, W//2)),\
            resize(img, (H//4, W//4)),\
            resize(img, (H//8, W//8))

import torch
from torch import nn

from .operators import LenslessCamera
from .unet import UNet


class PrimalBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()

        w = width

        p = w + w
        self.primal_conv = torch.nn.Sequential(*[
            nn.Conv2d(p, p * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(p * 2, w, 1),
        ])


    def forward(self, inputs):
        camera, primal, dual, image = inputs

        dual_fwd = color_pack(camera.forward(color_unpack(primal)))
        dual = image - dual_fwd

        primal_adj = color_pack(camera.adjoint(color_unpack(dual)))
        primal_cat = torch.cat([primal_adj, primal], dim=1)
        primal = primal + self.primal_conv(primal_cat)

        return primal, dual


class PrimalDualBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()

        w = width

        d = w + w + 1
        self.dual_conv = torch.nn.Sequential(*[
            nn.Conv2d(d, d * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d * 2, w, 1),
        ])

        p = w + w
        self.primal_conv = torch.nn.Sequential(*[
            nn.Conv2d(p, p * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(p * 2, w, 1),
        ])
        
    def forward(self, inputs):
        camera, primal, dual, image = inputs

        dual_fwd = color_pack(camera.forward(color_unpack(primal)))
        dual_cat = torch.cat([dual_fwd, dual, image], dim=1)
        dual = dual + self.dual_conv(dual_cat)

        primal_adj = color_pack(camera.adjoint(color_unpack(dual)))
        primal_cat = torch.cat([primal_adj, primal], dim=1)
        primal = primal + self.primal_conv(primal_cat)

        return primal, dual


class ImageOptimizer(nn.Module):

    def __init__(self, psf, width=5, depth=10, learned_models=1, primal_only=False):
        super().__init__()

        self.shape = psf.shape
        self.depth = depth
        self.width = width

        psfs = torch.tile(psf.unsqueeze(1), (1, 1, learned_models or 1, 1, 1))
        self.psfs = torch.nn.Parameter(psfs, requires_grad=learned_models > 0)

        block = PrimalBlock if primal_only else PrimalDualBlock

        # Just one of these is enough
        self.layers = torch.nn.Sequential(*[
            block(self.width)
            for i in range(0, self.depth)
        ])

        self.unet = UNet(
            in_channels=width,
            out_channels=1,
            init_features=width * 2,
            output_padding=[(1, 0), (1, 0), (1, 0), (0, 0)],  # diffusercam full
        )

    def model_images(self):
        b, c, v, h, w = self.psfs.shape
        return {
            "PSFs": self.psfs.reshape(b * v, c, h, w),
        }

    def forward(self, image, denoise=True, depth=None, kernel=0):
        b, c, h, w = image.shape
        
        v = self.width
        k = kernel

        # Reload camera on each epoch
        camera = LenslessCamera(self.psfs)
        image  = image.reshape(b * c, 1, h, w)
        dual   = torch.zeros(b * c, v, h, w, device=image.device)
        primal = torch.zeros(b * c, v, h * 2, w * 2, device=image.device)

        for i in range(0, depth or self.depth):
            primal, dual = self.layers[i]((camera, primal, dual, image))

        output = camera.crop(primal)

        if denoise:
            output = output[:, [k]] + self.unet(output)
        else:
            output = output[:, [k]]

        output = output.reshape((b, c, h, w))
        output = torch.sigmoid(output)
        return output


def color_unpack(x):
    """
    Unpack colors from the batch channel
    """
    c = 3
    bc, k, h, w = x.shape
    return x.reshape((bc // c, c, k, h, w))


def color_pack(x):
    """
    Hide colors in batch channel
    """
    b, c, k, h, w = x.shape
    assert c == 3
    return x.reshape((b * c, k, h, w))

import torch
from torch import nn

from .operators import LenslessCamera
from .unet import UNet


class PrimalDualBlock(torch.nn.Module):
    def __init__(self, kernels):
        super(PrimalDualBlock, self).__init__()

        k = kernels
        w = 3 * k

        d = 3 * w
        self.dual_conv = torch.nn.Sequential(*[
            nn.Conv2d(d, d, 3, groups=k, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, w, 1),
        ])

        p = 2 * w
        self.primal_conv = torch.nn.Sequential(*[
            nn.Conv2d(p, p, 3, groups=k, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(p, w, 1),
        ])

    def forward(self, inputs):
        camera, primal, dual, image = inputs

        dual_fwd = camera.forward(primal)
        dual_cat = torch.cat([dual_fwd, dual, image], dim=1)
        dual = dual + self.dual_conv(dual_cat)

        primal_adj = camera.adjoint(dual)
        primal_cat = torch.cat([primal_adj, primal], dim=1)
        primal = primal + self.primal_conv(primal_cat)

        return primal, dual


class ImageOptimizerMixColors(torch.nn.Module):

    def __init__(self, psf, depth=10):
        super().__init__()

        self.shape = psf.shape
        self.depth = depth
        self.kernels = k = 5

        # bckhw
        psfs = psf.tile((1, k, 1, 1))
        self.psfs = torch.nn.Parameter(psfs, requires_grad=True)

        # Every story has a beginning, a middle, and an end
        self.layers = torch.nn.Sequential(*[
            PrimalDualBlock(self.kernels)
            for i in range(0, self.depth)
        ])

        self.unet = UNet(
            in_channels=3 * self.kernels,
            out_channels=3,
            init_features=3 * self.kernels,
            # output_padding=[(1, 0), (0, 0), (0, 0), (0, 0)],  # olicam
            output_padding=[(1, 0), (1, 0), (1, 0), (0, 0)],  # diffusercam
            # output_padding=[(0, 0), (1, 0), (1, 0), (1, 0)],
        )

    #@property
    #def psfs(self):
    #    k = 1  # self.kernels
    #    b, ck, h, w = self.psfs.shape
    #    return self.psfs.reshape(b * k, ck // k, h, w)

    def forward(self, image, depth=None, denoise=True):
        b, c, h, w = image.shape

        k = self.kernels

        # a "Field of Experts"
        camera = LenslessCamera(self.psfs)

        # Initialize primal, dual in bckhw
        image  = image.tile((1, k, 1, 1))
        dual   = torch.zeros(b, c * k, h, w, device=image.device)
        primal = torch.zeros(b, c * k, h * 2, w * 2, device=image.device)

        # Iterative layers
        for i in range(0, depth or self.depth):
            primal, dual = self.layers[i]((camera, primal, dual, image))

        # Back down to normal size
        output = camera.crop(primal)

        # Optionally denoise
        if denoise:
            output = output[:, [0, 1, 2]] + self.unet(output)
        else:
            output = output[:, [0, 1, 2]]

        return torch.sigmoid(output)

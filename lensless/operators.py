from torch import fft
import torch.nn.functional as F


def LenslessCamera(psfs):
    """
    Lensless Camera with optionally separable forward and adjoint models
    """
    sensor = Crop(psfs.shape)
    diffuser = Convolution(sensor.pad(psfs))
    return LenslessOperator(sensor, diffuser)


class LenslessOperator:
    def __init__(self, sensor, diffuser):
        """
        Lensless imaging forward and adjoint models.
        Expects images in (B, C, H, W) format.
        """        
        self.crop  = sensor.forward
        self.pad   = sensor.adjoint
        self.sensor = sensor

        self.convolve = diffuser.forward
        self.cross_correlate = diffuser.adjoint
        self.autocorrelation = diffuser.autocorrelation

    def forward(self, image):
        output = image
        output = self.convolve(output)
        output = self.crop(output)
        return output

    def adjoint(self, image):
        output = image
        output = self.pad(output)
        output = self.cross_correlate(output)
        return output


class Crop:
    def __init__(self, shape):
        """
        Just pads and crops.
        """
        size = shape[-2:]
        pad_size = [s * 2 for s in size]

        ys, yr = divmod(pad_size[0] - size[0], 2)
        xs, xr = divmod(pad_size[1] - size[1], 2)
        ye = ys + size[0]
        xe = xs + size[1]

        self.size     = tuple(size)
        self.pad_size = tuple(pad_size)
        self.y_center = slice(ys, ye)
        self.x_center = slice(xs, xe)
        self.pad_args = xs, xs + xr, ys, ys + yr

        self.crop = self.forward
        self.pad  = self.adjoint

    def forward(self, image):
        if image.shape[-2:] != self.pad_size:
            raise ValueError(
                f"image shape {image.shape} "
                f"does not match padded sensor size {self.pad_size}"
            )
        return image[..., self.y_center, self.x_center]

    def adjoint(self, image):
        if image.shape[-2:] != self.size:
            raise ValueError(
                f"image shape {image.shape} "
                f"does not match sensor size {self.size}"
            )
        return F.pad(image, self.pad_args)


class Convolution:
    def __init__(self, padded_psf):
        """
        Multiplication in fourier domain.
        """
        self.h = ft(padded_psf)

    def forward(self, image):
        return ift(self.h * ft(image))

    def adjoint(self, image):
        return ift(self.h.conj() * ft(image))

    def autocorrelation(self):
        return (self.h * self.h.conj()).real


def ft(image):
    return fft.rfft2(fft.ifftshift(image, dim=(-2, -1)), norm='ortho')
    

def ift(image):
    return fft.fftshift(fft.irfft2(image, norm='ortho'), dim=(-2, -1))

import torch
import torchvision
import torch.fft as fft


class Regularizer:
    def __init__(self, reg):
        self.h = ft(pad(reg))
        
    def autocorrelation(self):
        return self.h * self.h.conj()

    def forward(self, scene):
        return ift(self.h * ft(scene))


class DiffuserCam:
    def __init__(self, psf):
        self.h = ft(pad(psf))
        
    def autocorrelation(self):
        return self.h * self.h.conj()

    def forward(self, scene):
        return crop(ift(self.h * ft(scene)))

    def adjoint(self, error):
        return ift(self.h.conj() * ft(pad(error)))


class WienerCam:
    def __init__(self, psf, regularization):
        self.h = ft(pad(psf))
        self.hth = 1.0 / (self.h * self.h.conj() + regularization)

    def autocorrelation(self):
        return self.h * self.h.conj()

    def forward(self, scene):
        return ift(self.h * ft(scene))

    def solve(self, measurement):
        return crop(ift(self.hth * self.h.conj() * ft(measurement)))


def crop(x):
    h, w = x.shape[-2:]
    return torchvision.transforms.functional.center_crop(x, (h // 2, w // 2))


def pad(x, mode='constant', fill=0):
    h, w = x.shape[-2:]
    ys, yr = divmod(h, 2)
    xs, xr = divmod(w, 2)
    return torchvision.transforms.functional.pad(x, (xs, ys, xs + xr, ys + yr), padding_mode=mode, fill=fill)


def ft(image):
    return fft.rfft2(fft.ifftshift(image), norm='ortho')


def ift(image):
    return fft.fftshift(fft.irfft2(image, norm='ortho'))


def cft(image):
    return fft.fft2(fft.ifftshift(image))


def icft(image):
    return fft.fftshift(fft.ifft2(image))

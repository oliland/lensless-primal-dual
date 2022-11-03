import torch
import torch.fft as fft
from tqdm import trange


class TotalVariation:

    def __init__(self, camera):
        self.pad_size = camera.sensor.pad_size

    def forward(self, image):
        diff1 = torch.roll(image, 1, dims=-1) - image
        diff2 = torch.roll(image, 1, dims=-2) - image
        return torch.stack([diff1, diff2])

    def adjoint(self, image):
        diff1 = torch.roll(image[0], -1, dims=-1) - image[0]
        diff2 = torch.roll(image[1], -1, dims=-2) - image[1]
        return diff1 + diff2
    
    def gramian_fourier(self):
        output = torch.zeros(self.pad_size)
        output[0,0] = 4
        output[0,1] = output[1,0] = output[0,-1] = output[-1,0] = -1
        output = fft.rfft2(output)
        return output


def reconstruct_gd(camera, image, iters=10000, tol=0, tqdm=True):
    """
    Reconstruct an image with good old gradient descent
    """
    output = torch.zeros_like(camera.pad(image))
    alpha = 1.8 / torch.max(camera.autocorrelation())

    prev_err = 1e6
    for i in trange(0, iters, disable=not tqdm):
        forward  = camera.forward(output)
        error    = forward - image
        if (error - prev_err).sum().abs() < tol:
            break
        prev_err = error
        gradient = camera.adjoint(error)
        output   = output - alpha * gradient
        output   = torch.maximum(output, torch.tensor(0))

    return camera.crop(torch.maximum(output, torch.tensor(0)))


def reconstruct_fista(camera, image, iters=10000, tol=0, tqdm=True):
    """
    Reconstruct an image with Fast Iterative Shrinkage Algorithm
    """
    vk = torch.zeros_like(camera.pad(image))
    xk_next = torch.zeros_like(vk)
    tk_next = torch.tensor(1)
    alpha = 1.8 / torch.max(camera.autocorrelation())

    prev_err = 1e6
    for i in trange(0, iters, disable=not tqdm):
        xk_prev  = xk_next
        forward  = camera.forward(vk)
        error    = forward - image
        if (error - prev_err).sum().abs() < tol:
            break
        prev_err = error
        gradient = camera.adjoint(error)
        vk       = vk - alpha * gradient
        xk_next  = torch.maximum(vk, torch.tensor(0))
        tk_prev  = (1 + torch.sqrt(1 + 4 * tk_next ** 2)) / 2
        vk       = xk_next + (tk_next - 1) / tk_prev * (xk_next - xk_prev)
        tk_next  = tk_prev

    return camera.crop(torch.maximum(vk, torch.tensor(0)))


def soft_threshold(image, thresh):
    return torch.sign(image) * torch.maximum(torch.abs(image) - thresh, torch.tensor(0))


def reconstruct_admm(camera, measurement, mu1, mu2, mu3, tau, iters=100):
    """
    Reconstruct an image with ADMM
    """
    # reconstruction
    v = torch.zeros_like(pad(measurement))

    # tv
    tv = TotalVariation()
    tv_fft = tv.gramian(v, fourier=True)

    # primals
    x = torch.zeros_like(v)
    u = torch.zeros_like(tv.forward(v))
    w = torch.zeros_like(v)
    
    # duals
    dual_x = torch.zeros_like(v)
    dual_u = torch.zeros_like(v)
    dual_w = torch.zeros_like(v)

    x_mul = 1.0 / (pad(torch.ones_like(measurement)) + mu1)
    v_mul = 1.0 / (mu1 * camera.autocorrelation().abs() + mu2 * tv_fft.abs() + mu3)
    
    for i in trange(0, iters):
        x = x_mul * (dual_x + mu1 * ift(camera.h * ft(v)) + pad(measurement))
        u = soft_threshold(tv.forward(v) + dual_u / mu2, tau / mu2)
        w = torch.maximum(dual_w / mu3 + v, torch.tensor(0))
        r = ift(camera.h.conj() * ft(mu1 * x - dual_x)) + tv.adjoint(mu2 * u - dual_u) + (mu3 * w - dual_w)
        v = ift(v_mul * ft(r))
        dual_x = dual_x + mu1 * (ift(camera.h * ft(v)) - x)
        dual_u = dual_u + mu2 * (tv.forward(v) - u)
        dual_w = dual_w + mu3 * (v - w)
    return torch.maximum(crop(v), torch.tensor(0))


def crop(x):
    h, w = x.shape[-2:]
    return torchvision.transforms.functional.center_crop(x, (h // 2, w // 2))


def pad(x, mode='constant'):
    h, w = x.shape[-2:]
    ys, yr = divmod(h, 2)
    xs, xr = divmod(w, 2)
    return torchvision.transforms.functional.pad(x, (xs, ys, xs + xr, ys + yr), padding_mode=mode)


def ft(image):
    return fft.rfft2(fft.ifftshift(image, dim=(-2, -1)), norm='ortho')


def ift(image):
    return fft.fftshift(fft.irfft2(image, norm='ortho'), dim=(-2, -1))


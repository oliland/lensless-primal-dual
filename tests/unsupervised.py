from pathlib import Path

import torch
import numpy as np
from PIL import Image

from lensless.models.camera import DiffuserCam
from lensless.models.solver import reconstruct_fista, reconstruct_gd


def test_fista():
    samples = Path(__file__).parent / 'unsupervised_samples'
    psf = Image.open(samples / 'psf.png')
    psf = torch.from_numpy(np.array(psf)).permute(2, 0, 1)
    psf = psf / 255.
    psf = psf / torch.linalg.norm(psf.ravel())

    image = Image.open(samples / 'circles.png')
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    image = image / 255.
    image = image / torch.linalg.norm(psf.ravel())

    # reconstruct image
    output = reconstruct_fista(DiffuserCam(psf), image, 300)
    output = (output / torch.amax(output)) * 255.

    # take average of color channels
    output = output.mean(0)

    # binary threshold
    output[output > 30] = 255
    output[output < 30] = 0
    output = output.byte().cpu().numpy()
    output = Image.fromarray(output)
    # uncomment to overwrite reference image
    # output.save(samples / 'recon_fista.png')

    # test against existing reconstruction
    recon = Image.open(samples / 'recon.png')
    recon = np.array(recon)
    assert np.allclose(output, recon)

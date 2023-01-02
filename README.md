# Unrolled Primal-Dual Networks for Lensless Imaging

This directory contains training and evaluation code for Learned Point Spread Functions for Lensless Imaging, using the [DiffuserCam Lensless Mirflickr Dataset](https://waller-lab.github.io/LenslessLearning/dataset.html).

## Models

### Ground Truth images
![ground truth](/samples/ground_truth.png)

### image_optimizer:

Colors are hidden in the batch axis during training to preserve color information, at the cost of noise.
It's unclear what kind of noise this is.

### image_optimizer_color:

PSF and Image colors are tiled 5 times when the image is input, once for each of the 5 learned PSFs.

The 3x3 convolutional layer is additionally split into 5 groups of 9 channels.

Each of the 5 groups contains 9 channels of RGBRGBRGB due to the way that the PSF and images are tiled channel wise.

This limits the connectivity of the resulting 3x3 convolution, preventing the network from becoming "confused" and creating a grayscale image. As a result, this network prevents "color noise", at the cost of some color inaccuracy (some objects are incorrectly colored).

# Weights and Updates:

## 2021-09-20

### image_optimizer

![denoise on](/samples/image_optimizer.png)
![denoise off](/samples/image_optimizer_noise.png)

- URL: weights/image_optimizer.ckpt
- Test MSE Loss: 0.0055 (88ms)
- Test MSE Loss (denoise off): 0.0099 (80ms)

### image_optimizer_colors

![denoise on](/samples/image_optimizer_colors.png)
![denoise off](/samples/image_optimizer_colors_noise.png)

- URL: weights/image_optimizer_colors.ckpt
- Test MSE Loss: 0.0038 (84ms)
- Test MSE Loss (denoise off): 0.0118 (82ms)

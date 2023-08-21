# %% [markdown]
"""
<hr style="height:2px;">

# Train a Noise2Void network

Both the CARE network and Noise2Noise network you trained in part 1 and 2 require that you acquire additional data for the purpose of denoising. For CARE we used a paired acquisition with high SNR, for Noise2Noise we had paired noisy acquisitions.
We will now train a Noise2Void network from single noisy images.

This notebook uses a single image from the SEM data from the Noise2Noise notebook, but as you'll see in Task 3.1 if you brought your own raw data you should adapt the notebook to use that instead.

We now use the [Noise2Void library](https://github.com/juglab/n2v) instead of csbdeep/care, but don't worry - they're pretty similar.

<div class="alert alert-danger">
Set your python kernel to <code>03_image_restoration_part2</code>
</div>
<div class="alert alert-danger">
Make sure your previous notebook is shutdown to avoid running into GPU out-of-memory problems.
</div>

---

<div class="alert alert-block alert-info"><h4>
    TASK 3.1</h4>
    <p>
This notebook uses a single image from the SEM data from the Noise2Noise notebook.

If you brought your own raw data, use that instead!
The only requirement is that the noise in your data is pixel-independent and zero-mean. If you're unsure whether your data fulfills that requirement or you don't yet understand why it is necessary ask one of us to discuss!

If you don't have suitable data of your own, feel free to find some online or ask your fellow course participants. You can however also stick with the SEM data provided here and compare the results to what you achieved with Noise2Noise in the previous part.
    </p>
</div>

---
"""
# %%
# We import all our dependencies.
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tifffile import imread
import zipfile

# %load_ext tensorboard

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# %% [markdown]
"""
<hr style="height:2px;">

## Part 1: Prepare data
Let's make sure the data is there!
"""
# %%
assert os.path.exists("data/SEM/train/train.tif")
assert os.path.exists("data/SEM/test/test.tif")
# %% [markdown]
"""
We create a N2V_DataGenerator object to help load data and extract patches for training and validation.
"""
# %%
datagen = N2V_DataGenerator()
# %% [markdown]
"""
The data generator provides two methods for loading data: `load_imgs_from_directory` and `load_imgs`. Let's look at their docstring to figure out how to use it.
"""
# %%
# ?N2V_DataGenerator.load_imgs_from_directory
# %%
# ?N2V_DataGenerator.load_imgs
# %% [markdown]
"""
The SEM images are all in one directory, so we'll use `load_imgs_from_directory`. We'll pass in that directory (`"data/SEM/train"`), our image matches the default filter (`"*.tif"`) so we do not need to specify that. But our tif image is a stack of several images, so as dims we need to specify `"TYX"`.
If you're using your own data adapt this part to match your use case. If these functions aren't suitable for your use case load your images manually.
Feel free to ask a TA for help if you're unsure how to get your data loaded!
"""
# %%
imgs = datagen.load_imgs_from_directory("data/SEM/train", dims="TYX")
print(f"Loaded {len(imgs)} images.")
print(f"First image has shape {imgs[0].shape}")
# %% [markdown]
"""
The method returned a list of images, as per the doc string the dimensions of each are "SYXC". However, we only want to use one of the images here since Noise2Void is designed to work with just one acquisition of the sample. Let's use the first image at $1\mu s$ scantime.
"""
# %%
imgs = [img[2:3, :, :, :] for img in imgs]
print(f"First image has shape {imgs[0].shape}")
# %% [markdown]
"""
For generating patches the datagenerator provides the methods `generate_patches` and `generate_patches_from_list`. As before, let's have a quick look at the docstring
"""
# %%
# ?N2V_DataGenerator.generate_patches
# %%
# ?N2V_DataGenerator.generate_patches_from_list
# %%
type(imgs)
# %% [markdown]
"""
Our `imgs` object is a list, so `generate_patches_from_list` is the suitable function.
"""
# %%
patches = datagen.generate_patches_from_list(imgs, shape=(96, 96))
# %%
# split into training and validation
n_train = int(round(0.9 * patches.shape[0]))
X, X_val = patches[:n_train, ...], patches[n_train:, ...]
# %% [markdown]
"""
As per usual, let's look at a training and validation patch to make sure everything looks okay.
"""
# %%
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(X[np.random.randint(X.shape[0]), ..., 0], cmap="gray_r")
plt.title("Training patch")
plt.subplot(1, 2, 2)
plt.imshow(X_val[np.random.randint(X_val.shape[0]), ..., 0], cmap="gray_r")
plt.title("Validation patch")
# %% [markdown]
"""
<hr style="height:2px;">

## Part 2: Configure and train the Noise2Void Network

Noise2Void comes with a special config-object, where we store network-architecture and training specific parameters. See the docstring of the <code>N2VConfig</code> constructor for a description of all parameters.

When creating the config-object, we provide the training data <code>X</code>. From <code>X</code> the library will extract <code>mean</code> and <code>std</code> that will be used to normalize all data before it is processed by the network.


Compared to supervised training (i.e. traditional CARE), we recommend to use N2V with an increased <code>train_batch_size</code> (e.g. 128) and <code>batch_norm</code>.

To keep the network from learning the identity we have to manipulate the input pixels for the blindspot during training. How to exactly manipulate those values is controlled via the <code>n2v_manipulator</code> parameter with default value <code>'uniform_withCP'</code> which samples a random value from the surrounding pixels, including the value at the control point. The size of the  surrounding area can be configured via <code>n2v_neighborhood_radius</code>.

The [paper supplement](https://arxiv.org/src/1811.10980v2/anc/supp_small.pdf) describes other pixel manipulators as well (section 3.1). If you want to configure one of those use the following values for <code>n2v_manipulator</code>:
* <code>"normal_additive"</code> for Gaussian (<code>n2v_neighborhood_radius</code> will set sigma)
* <code>"normal_fitted"</code> for Gaussian Fitting
* <code>"normal_withoutCP"</code> for Gaussian Pixel Selection

For faster training multiple pixels per input patch can be manipulated. In our experiments we manipulated about 0.198% of the input pixels per patch. For a patch size of 64 by 64 pixels this corresponds to about 8 pixels. This fraction can be tuned via <code>n2v_perc_pix</code>.

For Noise2Void training it is possible to pass arbitrarily large patches to the training method. From these patches random subpatches of size <code>n2v_patch_shape</code> are extracted during training. Default patch shape is set to (64, 64).

In the past we experienced bleedthrough artifacts between channels if training was terminated to early. To counter bleedthrough we added the `single_net_per_channel` option, which is turned on by default. In the back a single U-Net for each channel is created and trained independently, thereby removing the possiblity of bleedthrough. <br/>
Essentially the network gets multiplied by the number of channels, which increases the memory requirements. If your GPU gets too small, you can always split the channels manually and train a network for each channel one after another.

---
<div class="alert alert-block alert-info"><h4>
    TASK 3.2</h4>
    <p>
As suggested look at the docstring of the N2VConfig and then generate a configuration for your Noise2Void network, and choose a name to identify your model by.
    </p>
</div>
"""
# %%
# ?N2VConfig
# %%
###TODO###
config = N2VConfig()
vars(config)
model_name = ""
# %% tags=["solution"]
# train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch
# is shown once per epoch.
config = N2VConfig(
    X,
    unet_kern_size=3,
    train_steps_per_epoch=int(X.shape[0] / 128),
    train_epochs=200,
    train_loss="mse",
    batch_norm=True,
    train_batch_size=128,
    n2v_perc_pix=0.198,
    n2v_patch_shape=(64, 64),
    n2v_manipulator="uniform_withCP",
    n2v_neighborhood_radius=5,
)

# Let's look at the parameters stored in the config-object.
vars(config)
model_name = "n2v_2D"

# %% [markdown]
"""
---
"""
# %%
# initialize the model
model = N2V(config, model_name, basedir="models")
# %% [markdown]
"""
Now let's train the model and monitor the progress in tensorboard.
Adapt the command below as you did before.
"""
# %%
# %tensorboard --logdir=models
# %%
history = model.train(X, X_val)
# %%
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ["loss", "val_loss"])
# %% [markdown]
"""
<hr style="height:2px;">

## Part 3: Prediction

Similar to CARE a previously trained model is loaded by creating a new N2V-object without providing a `config`.
"""
# %%
model = N2V(config=None, name=model_name, basedir="models")
# %% [markdown]
"""
Let's load a $1\mu s$ scantime test images and denoise them using our network and like before we'll use the high SNR image to make a quantitative comparison. If you're using your own data and don't have an equivalent you can ignore that part.
"""
# %%
test_img = imread("data/SEM/test/test.tif")[2, ...]
test_img_highSNR = imread("data/SEM/test/test.tif")[-1, ...]
print(f"Loaded test image with shape {test_img.shape}")
# %%
test_denoised = model.predict(test_img, axes="YX", n_tiles=(2, 1))
# %% [markdown]
"""
Let's look at the results
"""
# %%
plt.figure(figsize=(30, 30))
plt.subplot(2, 3, 1)
plt.imshow(test_img, cmap="gray_r")
plt.title("Noisy test image")
plt.subplot(2, 3, 4)
plt.imshow(test_img[2000:2200, 500:700], cmap="gray_r")
plt.subplot(2, 3, 2)
plt.imshow(test_denoised, cmap="gray_r")
plt.title("Denoised test image")
plt.subplot(2, 3, 5)
plt.imshow(test_denoised[2000:2200, 500:700], cmap="gray_r")
plt.subplot(2, 3, 3)
plt.imshow(test_img_highSNR, cmap="gray_r")
plt.title("High SNR image (4x5us)")
plt.subplot(2, 3, 6)
plt.imshow(test_img_highSNR[2000:2200, 500:700], cmap="gray_r")
plt.show()
# %% [markdown]
"""
---
<div class="alert alert-block alert-info"><h4>
    TASK 3.3</h4>
    <p>

If you're using the SEM data (or happen to have a high SNR version of the image you predicted from) compare the structural similarity index and peak signal to noise ratio (wrt the high SNR image) of the noisy input image and the predicted image. If not, just skip this task.
    </p>
</div>
"""
# %%
###TODO###
ssi_input = ...  # TODO
ssi_restored = ...  # TODO
print(
    f"Structural similarity index (higher is better) wrt average of 4x5us images: \n"
    f"Input: {ssi_input} \n"
    f"Prediction: {ssi_restored}"
)
psnr_input = ...  # TODO
psnr_restored = ...  # TODO
print(
    f"Peak signal-to-noise ratio (higher is better) wrt average of 4x5us images:\n"
    f"Input: {psnr_input} \n"
    f"Prediction: {psnr_restored}"
)

# %% tags = ["solution"]
ssi_input = structural_similarity(test_img, test_img_highSNR, data_range=65535)
ssi_restored = structural_similarity(test_denoised, test_img_highSNR, data_range=65535)
print(
    f"Structural similarity index (higher is better) wrt average of 4x5us images: \n"
    f"Input: {ssi_input} \n"
    f"Prediction: {ssi_restored}"
)
psnr_input = peak_signal_noise_ratio(test_img, test_img_highSNR, data_range=65535)
psnr_restored = peak_signal_noise_ratio(
    test_denoised, test_img_highSNR, data_range=65535
)
print(
    f"Peak signal-to-noise ratio (higher is better) wrt average of 4x5us images:\n"
    f"Input: {psnr_input} \n"
    f"Prediction: {psnr_restored}"
)
# %% [markdown]
"""
---
<hr style="height:2px;">
<div class="alert alert-block alert-success"><h1>
    Congratulations!</h1>
    <p>
    <b>You have reached the third checkpoint of this exercise! Please mark your progress in the course chat!</b>
    </p>
    <p>
    Consider sharing some pictures of your results on element, especially if you used your own data.
    </p>
    <p>
    If there's still time, check out the bonus exercise.
    </p>
</div>
"""

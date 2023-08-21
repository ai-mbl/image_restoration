# %% [markdown]
"""
<hr style="height:2px;">

# Train a Noise2Noise network with CARE
<div class="alert alert-danger">
Set your python kernel to <code>03_image_restoration_part1</code>! That's the same as for the first notebook.
</div>

We will now train a 2D Noise2Noise network using CARE. We will closely follow along the previous example but now you will have to fill in some parts on your own!
You will have to make decisions - make them!

But first some clean up...
<div class="alert alert-danger">
Make sure your previous notebook is shutdown to avoid running into GPU out-of-memory problems.
</div>

![](nb_material/notebook_shutdown.png)
"""
# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import gc
import os

import matplotlib.pyplot as plt
import numpy as np
from csbdeep.data import RawData, create_patches
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import CARE, Config
from csbdeep.utils import (
    Path,
    axes_dict,
    plot_history,
    plot_some,
)
from csbdeep.utils.tf import limit_gpu_memory

# %matplotlib inline
# %load_ext tensorboard
# %config InlineBackend.figure_format = 'retina'
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tifffile import imread, imwrite

# %% [markdown]
"""
<hr style="height:2px;">

## Part 1: Training Data Generation

### Download example data

To train a Noise2Noise setup we need several acquisitions of the same sample.
The SEM data we downloaded during setup contains 2 tiff-stacks, one for training and one for testing, let's make sure it's there!
"""
# %%
assert os.path.exists("data/SEM/train/train.tif")
assert os.path.exists("data/SEM/test/test.tif")

# %% [markdown]
# Let's have a look at the data!
# Each image is a tiff stack containing 7 images of the same tissue recorded with different scan time settings of a Scanning Electron Miscroscope (SEM). The faster a SEM image is scanned, the noisier it gets.

# %%
imgs = imread("data/SEM/train/train.tif")
x_size = imgs.shape
print("image size =", x_size)
scantimes_all = ["0.2us", "0.5us", "1us", "1us", "2.1us", "5us", "5us, avg of 4"]
plt.figure(figsize=(40, 16))
plot_some(imgs, title_list=[scantimes_all], pmin=0.2, pmax=99.8, cmap="gray_r")

# %% [markdown]
# ---
# <div class="alert alert-block alert-info"><h4>
#     TASK 2.1:</h4>
#     <p>
#     The noise level is hard to see at this zoom level. Let's also look at a smaller crop of them! Play around with this until you have a feeling for what the data looks like.
#     </p>
# </div>

# %%
###TODO###

imgs_cropped = ...  # TODO
# %% tags=["solution"]
imgs_cropped = imgs[:, 1000:1128, 600:728]
# %%
plt.figure(figsize=(40, 16))
plot_some(imgs_cropped, title_list=[scantimes_all], pmin=0.2, pmax=99.8, cmap="gray_r")

# %% [markdown]
"""
---
"""
# %%
# checking that you didn't crop x_train itself, we still need that!
assert imgs.shape == x_size

# %% [markdown]
"""
As you can see the last image, which is the average of 4 images with 5us scantime, has the highest signal-to-noise-ratio. It is not noise-free but our best choice to be able to compare our results against quantitatively, so we will set it aside for that purpose.
"""
# %%
scantimes, scantime_highSNR = scantimes_all[:-1], scantimes_all[-1]
x_train, x_highSNR = imgs[:-1], imgs[-1]
print(scantimes, scantime_highSNR)
print(x_train.shape, x_highSNR.shape)

# %% [markdown]
"""
### Generate training data for CARE

Let's try and train a network to denoise images of $1 \mu s$ scan time!
Which images do you think could be used as input and which as target?

---
<div class="alert alert-block alert-info"><h4>
    TASK 2.2:</h4>
    <p>
    Decide which images to use as inputs and which as targets. Then, remember from part one how the data has to be organized to match up inputs and targets.
    </p>
</div>
"""
# %%
###TODO###
base_path = "data/SEM/train"
source_dir = os.path.join(base_path, "")  # pick path in which to save inputs
target_dir = os.path.join(base_path, "")  # pick path in which to save targets
# %% tags=["solution"]
# The names "low" and "GT" don't really fit here anymore, so use names "source" and "target" instead
base_path = "data/SEM/train"
source_dir = os.path.join(base_path, "source_1us")
target_dir = os.path.join(base_path, "target_1us")

# %%
os.makedirs(source_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

# %%
# Now save individual images into these directories
# You can use the imwrite function to save images. The ? command will pull up the docstring
# ?imwrite
# %% [markdown]
"""
<span style="color:blue;font-weight:bold;">Hint</span>: The tiff file you read earlier contained 7 images for the different instances. Here, use a single tiff file per image.
"""
# %% [markdown]
"""
<span style="color:blue;font-weight:bold;">Hint</span>: Remember we're trying to train a Noise2Noise network here, so the target does not need to be clean.
"""
# %%
###TODO###

# Put the pairs of input and target images into the `source_dir` and `target_dir`, respectively.
# The goal here is to the train a network for 1 us scan time.

# %% tags = ["solution"]
# Since we wanna train a network for images of 1us scan time, we will use the two images as our input images.
# For both of these images we can use every other image as our target - as long as the noise is different the
# only remaining structure is the signal, so mixing different scan times is totally fine.
# Images are paired by having the same name in `source_dir` and `target_dir`. This means we'll have several
# copies of the same image with different names. These images aren't very big, so that's fine.
counter = 0
for i in range(2, 4):
    for j in range(x_train.shape[0]):
        if i == j:
            continue
        imwrite(os.path.join(source_dir, f"{counter}.tif"), x_train[i, ...])
        imwrite(os.path.join(target_dir, f"{counter}.tif"), x_train[j, ...])
        counter += 1
# %% [markdown]
"""
---
---
<div class="alert alert-block alert-info"><h4>
    TASK 2.3:</h4>
    <p>
    Now that you arranged the training data we can now create the raw data object.
    </p>
</div>
"""

# %%
###TODO###
# raw_data = RawData.from_folder (
#    basepath    = 'data/SEM/train',
#    source_dirs = [''], # fill in your directory for source images
#    target_dir  = '', # fill in your directory of target images
#    axes        = '', # what should the axes tag be?
# )
#
# %% tags=["solution"]
raw_data = RawData.from_folder(
    basepath="data/SEM/train",
    source_dirs=["source_1us"],  # fill in your directory for source images
    target_dir="target_1us",  # fill in your directory of target images
    axes="YX",  # what should the axes tag be?
)
# %% [markdown]
"""
---
We generate 2D patches. If you'd like, you can play around with the parameters here.
"""
# %%
X, Y, XY_axes = create_patches(
    raw_data=raw_data,
    patch_size=(256, 256),
    n_patches_per_image=512,
    save_file="data/SEM/my_1us_training_data.npz",
)

assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

# %% [markdown]
"""
### Show

Let's look at some of the generated patch pairs. (odd rows: _source_, even rows: _target_)
"""
# %%
for i in range(2):
    plt.figure(figsize=(16, 4))
    sl = slice(8 * i, 8 * (i + 1)), 0
    plot_some(
        X[sl], Y[sl], title_list=[np.arange(sl[0].start, sl[0].stop)], cmap="gray_r"
    )
plt.show()


# %% [markdown]
"""
<hr style="height:2px;">

## Part 2: Training the network


### Load Training data

Load the patches generated in part 1, use 10% as validation data.
"""
# %%
(X, Y), (X_val, Y_val), axes = load_training_data(
    "data/SEM/my_1us_training_data.npz", validation_split=0.1, verbose=True
)

c = axes_dict(axes)["C"]
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


plt.figure(figsize=(12, 5))
plot_some(X_val[:5], Y_val[:5], cmap="gray_r", pmin=0.2, pmax=99.8)
plt.suptitle("5 example validation patches (top row: source, bottom row: target)")

config = Config(
    axes, n_channel_in, n_channel_out, train_steps_per_epoch=10, train_epochs=100
)
vars(config)

# %% [markdown]
"""
We now create a CARE model with the chosen configuration:
"""
# %%
model = CARE(config, "my_N2N_model", basedir="models")

# %% [markdown]
"""
### Training

Training the model will likely take some time. We recommend to monitor the progress with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard), which allows you to inspect the losses during training.
Furthermore, you can look at the predictions for some of the validation images, which can be helpful to recognize problems early on.

Start tensorboard as you did in the previous notebook.
"""
# %%
# %tensorboard --logdir models
# %%
history = model.train(X, Y, validation_data=(X_val, Y_val))

# %% [markdown]
# Plot final training history (available in TensorBoard during training):

# %%
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ["loss", "val_loss"], ["mse", "val_mse", "mae", "val_mae"])

# %% [markdown]
"""
### Evaluation
Example results for validation images.
"""
# %%
plt.figure(figsize=(12, 7))
_P = model.keras_model.predict(X_val[:5])
if config.probabilistic:
    _P = _P[..., : (_P.shape[-1] // 2)]
plot_some(X_val[:5], Y_val[:5], _P, pmin=0.2, pmax=99.8, cmap="gray_r")
plt.suptitle(
    "5 example validation patches\n"
    "top row: input (noisy source),  "
    "mid row: target (independently noisy),  "
    "bottom row: predicted from source,   "
)

# %% [markdown]
"""
<hr style="height:2px;">

## Part 3: Prediction


### Load CARE model

Load trained model (located in base directory `models` with name `my_model`) from disk.
The configuration was saved during training and is automatically loaded when `CARE` is initialized with `config=None`.
"""
# %%
model = CARE(config=None, name="my_N2N_model", basedir="models")
# %% [markdown]
"""
### Apply CARE network to raw image
Now use the trained model to denoise some test images. Let's load the whole tiff stack first
"""
# %%
path_test_data = "data/SEM/test/test.tif"
test_imgs = imread(path_test_data)
axes = "YX"

# separate out the high SNR image as before
x_test, x_test_highSNR = test_imgs[:-1], test_imgs[-1]


# %% [markdown]
"""
---
<div class="alert alert-block alert-info"><h4>
    TASK 2.4:</h4>
    <p>
    Write a function that applies the model to one of the images in the tiff stack. Code to visualize the result by plotting the noisy image alongside the restored image as well as smaller crops of each is provided.
    </p>
</div>
"""


# %%
###TODO###
def apply_on_test(predict_model, img_idx, plot=True):
    """
    Apply the given model on the test image at the given index of the tiff stack.
    Returns the noisy image, restored image and the scantime.
    """
    # TODO: insert your code for prediction here
    scantime = ...  # get scantime for `img_idx`th image
    img = ...  # get `img_idx`th image
    restored = ...  # apply model to `img`
    if plot:
        img_crop = img[500:756, 200:456]
        restored_crop = restored[500:756, 200:456]
        x_test_highSNR_crop = x_test_highSNR[500:756, 200:456]
        plt.figure(figsize=(20, 30))
        plot_some(
            np.stack([img, restored, x_test_highSNR]),
            np.stack([img_crop, restored_crop, x_test_highSNR_crop]),
            cmap="gray_r",
            title_list=[[scantime, "restored", scantime_highSNR]],
        )
    return img, restored, scantime


# %% tags = ["solution"]
def apply_on_test(predict_model, img_idx, plot=True):
    """
    Apply the given model on the test image at the given index of the tiff stack.
    Returns the noisy image, restored image and the scantime.
    """
    scantime = scantimes[img_idx]
    img = x_test[img_idx, ...]
    axes = "YX"
    restored = predict_model.predict(img, axes)
    if plot:
        img_crop = img[500:756, 200:456]
        restored_crop = restored[500:756, 200:456]
        x_test_highSNR_crop = x_test_highSNR[500:756, 200:456]
        plt.figure(figsize=(20, 30))
        plot_some(
            np.stack([img, restored, x_test_highSNR]),
            np.stack([img_crop, restored_crop, x_test_highSNR_crop]),
            cmap="gray_r",
            title_list=[[scantime, "restored", scantime_highSNR]],
        )
    return img, restored, scantime


# %% [markdown]
"""
---

Using the function you just wrote to restore one of the images with 1us scan time.
"""
# %%
noisy_img, restored_img, scantime = apply_on_test(model, 2)

ssi_input = structural_similarity(noisy_img, x_test_highSNR, data_range=65535)
ssi_restored = structural_similarity(restored_img, x_test_highSNR, data_range=65535)
print(
    f"Structural similarity index (higher is better) wrt average of 4x5us images: \n"
    f"Input: {ssi_input} \n"
    f"Prediction: {ssi_restored}"
)

psnr_input = peak_signal_noise_ratio(noisy_img, x_test_highSNR, data_range=65535)
psnr_restored = peak_signal_noise_ratio(restored_img, x_test_highSNR, data_range=65535)
print(
    f"Peak signal-to-noise ratio wrt average of 4x5us images:\n"
    f"Input: {psnr_input} \n"
    f"Prediction: {psnr_restored}"
)

# %% [markdown]
"""
---
<div class="alert alert-block alert-info"><h4>
    TASK 2.5:</h4>
    <p>
    Be creative!

Can you improve the results by using the data differently or by tweaking the settings?

How could you train a single network to process all scan times?
    </p>
</div>
"""

# %% [markdown]
"""
To train a network to process all scan times use this instead as the solution to Task 2.3:
The names "low" and "GT" don't really fit here anymore, so use names "source_all" and "target_all" instead
"""
# %%
source_dir = "data/SEM/train/source_all"
target_dir = "data/SEM/train/target_all"
# %%
os.makedirs(source_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

# %% [markdown]
"""
Since we wanna train a network for all scan times, we will use all images as our input images.
To train Noise2Noise we can use every other image as our target - as long as the noise is different the only remianing structure is the signal, so mixing different scan times is totally fine.
Images are paired by having the same name in `source_dir` and `target_dir`. This means we'll have several copies of the same image with different names. These images aren't very big, so that's fine.
"""
# %%
counter = 0
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[0]):
        if i == j:
            continue
        imwrite(os.path.join(source_dir, f"{counter}.tif"), x_train[i, ...])
        imwrite(os.path.join(target_dir, f"{counter}.tif"), x_train[j, ...])
        counter += 1

# %% [markdown]
"""
---
<hr style="height:2px;">
<div class="alert alert-block alert-success"><h1>
    Congratulations!</h1>
    <p>
    <b>You have reached the second checkpoint of this exercise! Please mark your progress in the course chat!</b>
    </p>
</div>
"""

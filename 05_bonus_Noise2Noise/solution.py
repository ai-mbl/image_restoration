# %% tags=["solution", "task"]
# ruff: noqa: F811
# %% [markdown] tags=[]
# # Noise2Noise
#
# CARE networks like the one you trained in the first image restoration exercise require that you acquire pairs
# of high and low signal-to-noise ratio images. However, this often extremely challenging or, even, not possible. One such case is when it is simply
# not possible to acquire high SNR images as the sample is too much susceptible to illumination.
#
# What to do when you are stuck with just noisy images? We have already seen Noise2Void, which
# is a self-supervised method that can be trained on noisy images. But there are other 
# supervised approaches that can be trained on noisy images only, such as Noise2Noise. 
# Noise2Noise (N2N) is very similar to CARE, except instead of using noisy inputs and clean targets, N2N uses noisy inputs and noisy targets.
# This paired data would be aquired by taking two images of your sample in quick succession.
#
# Noise2Noise relies on 2 assumptions: 
# 1. The noise in one image is statistically independent of the noise in any other image.
# That is, knowing the value of the random noise in one image tells you nothing about the random noise in another other image.
# 2. The noise is on average zero. Meaning that, while noise can randomly increase or decrease the intensity of a pixel, the average change will be zero.
#
# These assumptions are widely met by imaging noise.
# Therefore, if we train a neural network to predict one noisy image from another using the mean squared error loss function, the network will learn to predict a denoised image.
# Theoretically, we can achieve the exact same result as CARE without any clean images!
# However, in practice, N2N will require more training data than CARE to make up for the noisier training signal.
#
# In this notebook, we will again use the [Careamics](https://careamics.github.io) library.
#
# <p align="center">
#     <img src="https://raw.githubusercontent.com/CAREamics/.github/main/profile/images/banner_careamics.png" width=400>
# </p>
#
# ## Reference
#
# Lehtinen, Jaakko, et al. "[Noise2Noise: Learning image restoration without clean data.](https://arxiv.org/abs/1803.04189)" arXiv preprint arXiv:1803.04189 (2018).
#
#
# <div class="alert alert-block alert-success"><h3>Objectives</h3>
#     
# - Understand the differences between CARE, Noise2Noise and Noise2Void
# - Train Noise2Noise with CAREamics
#   
# </div>
#
#

# %% [markdown] tags=[]
# <div class="alert alert-danger">
# Set your python kernel to <code>05_image_restoration</code>
# </div>

# %% tags=[]
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from careamics import CAREamist
from careamics.config import create_n2n_configuration

# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# ## Part 1: Prepare the data
#
# The N2N SEM dataset consists of EM images with 7 different levels of noise:
#
# - Image 0 is recorded with 0.2 us scan time
# - Image 1 is recorded with 0.5 us scan time
# - Image 2 is recorded with 1 us scan time
# - Image 3 is recorded with 1 us scan time
# - Image 4 is recorded with 2.1 us scan time
# - Image 5 is recorded with 5.0 us scan time
# - Image 6 is recorded with 5.0 us scan time and is the avg. of 4 images
#
# Let's have a look at them.

# %% [markdown] tags=[]
# ### Visualize training data
#
# In this cell we can see the different levels of noise in the SEM dataset

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 1: Explore the data</h3>
#
# Here we load a training image to visualize it. Can you visually tell if the noise is pixel-independent?
# In case you are not sure, try to think how you would experimentally verify this assumption (you do not need to actually do it, just think about it remembering what we saw in previous exercises).
#
# </div>

# %% tags=[]
# Load images
root_path = Path("./../data")
train_image = tifffile.imread(root_path / "denoising-N2N_SEM.unzip/SEM/train.tif")
print(f"Train image shape: {train_image.shape}")

# plot image
vmin, vmax = np.percentile(train_image, (1, 99))
fig, ax = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
fig.patch.set_facecolor('black')
ax[0, 0].imshow(train_image[6, :1024, :1024], cmap="gray", vmin=vmin, vmax=vmax)
ax[0, 0].set_title("Train image - Lowest noise level", color='white')
ax[0, 0].axis("off")
ax[0, 1].axis("off")
ax[0, 2].axis("off")
ax[1, 0].imshow(train_image[5, :1024, :1024], cmap="gray", vmin=vmin, vmax=vmax)
ax[1, 0].axis("off")
ax[1, 1].imshow(train_image[4, :1024, :1024], cmap="gray", vmin=vmin, vmax=vmax)
ax[1, 1].axis("off")
ax[1, 2].imshow(train_image[3, :1024, :1024], cmap="gray", vmin=vmin, vmax=vmax)
ax[1, 2].axis("off")
ax[2, 0].imshow(train_image[2, :1024, :1024], cmap="gray", vmin=vmin, vmax=vmax)
ax[2, 0].axis("off")
ax[2, 1].imshow(train_image[1, :1024, :1024], cmap="gray", vmin=vmin, vmax=vmax)
ax[2, 1].axis("off")
ax[2, 2].imshow(train_image[0, :1024, :1024], cmap="gray", vmin=vmin, vmax=vmax)
ax[2, 2].axis("off")


# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# ## Part 2: Create the configuraion
#
# As in the Noise2Void exercise, a good CAREamics pipeline starts with a configuration!

# %% tags=[]
training_config = create_n2n_configuration(
    experiment_name="N2N_SEM",
    data_type="array",
    axes="SYX",
    patch_size=[128, 128],
    batch_size=128,
    num_epochs=20,
    logger="tensorboard"
)

# Visualize training configuration (also includes default parameters)
print(training_config)

# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# ## Part 3: Train the network
#
# In this part, we create our training engine (`CAREamics`) and start training the network.

# %% tags=[]
# create the engine
careamist = CAREamist(source=training_config)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 2: Which data to choose?</h3>
#
# How would you train a network to denoise images of 1 us scan time? Which images do you think could be used as input and which as target?
#
# Set the `train_source` and `train_target` accordingly and train the network.
#
# </div>

# %% tags=["task"]
# Create the training data and targets pairs
data1 = train_image[[2, 2, 2, 2, 2, 3, 3, 3, 3, 3], ...]
data2 = train_image[[0, 1, 3, 4, 5, 0, 1, 3, 4, 5], ...]
train_source = # YOUR CODE HERE
train_target = # YOUR CODE HERE

# %% tags=["solution"]
# Create the training data and targets pairs
data1 = train_image[[2, 2, 2, 2, 2, 3, 3, 3, 3, 3], ...]
data2 = train_image[[0, 1, 3, 4, 5, 0, 1, 3, 4, 5], ...]
train_source = data1
train_target = data2

# %% tags=[]
careamist.train(
    train_source=train_source,
    train_target=train_target
)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h1>Checkpoint 1: Training N2N</h1>
# </div>
#
#
# <hr style="height:2px;">
#
# ## Part 4: Prediction

# %% [markdown] tags=[]
# Let's load the test data and predict on it to assess how well the network performs!

# %% tags=[]
# Load images
test_image = tifffile.imread(root_path / "denoising-N2N_SEM.unzip/SEM/test.tif")

# %% tags=[]
prediction = careamist.predict(source=test_image[2], tile_size=(256, 256), axes="YX", tta_transforms=False)[0]

# %% [markdown] tags=[]
# ### Visualize predictions

# %% tags=[]
fig, ax = plt.subplots(1, 2, figsize=(10, 10), constrained_layout=True)
fig.patch.set_facecolor('black')
ax[0].imshow(test_image[-1], cmap="gray")
ax[0].set_title("Test image lowest noise level", color='white')
ax[0].axis("off")
ax[1].imshow(prediction[0, 0], cmap="gray")
ax[1].set_title("Prediction", color='white')
ax[1].axis("off")

# %% tags=[]
fig, ax = plt.subplots(1, 2, figsize=(15, 15), constrained_layout=True)
fig.patch.set_facecolor('black')
vmin  = test_image[0].min()
vmax = test_image[0].max()
ax[0].imshow((prediction.squeeze())[1000:1128, 500:628], cmap="gray",vmin=vmin, vmax=vmax)
ax[0].set_title("Prediction", color='white')
ax[1].imshow(test_image[-1].squeeze()[1000:1128, 500:628], cmap="gray", vmin=vmin, vmax=vmax)
ax[1].set_title("Test image lowest noise level", color='white')
ax[0].axis("off")
ax[1].axis("off")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 3: Different noise pairs</h3>
#
# Can you further improve your results by usign different `source` and `target`?
#
# How would you train a network to denoise all images, rather than just the 1 us ones?
#
# Try it and be creative!
#
# </div>

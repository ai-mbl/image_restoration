# %% tags=["solution", "task"]
# ruff: noqa: F811
# %% [markdown] tags=[]
# # Noise2Noise
#
# CARE networks like the one you trained in the first restoration exercise require that you acquire pairs
# of high and low SNR. However, this often not possible. One such case is when it is simply
# not possible to acquire high SNR images.
#
# What to do when you are stuck with just noisy images? We also have seen Noise2Void, which
# is a self-supervised method that can be trained on noisy images. But there are other 
# supervised approaches that can be trained on noisy images only, such as Noise2Noise. 
#
# Noise2Noise follows the same training method as CARE, except that instead of predicting clean images the UNet is trained to predict noisy images.
# The training data is obtained by imaging a sample twice, obtaining two images with the same underlying signal but different samples of noise.
# The noise in each image will be statistically independent of the noise in the other, meaning that seeing one image tells us nothing about the noise content of the other image.
# Therefore, a UNet that tries to guess one noisy image from another will be able to accurately predict the signal content, which is identical, but will be completely unable to predict the noise content.
# Nonetheless, it will try to make its best guess of the noise content - one that minimises the mean square error loss.
# Luckily for us, noise content is on average zero, meaning that the best best guess is zero-valued noise, i.e., no noise.
# This means that the network will converge to the same solution as it would if it were trained with clean targets, although it will require more data to compensate for the reduced signal content of the targets.
#
# In this notebook, we will again use the [Careamics](https://careamics.github.io) library.
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

# %% tags=[]
# Load images
root_path = Path("./../data")
train_image = tifffile.imread(root_path / "denoising-N2N_SEM.unzip/SEM/train.tif")
print(f"Train image shape: {train_image.shape}")

# plot image
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(train_image[0,100:356, 500:756], cmap="gray")
ax[0].set_title("Train image highest noise level")
ax[1].imshow(train_image[-1, 100:356, 500:756], cmap="gray")
ax[1].set_title("Train image lowest noise level")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 1: Explore the data</h3>
#
# Visualize each different noise level!
#
# </div>

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
    num_epochs=50,
    logger="tensorboard"
)

# Visualize training configuration
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

# %% tags=[]
# Create the training data and targets
train_data = train_image[[2, 2, 2, 2, 2, 3, 3, 3, 3, 3], ...]
train_target = train_image[[0, 1, 3, 4, 5, 0, 1, 3, 4, 5], ...]

# %% tags=["task"]
careamist.train(
    train_source=...,
    train_target=...
)

# %% tags=["solution"]
careamist.train(
    train_source=train_data,
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
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(test_image[-1], cmap="gray")
ax[0].set_title("Test image lowest noise level")
ax[1].imshow(prediction[0, 0], cmap="gray")
ax[1].set_title("Prediction")

# %% tags=[]
fi, ax = plt.subplots(1, 2, figsize=(15, 15))
vim  = test_image[0].min()
vmax = test_image[0].max()
ax[0].imshow((prediction.squeeze())[1000:1128, 500:628], cmap="gray",vmin=vim, vmax=vmax)
ax[0].set_title("Prediction")
ax[1].imshow(test_image[-1].squeeze()[1000:1128, 500:628], cmap="gray", vmin=vim, vmax=vmax)
ax[1].set_title("Test image lowest noise level")

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
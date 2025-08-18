# %% tags=["solution", "task"]
# ruff: noqa: F811
# %% [markdown] tags=[]
# # Noise2Void
#
# In the first exercise, we denoised images with CARE using supervised training. As 
# discussed during the lecture, ground-truth data is not always available in life 
# sciences. But no panic, Noise2Void is here to help!
#
# Indeed Noise2Void is a self-supervised algorithm, meaning that it trains on the data
# itself and does not require clean images. The idea is to predict the value of a masked
# pixels based on the information from the surrounding pixels. Two underlying hypothesis
# allow N2V to work: the structures are continuous and the noise is pixel-independent, 
# that is to say the amount of noise in one pixel is independent from the amount of noise
# in the surrounding pixels. Fortunately for us, it is very often the case in microscopy images!
#
# If N2V does not require pairs of noisy and clean images, then how does it train?
#
# First it selects random pixels in each patch, then it masks them. The masking is 
# not done by setting their value to 0 (which could disturb the network since it is an
# unexpected value) but by replacing the value with that of one of the neighboring pixels.
#
# Then, the network is trained to predict the value of the masked pixels. Since the masked
# value is different from the original value, the network needs to use the information
# contained in all the pixels surrounding the masked pixel. If the noise is pixel-independent,
# then the network cannot predict the amount of noise in the original pixel and it ends
# up predicting a value close to the "clean", or denoised, value.
#
# In this notebook, we will use an existing library called [Careamics](https://careamics.github.io)
# that includes N2V and other algorithms:
#
# <p align="center">
#     <img src="https://raw.githubusercontent.com/CAREamics/.github/main/profile/images/banner_careamics.png" width=400>
# </p>
#
#
# ## References
#
# - Alexander Krull, Tim-Oliver Buchholz, and Florian Jug. "[Noise2Void - learning denoising from single noisy images.](https://openaccess.thecvf.com/content_CVPR_2019/html/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.html)" Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2019.
# - Joshua Batson, and Loic Royer. "[Noise2self: Blind denoising by self-supervision.](http://proceedings.mlr.press/v97/batson19a.html)" International Conference on Machine Learning. PMLR, 2019.

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h3>Objectives</h3>
#     
# - Understand how N2V masks pixels for training
# - Learn how to use CAREamics to train N2V
# - Think about pixel noise and noise correlation
#   
# </div>
#

# %% [markdown] tags=[]
# ### Mandatory actions

# %% [markdown] tags=[]
# <div class="alert alert-danger">
# Set your python kernel to <code>05_image_restoration</code>
# </div>

# %% tags=[]
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile

from careamics import CAREamist
from careamics.config import (
    create_n2v_configuration,
)
from careamics.transforms import N2VManipulate

# %matplotlib inline

# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# ## Part 1 Visualize the masking algorithm
#
# In this first part, let's inspect how this pixel masking is done before training a N2V network!
#
# Before feeding patches to the network, a set of transformations, or augmentations, are 
# applied to them. For instance in microscopy, we usually apply random 90 degrees rotations
# or flip the images. In Noise2Void, we apply one more transformation that replace random pixels
# by a value from their surrounding.
#
# In CAREamics, the transformation is called `N2VManipulate`. It has different 
# parameters: `roi_size`, `masked_pixel_percentage` and `strategy`.

# %% tags=[]
# Define a patch size for this exercise
dummy_patch_size = 10

# Define masking parameters
roi_size = 3
masked_pixel_percentage = 10
strategy = 'uniform'

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3><b>Task 1: Explore the N2VManipulate parameters</b></h3>
#
# Can you understand what `roi_size` and `masked_pixel_percentage` do? What can go wrong if they are too small or too high?
#
#
# Run the cell below to observe the effects!
# </div>

# %% tags=[]
# Create a dummy patch
patch = np.arange(dummy_patch_size**2).reshape(dummy_patch_size, dummy_patch_size)

# The pixel manipulator expects a channel dimension, so we need to add it to the patch
patch = patch[np.newaxis]

# Instantiate the pixel manipulator
manipulator = N2VManipulate(
    roi_size=roi_size,
    masked_pixel_percentage=masked_pixel_percentage,
    strategy=strategy,
)

# And apply it
masked_patch, original_patch, mask = manipulator(patch)

# Visualize the masked patch and the mask
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(masked_patch[0])
ax[0].title.set_text("Manipulated patch")
ax[1].imshow(mask[0], cmap="gray")
ax[1].title.set_text("Mask")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Questions: Noise2Void masking strategy</h3>
#
#
# So what's really happening on a technical level? 
#
# In the basic setting N2V algorithm replaces certain pixels with the values from the vicinity
# Other masking stategies also exist, e.g. median, where replacement value is the median off all the pixels inside the region of interest.
#
# Feel free to play around with the ROI size, patch size and masked pixel percentage parameters
#
# </div>
#

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h1><b>Checkpoint 1: N2V masking</b></h1>
# </div>

# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# ## Part 2: Prepare the data
#
# Now that we understand how the masking works, let's train a Noise2Void network! We will
# use a scanning electron microscopy image (SEM).

# %% tags=[]
# Define the paths
root_path = Path("./../data")
root_path = root_path / "denoising-N2V_SEM.unzip/SEM"
assert root_path.exists(), f"Path {root_path} does not exist"

train_images_path = root_path / "train.tif"
validation_images_path = root_path / "validation.tif"

# %% [markdown] tags=[]
# #### Visualize training data

# %% tags=[]
# Load images
train_image = tifffile.imread(train_images_path)
print(f"Train image shape: {train_image.shape}")
plt.imshow(train_image, cmap="gray")

# %% [markdown] tags=[]
# #### Visualize validation data

# %% tags=[]
val_image = tifffile.imread(validation_images_path)
print(f"Validation image shape: {val_image.shape}")
plt.imshow(val_image, cmap="gray")

# %% [markdown] tags=[]
# ## Part 3: Create a configuration
#
# CAREamics can be configured either from a yaml file, or with an explicitly created config object.
# In this note book we will create the config object using helper functions. CAREamics will 
# validate all the parameters and will output explicit error if some parameters or a combination of parameters isn't allowed. It will also provide default values for missing parameters.
#
# The helper function limits the parameters to what is relevant for N2V, here is a break down of these parameters:
#
# - `experiment_name`: name used to identify the experiment
# - `data_type`: data type, in CAREamics it can only be `tiff` or `array` 
# - `axes`: axes of the data, here it would be `YX`. Remember: pytorch and numpy order axes in reverse of what you might be used to. If the data were 3D, the axes would be `ZYX`.
# - `patch_size`: size of the patches used for training
# - `batch_size`: size of each batch
# - `num_epochs`: number of epochs
#
#
# There are also optional parameters, for more fine grained details:
#
# - `use_augmentations`: whether to use augmentations (flip and rotation)
# - `use_n2v2`: whether to use N2V2, a N2V variant (see optional exercise)
# - `n_channels`: the number of channels 
# - `roi_size`: size of the N2V manipulation region (remember that parameter?)
# - `masked_pixel_percentage`: percentage of pixels to mask
# - `logger`: which logger to use
#
#
# Have a look at the [documentation](https://careamics.github.io) to see the full list of parameters and 
# their use!
#

# %% tags=[]
# Create a configuration using the helper function
training_config = create_n2v_configuration(
    experiment_name="dl4mia_n2v_sem",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=128,
    num_epochs=10,
    roi_size=11,
    masked_pixel_percentage=0.2,
    logger="tensorboard"
)

# %% [markdown] tags=[]
# #### Initialize the Model
#
# Let's instantiate the model with the configuration we just created. CAREamist is the main class of the library, it will handle creation of the data pipeline, the model, training and inference methods.

# %% tags=[]
careamist = CAREamist(source=training_config)

# %% [markdown] tags=[]
# ## Part 4: Train
#
# Here, we need to specify the paths to training and validation data. We can point to a folder containing 
# the data or to a single file. If it fits in memory, then CAREamics will load everything and train on it. If it doesn't, then CAREamics will load the data file by file.

# %% tags=[]
careamist.train(train_source=train_images_path, val_source=validation_images_path)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3><b>Task 2: Tensorboard</b></h3>
#
# Remember the configuration? Didn't we set `logger` to `tensorboard`? Then we can visualize the loss curve!
#
# Open Tensorboard in VS Code (check Task 3 in 01_CARE) to monitor training. 
# Logs for this model are stored in the `02_Noise2Void/tb_logs/` folder.
# </div>
#
# <div class="alert alert-block alert-warning"><h3>Question: N2V loss curve</h3>
#
# Do you remember what the loss is in Noise2Void? What is the meaning of the loss curve in that case? Can
# it be easily interpreted?
# </div>
#
# <div class="alert alert-block alert-success"><h1>Checkpoint 2: Training Noise2Void</h1>
# </div>

# %% [markdown] tags=[]
# <hr style="height:2px;">
#
# We trained, but how well did it do?

# %% [markdown] tags=[]
# ## Part 5. Prediction
#
# In order to predict on an image, we also need to specify the path. We also typically need
# to cut the image into patches, predict on each patch and then stitch the patches back together.
#
# To make the process faster, we can choose bigger tiles than the patches used during training. By default CAREamics uses tiled prediction to handle large images. The tile size can be set via the `tile_size` parameter. Tile overlap is computed automatically based on the network architecture.

# %% tags=[]
preds = careamist.predict(source=train_images_path, tile_size=(256, 256))[0]

# %% [markdown] tags=[]
# ### Visualize predictions

# %% tags=[]
# Show the full image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(train_image, cmap="gray")
ax[1].imshow(preds.squeeze(), cmap="gray")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3><b>Question: Inspect the image closely</b></h3>
#
# If you got a good result, try to inspect the image closely. For instance, the default
# window we used for the close-up image:
#
# `y_start` = 200
#
# `y_end` = 450
#
# `x_start` = 600
#
# `x_end` = 850
#
# Do you see anything peculiar in the fine grained details? What could be the reason for that?
# </div>

# %% tags=[]
# Show a close up image
y_start = 200
y_end = 450
x_start = 600
x_end = 850

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(train_image[y_start:y_end, x_start:x_end], cmap="gray")
ax[1].imshow(preds.squeeze()[y_start:y_end, x_start:x_end], cmap="gray")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3><b>Question: Check the residuals</b></h3>
#
# Compute the absolute difference between original and denoised image. What do you see? 
#
# </div>

# %% tags=[]
plt.imshow(preds.squeeze() - train_image, cmap="gray")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3><b>Task 4(Optional): Improving the results</b></h3>
#
# CAREamics configuration won't allow you to use parameters which are clearly wrong. However, there are many parameters that can be tuned to improve the results. Try to play around with the `roi_size` and `masked_pixel_percentage` and see if you can improve the results.
#
# Do the fine-grained structures observed in during the closer look at the image disappear?
#
# </div>

# %% [markdown] tags=[]
# ### How to predict without training?
#
# Here again, CAREamics provides a way to create a CAREamist from a checkpoint only,
# allowing predicting without having to retrain.

# %% tags=[]
# Instantiate a CAREamist from a checkpoint
other_careamist = CAREamist(source="checkpoints/last.ckpt")

# And predict
new_preds = other_careamist.predict(source=train_images_path, tile_size=(256, 256))[0]

# Show the full image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(train_image, cmap="gray")
ax[1].imshow(new_preds.squeeze(), cmap="gray")

# %% tags=[]
train_image[:128, :128].shape

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h1>Checkpoint 3: Prediction</h1>
# </div>
#
# <hr style="height:2px;">
#
# ## Part 6: Exporting the model
#
# Have you heard of the [BioImage Model Zoo](https://bioimage.io/#/)? It provides a format for FAIR AI models and allows
# researchers to exchange and reproduce models. 

# %% tags=[]
# Export model as BMZ
careamist.export_to_bmz(
    path_to_archive="n2v_model.zip",
    input_array=train_image[:128, :128],
    friendly_model_name="SEM_N2V",
    authors= [{"name": "Jane", "affiliation": "Doe University"}],
    general_description='',
    data_description='',
)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h4><b>Task 5: Train N2V(2) on a different dataset</b></h4>
#
# As you remember from the lecture, N2V can only deal with the noise that is pixelwise independent. 
#
# Use these cells to train on a different dataset: Mito Confocal, which has noise that is not pixelwise independent, but is spatially correlated. This will be loaded in the following cell.
#
# In the next cells we'll show you how the result of training a N2V model on this dataset looks.
#
# In the next exercise of the course we'll learn how to deal with this kind of noise! 

# %% tags=[]
mito_path = "./../data/mito-confocal-lowsnr.tif"
mito_image = tifffile.imread(mito_path)

# %% tags=[]
# Configure the model
mito_training_config = create_n2v_configuration(
    experiment_name="dl4mia_n2v2_mito",
    data_type="array",
    axes="SYX", # <-- we are adding S because we have a stack of images
    patch_size=[64, 64],
    batch_size=64,
    num_epochs=10,
    logger="tensorboard",
)

# %% tags=[]
careamist = CAREamist(source=mito_training_config)
careamist.train(
    train_source=mito_image,
    val_percentage=0.1
)

# %% tags=[]
preds = careamist.predict(
    source=mito_image[:1], # <-- we predict on a small subset
    data_type="array",
    tile_size=(64, 64),
)[0]

# %% [markdown] tags=[]
# In the following cell, look closely at the denoising result of applying N2V to data with spatially correlated noise. Zoom in and see if you can find the horizontal artifacts.

# %% tags=[]
vmin = np.percentile(mito_image, 1)
vmax = np.percentile(mito_image, 99)

y_start = 0
y_end = 1024
x_start = 0
x_end = 1024

# Feel free to play around with the visualization
_, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(preds[0, 0, 600:700, 300:400], vmin=vmin, vmax=vmax)
ax[0].title.set_text("Predicted")
ax[1].imshow(mito_image[0, 600:700, 300:400], vmin=vmin, vmax=vmax)
ax[1].title.set_text("Original")

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h1>Checkpoint 4: Dealing with artifacts</h1>
# </div>

# %% [markdown] tags=[]
# <hr style="height:2px;"><div class="alert alert-block alert-warning"><h3>Take away questions</h3>
#
# - Which is the best saved checkpoint for Noise2Void, the one at the end of the training or the one with lowest validation loss?
#
# - Is validation useful in Noise2Void?
#
# - We predicted on the same image we trained on, is that a good idea?
#
# - Can you reuse the model on another image?
#
# - Can you train on images with multiple channels? RGB images? Biological channels (GFP, RFP, DAPI)?
#
# - N2V training is unsupervised, how can you be sure that the training worked and is not hallucinating?
# </div>
#

# %% [markdown] tags=[]
# <hr style="height:2px;"><div class="alert alert-block alert-success"><h1>End of the exercise</h1>
# </div>
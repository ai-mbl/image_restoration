# %% [markdown]
"""
<hr style="height:2px;">

# Train your first CARE model (supervised)

In this first example we will train a CARE model for a 2D denoising and upsampling task, where corresponding pairs of low and high signal-to-noise ratio (SNR) images of cells are available. Here the high SNR images are acquisitions of Human U2OS cells taken from the [Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC006/) and the low SNR images were created by synthetically adding *strong read-out and shot-noise* and applying *pixel binning* of 2x2, thus mimicking acquisitions at a very low light level.

![](nb_material/denoising_binning_overview.png)


For CARE, image pairs should be registered, which in practice is best achieved by acquiring both stacks _interleaved_, i.e. as different channels that correspond to the different exposure/laser settings.

Since the image pairs were synthetically created in this example, they are already aligned perfectly. Note that when working with real paired acquisitions, the low and high SNR images are not pixel-perfect aligned so typically need to be co-registered before training a CARE model.

To train a denoising network, we will use the [CSBDeep Repo](https://github.com/CSBDeep/CSBDeep). This notebook has a very similar structure to the examples you can find there.
More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.

This part will not have any coding tasks, but go through each cell and try to understand what's going on - it will help you in the next part! We also put some questions along the way. For some of them you might need to dig a bit deeper.

<div class="alert alert-danger">
Set your python kernel to <code>03_image_restoration_part1</code>
</div>
"""

# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import os
import numpy as np
from csbdeep.data import (
    RawData,
    create_patches,
    no_background_patches,
    norm_percentiles,
    sample_percentiles,
)
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import CARE, Config
from csbdeep.utils import (
    Path,
    axes_dict,
    normalize,
    plot_history,
    plot_some,
)
from csbdeep.utils.tf import limit_gpu_memory

# %matplotlib inline
# %load_ext tensorboard
# %config InlineBackend.figure_format = 'retina'
from tifffile import imread

# %% [markdown]
"""
<hr style="height:2px;">

## Part 1: Training Data Generation
Network training usually happens on batches of smaller sized images than the ones recorded on a microscopy. In this first part of the exercise, we will load all of the image data and chop it into smaller pieces, a.k.a. patches.

### Look at example data

During setup, we downloaded some example data, consisting of low-SNR and high-SNR 3D images of Tribolium.
Note that `GT` stands for ground truth and represents high signal-to-noise ratio (SNR) stacks.
"""
# %%
assert os.path.exists("data/U2OS")
# %% [markdown]
"""
As we can see, the data set is already split into a **train** and **test** set, each containing (synthetically generated) low SNR ("low") and corresponding high SNR ("GT") images.

Let's look at an example pair of training images:
"""
# %%
y = imread("data/U2OS/train/GT/img_0010.tif")
x = imread("data/U2OS/train/low/img_0010.tif")
print("GT image size =", x.shape)
print("low-SNR image size =", y.shape)

# %%
plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.imshow(x, cmap="magma")
plt.colorbar()
plt.title("low")
plt.subplot(1, 2, 2)
plt.imshow(y, cmap="magma")
plt.colorbar()
plt.title("high")
plt.show()
# %% [markdown]
"""
### Generate training data for CARE

We first need to create a `RawData` object, which defines how to get the pairs of low/high SNR stacks and the semantics of each axis (e.g. which one is considered a color channel, etc.). In general the names for the axes are:

X: columns, Y: rows, Z: planes, C: channels, T: frames/time, (S: samples/images)

Here we have two folders "low" and "GT", where corresponding low and high-SNR stacks are TIFF images with identical filenames.

For this case, we can simply use `RawData.from_folder` and set `axes = 'YX'` to indicate the semantic order of the image axes, i.e. we have two-dimensional images in standard xy layout.
"""
# %%
raw_data = RawData.from_folder(
    basepath="data/U2OS/train",
    source_dirs=["low"],
    target_dir="GT",
    axes="YX",
)
# %% [markdown]
"""
From corresponding images, we now generate some 2D patches to use for training.

As a general rule, use a *patch size* that is a power of two along all axes, or at least divisible by 8.  Typically, you should use more patches the more trainings images you have.

An important aspect is *data normalization*, i.e. the rescaling of corresponding patches to a dynamic range of ~ (0,1). By default, this is automatically provided via percentile normalization, which can be adapted if needed.

By default, patches are sampled from *non-background regions* (i.e. that are above a relative threshold). We will disable this for the current example as most image regions already contain foreground pixels and thus set the threshold to 0. See the documentation of `create_patches` for details.

Note that returned values `(X, Y, XY_axes)` by `create_patches` are not to be confused with the image axes X and Y. By convention, the variable name X (or x) refers to an input variable for a machine learning model, whereas Y (or y) indicates an output variable.
"""
# %%
X, Y, XY_axes = create_patches(
    raw_data=raw_data,
    patch_size=(128, 128),
    patch_filter=no_background_patches(0),
    n_patches_per_image=2,
    save_file="data/U2OS/my_training_data.npz",
)

assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

# %% [markdown]
"""
### Show

This shows some of the generated patch pairs (odd rows: *source*, even rows: *target*).
"""
# %%
for i in range(2):
    plt.figure(figsize=(16, 4))
    sl = slice(8 * i, 8 * (i + 1)), 0
    plot_some(
        X[sl], Y[sl], title_list=[np.arange(sl[0].start, sl[0].stop)]
    )  # convenience function provided by CSB Deep
    plt.show()
# %% [markdown]
"""
<div class="alert alert-block alert-warning"><h3>
    Questions:</h3>
    <ol>
        <li>Where is the training data located?</li>
        <li>How is the data organized to identify the pairs of HR and LR images?</li>
    </ol>
</div>

<hr style="height:2px;">

## Part 2: Training the network


### Load Training data

Load the patches generated in part 1, use 10% as validation data.
"""
# %%
(X, Y), (X_val, Y_val), axes = load_training_data(
    "data/U2OS/my_training_data.npz", validation_split=0.1, verbose=True
)

c = axes_dict(axes)["C"]
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

# %%
plt.figure(figsize=(12, 5))
plot_some(X_val[:5], Y_val[:5])
plt.suptitle("5 example validation patches (top row: source, bottom row: target)")
plt.show()

# %% [markdown]
"""
### Configure the CARE model
Before we construct the actual CARE model, we have to define its configuration via a `Config` object, which includes
* parameters of the underlying neural network,
* the learning rate,
* the number of parameter updates per epoch,
* the loss function, and
* whether the model is probabilistic or not.

![](nb_material/carenet.png)

The defaults should be sensible in many cases, so a change should only be necessary if the training process fails.

<span style="color:red;font-weight:bold;">Important</span>: Note that for this notebook we use a very small number of update steps for immediate feedback, whereas the number of epochs and steps per epoch should be increased considerably (e.g. `train_steps_per_epoch=400`, `train_epochs=100`) to obtain a well-trained model.
"""
# %%
config = Config(
    axes,
    n_channel_in,
    n_channel_out,
    train_batch_size=8,
    train_steps_per_epoch=40,
    train_epochs=20,
)
vars(config)
# %% [markdown]
"""
We now create a CARE model with the chosen configuration:
"""
# %%
model = CARE(config, "my_CARE_model", basedir="models")
# %% [markdown]
"""
We can get a summary of all the layers in the model and the number of parameters:
"""
# %%
model.keras_model.summary()
# %% [markdown]
"""
### Training

Training the model will likely take some time. We recommend to monitor the progress with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard), which allows you to inspect the losses during training.
Furthermore, you can look at the predictions for some of the validation images, which can be helpful to recognize problems early on.

We can start tensorboard within the notebook.

Alternatively, you can launch the notebook in an independent tab by changing the `%` to `!`
<div class="alert alert-danger">
If you're using ssh add <code>--host &lt;hostname&gt;</code> to the command:
<code>! tensorboard --logdir models --host &lt;hostname&gt;</code> where <code>&lt;hostname&gt;</code> is the thing that ends in amazonaws.com.
</div>
"""
# %%
# %tensorboard --logdir models
# %%
history = model.train(X, Y, validation_data=(X_val, Y_val))
# %% [markdown]
"""
Plot final training history (available in TensorBoard during training):
"""
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
plot_some(X_val[:5], Y_val[:5], _P, pmax=99.5)
plt.suptitle(
    "5 example validation patches\n"
    "top row: input (source),  "
    "middle row: target (ground truth),  "
    "bottom row: predicted from source"
)
# %% [markdown]
"""
<div class="alert alert-block alert-warning"><h3>
    Questions:</h3>
    <ol>
        <li>Where are trained models stored? What models are being stored, how do they differ?</li>
        <li>How does the name of the saved models get specified?</li>
        <li>How can you influence the number of training steps per epoch? What did you use?</li>
    </ol>
</div>

<hr style="height:2px;">

## Part 3: Prediction

Plot the test stack pair and define its image axes, which will be needed later for CARE prediction.
"""
# %%
y_test = imread("data/U2OS/test/GT/img_0010.tif")
x_test = imread("data/U2OS/test/low/img_0010.tif")

axes = "YX"
print("image size =", x_test.shape)
print("image axes =", axes)

plt.figure(figsize=(16, 10))
plot_some(np.stack([x_test, y_test]), title_list=[["low", "high"]])


# %% [markdown]
"""
### Load CARE model

Load trained model (located in base directory `models` with name `my_CARE_model`) from disk.
The configuration was saved during training and is automatically loaded when `CARE` is initialized with `config=None`.
"""
# %%
model = CARE(config=None, name="my_CARE_model", basedir="models")
# %% [markdown]
"""
### Apply CARE network to raw image
Predict the restored image (image will be successively split into smaller tiles if there are memory issues).
"""
# %%
# %%time
restored = model.predict(x_test, axes)

# %% [markdown]
"""
### Save restored image

Save the restored image stack as a ImageJ-compatible TIFF image, i.e. the image can be opened in ImageJ/Fiji with correct axes semantics.
"""
# %%
Path("results").mkdir(exist_ok=True)
save_tiff_imagej_compatible("results/%s_img_0010.tif" % model.name, restored, axes)

# %% [markdown]
"""
### Visualize results
Plot the test stack pair and the predicted restored stack (middle).
"""

# %%
plt.figure(figsize=(15, 10))
plot_some(
    np.stack([x_test, restored, y_test]),
    title_list=[["low", "CARE", "GT"]],
    pmin=2,
    pmax=99.8,
)

plt.figure(figsize=(10, 5))
for _x, _name in zip((x_test, restored, y_test), ("low", "CARE", "GT")):
    plt.plot(normalize(_x, 1, 99.7)[180], label=_name, lw=2)
plt.legend()
plt.show()

# %% [markdown]
"""
<hr style="height:2px;">
<div class="alert alert-block alert-success"><h1>
    Congratulations!</h1>
    <p>
    <b>You have reached the first checkpoint of this exercise! Please mark your progress in the course chat!</b>
    </p>
</div>
"""

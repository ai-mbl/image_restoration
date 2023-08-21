# %% [markdown]
"""
<hr style="height:2px;">

# Train Probabilistic Noise2Void

Probabilistic Noise2Void, just as N2V, allows training from single noisy images.

In order to get some additional quality squeezed out of your noisy input data, PN2V employs an additional noise model which can either be measured directly at your microscope or approximated by a process called ‘bootstrapping’.
Below we will give you a noise model for the first network to train and then bootstrap one, so you can apply PN2V to your own data if you'd like.

Note: The PN2V implementation is written in pytorch, not Keras/TF.

Note: PN2V experienced multiple updates regarding noise model representations. Hence, the [original PN2V repository](https://github.com/juglab/pn2v) is not any more the one we suggest to use (despite it of course working just as described in the original publication). So here we use the [PPN2V repo](https://github.com/juglab/PPN2V) which you installed during setup.

<div class="alert alert-danger">
Set your python kernel to <code>03_image_restoration_bonus</code>
</div>
<div class="alert alert-danger">
Make sure your previous notebook is shutdown to avoid running into GPU out-of-memory problems.
</div>

"""
# %%
import warnings

warnings.filterwarnings("ignore")
import torch

dtype = torch.float
device = torch.device("cuda:0")
from torch.distributions import normal
import matplotlib.pyplot as plt, numpy as np, pickle
from scipy.stats import norm
from tifffile import imread
import sys
import os
import urllib
import zipfile

# %%
from ppn2v.pn2v import histNoiseModel, gaussianMixtureNoiseModel
from ppn2v.pn2v.utils import plotProbabilityDistribution, PSNR
from ppn2v.unet.model import UNet
from ppn2v.pn2v import training, prediction

# %% [markdown]
"""
## Data Preperation

Here we use a dataset of 2D images of fluorescently labeled membranes of Convallaria (lilly of the valley) acquired with a spinning disk microscope.
All 100 recorded images (1024×1024 pixels) show the same region of interest and only differ in their noise.
"""

# %%
# Check that data download was successful
assert os.path.exists("data/Convallaria_diaphragm")


# %%
path = "data/Convallaria_diaphragm/"
data_name = "convallaria"  # Name of the noise model
calibration_fn = "20190726_tl_50um_500msec_wf_130EM_FD.tif"
noisy_fn = "20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif"
noisy_imgs = imread(path + noisy_fn)
calibration_imgs = imread(path + calibration_fn)

# %% [markdown]
"""
This notebook has a total of four options to generate a noise model for PN2V. You can pick which one you would like to use (and ignore the tasks in the options you don't wanna use)!

There are two types of noise models for PN2V: creating a histogram of the noisy pixels based on the averaged GT or using a gaussian mixture model (GMM).
For both we need to provide a clean signal as groundtruth. For the dataset we have here we have calibration data available so you can choose between using the calibration data or bootstrapping the model by training a N2V network.
"""
# %%
n_gaussian = 3  # Number of gaussians to use for Gaussian Mixture Model
n_coeff = 2  # No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components.

# %% [markdown]
"""
<hr style="height:2px;">

## Choice 1: Generate a Noise Model using Calibration Data
The noise model is a characteristic of your camera. The downloaded data folder contains a set of calibration images (For the Convallaria dataset, it is ```20190726_tl_50um_500msec_wf_130EM_FD.tif``` and the data to be denoised is named ```20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif```). We can either bin the noisy - GT pairs (obtained from noisy calibration images) as a 2-D histogram or fit a GMM distribution to obtain a smooth, parametric description of the noise model.

We will use pairs of noisy calibration observations $x_i$ and clean signal $s_i$ (created by averaging these noisy, calibration images) to estimate the conditional distribution $p(x_i|s_i)$. Histogram-based and Gaussian Mixture Model-based noise models are generated and saved.
"""
# %%
name_hist_noise_model_cal = "_".join(["HistNoiseModel", data_name, "calibration"])
name_gmm_noise_model_cal = "_".join(
    ["GMMNoiseModel", data_name, str(n_gaussian), str(n_coeff), "calibration"]
)
# %% [markdown]
"""
---
<div class="alert alert-block alert-info"><h4>
    TASK 4.1</h4>
    <p>

The calibration data contains 100 images of a static sample. Estimate the clean signal by averaging all the images.
    </p>
</div>
"""
# %%
###TODO###
# Average the images in `calibration_imgs`
signal_cal = ...  # TODO


# %% tags = ["solution"]
# Average the images in `calibration_imgs`
signal_cal = np.mean(calibration_imgs[:, ...], axis=0)[np.newaxis, ...]
# %% [markdown]
"""
Let's visualize a single image from the observation array alongside the average to see how the raw data compares to the pseudo ground truth signal.
"""
# %% [markdown]
"""
---
"""
# %%
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 2)
plt.title(label="average (ground truth)")
plt.imshow(signal_cal[0], cmap="gray")
plt.subplot(1, 2, 1)
plt.title(label="single raw image")
plt.imshow(calibration_imgs[0], cmap="gray")
plt.show()


# %%
# The subsequent code expects the signal array to have a dimension for the samples
if signal_cal.shape == calibration_imgs.shape[1:]:
    signal_cal = signal_cal[np.newaxis, ...]

# %% [markdown]
"""
There are two ways of generating a noise model for PN2V: creating a histogram of the noisy pixels based on the averaged GT or using a gaussian mixture model (GMM). You can pick which one you wanna use!

<hr style="height:1px;">

### Choice 1A: Creating the Histogram Noise Model
Using the raw pixels $x_i$, and our averaged GT $s_i$, we are now learning a histogram based noise model. It describes the distribution $p(x_i|s_i)$ for each $s_i$.

---
<div class="alert alert-block alert-info"><h4>
    TASK 4.2</h4>
    <p>
        Look at the docstring for <tt>createHistogram</tt> and use it to create a histogram based on the calibration data using the clean signal you created by averaging as groundtruth.    </p>
</div>
"""
# %%
# ?histNoiseModel.createHistogram

# %%
###TODO###
# Define the parameters for the histogram creation
bins = 256
# Values falling outside the range [min_val, max_val] are not included in the histogram, so the values in the images you want to denoise should fall within that range
min_val = ...  # TODO
max_val = ...  # TODO
# Create the histogram
histogram_cal = histNoiseModel.createHistogram(bins, ...)  # TODO

# %% tags = ["solution"]
# Define the parameters for the histogram creation
bins = 256
# Values falling outside the range [min_val, max_val] are not included in the histogram, so the values in the images you want to denoise should fall within that range
min_val = 234  # np.min(noisy_imgs)
max_val = 7402  # np.max(noisy_imgs)
print("min:", min_val, ", max:", max_val)
# Create the histogram
histogram_cal = histNoiseModel.createHistogram(
    bins, min_val, max_val, calibration_imgs, signal_cal
)
# %% [markdown]
"""
---
"""
# %%
# Saving histogram to disk.
np.save(path + name_hist_noise_model_cal + ".npy", histogram_cal)
histogramFD_cal = histogram_cal[0]

# %%
# Let's look at the histogram-based noise model.
plt.xlabel("Observation Bin")
plt.ylabel("Signal Bin")
plt.imshow(histogramFD_cal**0.25, cmap="gray")
plt.show()

# %% [markdown]
"""
<hr style="height:1px;">

### Choice 1B: Creating the GMM noise model
Using the raw pixels $x_i$, and our averaged GT $s_i$, we are now learning a GMM based noise model. It describes the distribution $p(x_i|s_i)$ for each $s_i$.
"""
# %%
min_signal = np.min(signal_cal)
max_signal = np.max(signal_cal)
print("Minimum Signal Intensity is", min_signal)
print("Maximum Signal Intensity is", max_signal)

# %% [markdown]
"""
Iterating the noise model training for `n_epoch=2000` and `batchSize=250000` works the best for `Convallaria` dataset.
"""
# %%
# ?gaussianMixtureNoiseModel.GaussianMixtureNoiseModel
# %%
gmm_noise_model_cal = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(
    min_signal=min_signal,
    max_signal=max_signal,
    path=path,
    weight=None,
    n_gaussian=n_gaussian,
    n_coeff=n_coeff,
    min_sigma=50,
    device=device,
)
# %%
gmm_noise_model_cal.train(
    signal_cal,
    calibration_imgs,
    batchSize=250000,
    n_epochs=2000,
    learning_rate=0.1,
    name=name_gmm_noise_model_cal,
)
# %% [markdown]
"""
<hr style="height:1px;">

### Visualizing the Histogram-based and GMM-based noise models

This only works if you generated both a histogram (Choice 1A) and GMM-based (Choice 1B) noise model
"""
# %%
plotProbabilityDistribution(
    signalBinIndex=170,
    histogram=histogramFD_cal,
    gaussianMixtureNoiseModel=gmm_noise_model_cal,
    min_signal=min_val,
    max_signal=max_val,
    n_bin=bins,
    device=device,
)
# %% [markdown]
"""
<hr style="height:2px;">

## Choice 2: Generate a Noise Model by Bootstrapping

Here we bootstrap a suitable histogram noise model and a GMM noise model after denoising the noisy images with Noise2Void and then using these denoised images as pseudo GT.
So first, we need to train a N2V model (now with pytorch) to estimate the conditional distribution $p(x_i|s_i)$. No additional calibration data is used for bootstrapping (so no need to use `calibration_imgs` or `singal_cal` again).
"""
# %%
model_name = data_name + "_n2v"
name_hist_noise_model_bootstrap = "_".join(["HistNoiseModel", data_name, "bootstrap"])
name_gmm_noise_model_bootstrap = "_".join(
    ["GMMNoiseModel", data_name, str(n_gaussian), str(n_coeff), "bootstrap"]
)

# %%
# Configure the Noise2Void network
n2v_net = UNet(1, depth=3)

# %%
# Prepare training+validation data
train_data = noisy_imgs[:-5].copy()
val_data = noisy_imgs[-5:].copy()
np.random.shuffle(train_data)
np.random.shuffle(val_data)

# %%
train_history, val_history = training.trainNetwork(
    net=n2v_net,
    trainData=train_data,
    valData=val_data,
    postfix=model_name,
    directory=path,
    noiseModel=None,
    device=device,
    numOfEpochs=200,
    stepsPerEpoch=10,
    virtualBatchSize=20,
    batchSize=1,
    learningRate=1e-3,
)

# %%
# Let's look at the training and validation loss
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(val_history, label="validation loss")
plt.plot(train_history, label="training loss")
plt.legend()
plt.show()

# %%
# We now run the N2V model to create pseudo groundtruth.
n2v_result_imgs = []
n2v_input_imgs = []

for index in range(noisy_imgs.shape[0]):
    im = noisy_imgs[index]
    # We are using tiling to fit the image into memory
    # If you get an error try a smaller patch size (ps)
    n2v_pred = prediction.tiledPredict(
        im, n2v_net, ps=256, overlap=48, device=device, noiseModel=None
    )
    n2v_result_imgs.append(n2v_pred)
    n2v_input_imgs.append(im)
    if index % 10 == 0:
        print("image:", index)

# %%
# In bootstrap mode, we estimate pseudo GT by using N2V denoised images.
signal_bootstrap = np.array(n2v_result_imgs)
# Let's look the raw data and our pseudo ground truth signal
print(signal_bootstrap.shape)
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 2)
plt.title(label="pseudo GT (generated by N2V denoising)")
plt.imshow(signal_bootstrap[0], cmap="gray")
plt.subplot(2, 2, 4)
plt.imshow(signal_bootstrap[0, -128:, -128:], cmap="gray")
plt.subplot(2, 2, 1)
plt.title(label="single raw image")
plt.imshow(noisy_imgs[0], cmap="gray")
plt.subplot(2, 2, 3)
plt.imshow(noisy_imgs[0, -128:, -128:], cmap="gray")
plt.show()
# %% [markdown]
"""
Now that we have pseudoGT, you can pick again between a histogram based noise model and a GMM noise model

<hr style="height:1px;">

### Choice 2A: Creating the Histogram Noise Model

---
<div class="alert alert-block alert-info"><h4>
    TASK 4.3</h4>
    <p>
    If you've already done Task 4.2, this is very similar!
        Look at the docstring for <tt>createHistogram</tt> and use it to create a histogram using the bootstraped signal you created from the N2V predictions.
    </p>
</div>
"""
# %%
# ?histNoiseModel.createHistogram
# %%
###TODO###
# Define the parameters for the histogram creation
bins = 256
# Values falling outside the range [min_val, max_val] are not included in the histogram, so the values in the images you want to denoise should fall within that range
min_val = ...  # TODO
max_val = ...  # TODO
# Create the histogram
histogram_bootstrap = histNoiseModel.createHistogram(bins, ...)  # TODO
# %% tags=["solution"]
# Define the parameters for the histogram creation
bins = 256
# Values falling outside the range [min_val, max_val] are not included in the histogram, so the values in the images you want to denoise should fall within that range
min_val = np.min(noisy_imgs)
max_val = np.max(noisy_imgs)
# Create the histogram
histogram_bootstrap = histNoiseModel.createHistogram(
    bins, min_val, max_val, noisy_imgs, signal_bootstrap
)
# %% [markdown]
"""
---
"""
# %%
# Saving histogram to disk.
np.save(path + name_hist_noise_model_bootstrap + ".npy", histogram_bootstrap)
histogramFD_bootstrap = histogram_bootstrap[0]
# %%
# Let's look at the histogram-based noise model
plt.xlabel("Observation Bin")
plt.ylabel("Signal Bin")
plt.imshow(histogramFD_bootstrap**0.25, cmap="gray")
plt.show()

# %% [markdown]
"""
<hr style="height:1px;">

### Choice 2B: Creating the GMM noise model
Using the raw pixels $x_i$, and our averaged GT $s_i$, we are now learning a GMM based noise model. It describes the distribution $p(x_i|s_i)$ for each $s_i$.
"""
# %%
min_signal = np.percentile(signal_bootstrap, 0.5)
max_signal = np.percentile(signal_bootstrap, 99.5)
print("Minimum Signal Intensity is", min_signal)
print("Maximum Signal Intensity is", max_signal)
# %% [markdown]
"""
Iterating the noise model training for `n_epoch=2000` and `batchSize=250000` works the best for `Convallaria` dataset.
"""
# %%
gmm_noise_model_bootstrap = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(
    min_signal=min_signal,
    max_signal=max_signal,
    path=path,
    weight=None,
    n_gaussian=n_gaussian,
    n_coeff=n_coeff,
    device=device,
    min_sigma=50,
)
# %%
gmm_noise_model_bootstrap.train(
    signal_bootstrap,
    noisy_imgs,
    batchSize=250000,
    n_epochs=2000,
    learning_rate=0.1,
    name=name_gmm_noise_model_bootstrap,
    lowerClip=0.5,
    upperClip=99.5,
)
# %% [markdown]
"""
### Visualizing the Histogram-based and GMM-based noise models

This only works if you generated both a histogram (Choice 2A) and GMM-based (Choice 2B) noise model
"""
# %%
plotProbabilityDistribution(
    signalBinIndex=170,
    histogram=histogramFD_bootstrap,
    gaussianMixtureNoiseModel=gmm_noise_model_bootstrap,
    min_signal=min_val,
    max_signal=max_val,
    n_bin=bins,
    device=device,
)
# %% [markdown]
"""
<hr style="height:2px;">

## PN2V Training

---
<div class="alert alert-block alert-info"><h4>
    TASK 4.4</h4>
    <p>
    Adapt to use the noise model of your choice here to then train PN2V with.
    </p>
</div>
"""
# %%
###TODO###
noise_model_type = "gmm"  # pick: "hist" or "gmm"
noise_model_data = "bootstrap"  # pick: "calibration" or "bootstrap"

# %% tags = ["solution"]
if noise_model_type == "hist":
    noise_model_name = "_".join(["HistNoiseModel", data_name, noise_model_data])
    histogram = np.load(path + noise_model_name + ".npy")
    noise_model = histNoiseModel.NoiseModel(histogram, device=device)
elif noise_model_type == "gmm":
    noise_model_name = "_".join(
        ["GMMNoiseModel", data_name, str(n_gaussian), str(n_coeff), noise_model_data]
    )
    params = np.load(path + noise_model_name + ".npz")
    noise_model = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(
        params=params, device=device
    )
# %% [markdown]
"""
---
"""
# %%
# Create a network with 800 output channels that are interpreted as samples from the prior.
pn2v_net = UNet(800, depth=3)
# %%
# Start training.
trainHist, valHist = training.trainNetwork(
    net=pn2v_net,
    trainData=train_data,
    valData=val_data,
    postfix=noise_model_name,
    directory=path,
    noiseModel=noise_model,
    device=device,
    numOfEpochs=200,
    stepsPerEpoch=5,
    virtualBatchSize=20,
    batchSize=1,
    learningRate=1e-3,
)
# %% [markdown]
"""
<hr style="height:2px;">

## PN2V Evaluation
"""
# %%
test_data = noisy_imgs[
    :, :512, :512
]  # We are loading only a sub image to speed up computation

# %%
# We estimate the ground truth by averaging.
test_data_gt = np.mean(test_data[:, ...], axis=0)[np.newaxis, ...]

# %%
pn2v_net = torch.load(path + "/last_" + noise_model_name + ".net")

# %%
# Now we are processing data and calculating PSNR values.
mmse_psnrs = []
prior_psnrs = []
input_psnrs = []
result_ims = []
input_ims = []

# We iterate over all test images.
for index in range(test_data.shape[0]):
    im = test_data[index]
    gt = test_data_gt[0]  # The ground truth is the same for all images

    # We are using tiling to fit the image into memory
    # If you get an error try a smaller patch size (ps)
    means, mse_est = prediction.tiledPredict(
        im, pn2v_net, ps=192, overlap=48, device=device, noiseModel=noise_model
    )

    result_ims.append(mse_est)
    input_ims.append(im)

    range_psnr = np.max(gt) - np.min(gt)
    psnr = PSNR(gt, mse_est, range_psnr)
    psnr_prior = PSNR(gt, means, range_psnr)
    input_psnr = PSNR(gt, im, range_psnr)
    mmse_psnrs.append(psnr)
    prior_psnrs.append(psnr_prior)
    input_psnrs.append(input_psnr)

    print("image:", index)
    print("PSNR input", input_psnr)
    print("PSNR prior", psnr_prior)  # Without info from masked pixel
    print("PSNR mse", psnr)  # MMSE estimate using the masked pixel
    print("-----------------------------------")

# %%
# ?prediction.tiledPredict

# %%
# We display the results for the last test image
vmi = np.percentile(gt, 0.01)
vma = np.percentile(gt, 99)

plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.title(label="Input Image")
plt.imshow(im, vmax=vma, vmin=vmi, cmap="magma")

plt.subplot(1, 3, 2)
plt.title(label="Avg. Prior")
plt.imshow(means, vmax=vma, vmin=vmi, cmap="magma")

plt.subplot(1, 3, 3)
plt.title(label="PN2V-MMSE estimate")
plt.imshow(mse_est, vmax=vma, vmin=vmi, cmap="magma")
plt.show()

plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.title(label="Input Image")
plt.imshow(im[100:200, 150:250], vmax=vma, vmin=vmi, cmap="magma")
plt.axhline(y=50, linewidth=3, color="white", alpha=0.5, ls="--")

plt.subplot(1, 3, 2)
plt.title(label="Avg. Prior")
plt.imshow(means[100:200, 150:250], vmax=vma, vmin=vmi, cmap="magma")
plt.axhline(y=50, linewidth=3, color="white", alpha=0.5, ls="--")

plt.subplot(1, 3, 3)
plt.title(label="PN2V-MMSE estimate")
plt.imshow(mse_est[100:200, 150:250], vmax=vma, vmin=vmi, cmap="magma")
plt.axhline(y=50, linewidth=3, color="white", alpha=0.5, ls="--")


plt.figure(figsize=(15, 5))
plt.plot(im[150, 150:250], label="Input Image")
plt.plot(means[150, 150:250], label="Avg. Prior")
plt.plot(mse_est[150, 150:250], label="PN2V-MMSE estimate")
plt.plot(gt[150, 150:250], label="Pseudo GT by averaging")
plt.legend()

plt.show()
print(
    "Avg PSNR Prior:",
    np.mean(np.array(prior_psnrs)),
    "+-(2SEM)",
    2 * np.std(np.array(prior_psnrs)) / np.sqrt(float(len(prior_psnrs))),
)
print(
    "Avg PSNR MMSE:",
    np.mean(np.array(mmse_psnrs)),
    "+-(2SEM)",
    2 * np.std(np.array(mmse_psnrs)) / np.sqrt(float(len(mmse_psnrs))),
)

# %% [markdown]
"""
---
---
<div class="alert alert-block alert-info"><h4>
    TASK 4.5</h4>
    <p>
    Try PN2V for your own data! You probably don't have calibration data, but with the bootstrapping method you don't need any!
    </p>
</div>

---

<hr style="height:2px;">
<div class="alert alert-block alert-success"><h1>
    Congratulations!</h1>
    <p>
    <b>You have completed the bonus exercise!</b>
    </p>
</div>
"""

# Exercise 3: Image Restoration

In this exercise you will get to try out some of the image restoration techniques that you just learned about in the lecture.

Start by setting up all the environments and downloading the example data: Open a terminal and run `source setup.sh`.

In the first part of the exercise `exercise1.ipynb` you will use paired images with high and low signal to noise ratios to train a supervised CARE network. The second part (`exercise2.ipynb`) you will train a Noise2Noise network with multiple SEM acquisitions of the same sample at various noise levels. And in part 3 (`exercise3.ipynb`) you can train a Noise2Void network on your own data (if you'd like).

All exercise notebooks are closely modeled after example notebooks that were provided as part of the respective repositories by their authors. This won't always be the case but think of this exercise as a good example of the situation you'll find yourself in if you find a deep learning method "in the wild" that you would like to try out yourself.

If you have extra time in the end check out `exercise_bonus1.ipynb` if you're interested in Probabilistic Noise2Void or `exercise_bonus2.md` for DivNoising where you will go one step further by cloning the repo yourself, setting up your own environment and running an example notebook from the repo.








## Task Overview for TAs

### Exercise1
#### Questions:
- where is the training data located?
- how is the data organized to identify the pairs of HR and LR images?
#### Questions:
- Where are the trained models stored? What models are being stored, how do they differ?
- How does the name of the saved models get specified?
- How can you influence the number of training steps per epoch? What did you use?

-> CHECKPOINT1

### Exercise2
#### Task 2.1:
- Crop image for visualization to get a feeling for what the data looks like
#### Task 2.2:
- Pick input and target images for Noise2Noise
#### Task 2.3:
- create raw data object
#### Task 2.4:
- write function that applies the model to one of the images
#### Task 2.5:
- play around by tweaking setup and/or train network for all scan times
-> CHECKPOINT2

### Exercise3
#### Task 3.1:
- use your own data for N2V
#### Task 3.2:
- configure N2V model
#### Task 3.3:
- measure performance (if high SNR image available)
-> CHECKPOINT3

### Bonus Exercise 1
#### Task 4.1
- estimate clean signal from calibration data
#### Task 4.2
- create histogram from calibration data
#### Task 4.3
- create histogram from bootstrapped signal
#### Task 4.4
- train PN2V model
#### Task 4.5
- try PN2V model for your own data

### Bonus Exercise 2
- run an example notebook from DivNoising repo


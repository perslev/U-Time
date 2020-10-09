# U-Sleep

This repository stores code for training and evaluating the U-Sleep sleep staging model. The U-Sleep repository builds upon and significantly extends our [U-Time](https://github.com/perslev/U-Time) repository, published in [[1]](#[1]-u-time). In the following, we will use the term *U-Sleep* to denote the recilient high frequency sleep staging model, and *u-time* to denote this repository of code used to train and evaluate U-Sleep.

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Full Reproduction of U-Sleep](#full-reproduction-of-u-sleep)
- [U-Time and U-Sleep Citations](#citations)


## Overview
TODO.

## System Requirements
**Hardware Requirements:** TODO

**Software Requirements:** TODO

## Installation Guide
On a Linux machine with at least 1 CUDA enabled GPU available and `Anaconda` installed, run the following two commands to create your `u-sleep` environment and setup the U-Time software package:

```console
$ conda env create --file u-sleep/environment.yaml
$ conda activate u-sleep
(u-sleep) $ pip install -e u-sleep
```

This installation process may take up to 10 minutes to complete.

## Demo
In this following we will demonstrate how to launch a short training session of U-Sleep on a significantly limited subset of the datasets used in [[2]](#U-Sleep).

First, we create a project directory that will store all of our data for this demo. The `ut init_project` command will create a folder and populate it with a set of default hyperparameter values:

```console
(u-sleep) $ ut init_project --name demo --model usleep_demo
```

Entering the newly created project directory we will find a folder storing hyperparameters:

```console
(u-sleep) $ cd demo
(u-sleep) $ ls
> hyperparameters
```

We will download 10 PSG studies from the public sleep databases [Sleep-EDF](https://doi.org/10.13026/C2X676) and [DCSM](https://sid.erda.dk/wsgi-bin/ls.py?share_id=fUH3xbOXv8) using the `ut fetch` command:

```console
(u-sleep) $ ut fetch --dataset sedf_sc --out_dir data/sedf_sc --N_first 10
(u-sleep) $ ut fetch --dataset dcsm --out_dir data/dcsm --N_first 10
```

The raw data has now been downloaded. We split each dataset into train ()validation/test splits using the `ut cv_split` command:

```console
(u-sleep) $ ut cv_split --data_dir data/sedf_sc/ \
						--subject_dir_pattern 'SC*' \
						--CV 1 \
						--validation_fraction 0.10 \
						--test_fraction 0.15 \
						--subject_matching_regex 'SC4(\d{2}).*'
```

*Please be aware that if you modify any of the above commands to e.g. use different output directory names, you will need to modify paths in dataset hyperparameter files stored under `hyperparameters/dataset_configurations` as appropriate before proceding with the next steps.*

Run the following command to prepare the data for training:

```console
(u-sleep) $ ut preprocess --out_path data/processed_data.h5 --dataset_splits train_data val_data
```

Start training:

```console
(u-sleep) $ ut train --num_GPUs=1 --preprocessed
```

Predict on the testing sets using all channel combinations and compute majority votes:

```console
(u-sleep) $ ut predict	--num_GPUs=1 \
						--data_split test_data \
						--strip_func strip_to_match \
						--one_shot \
						--save_true \
						--majority \
						--out_dir predictions
```

Print a global confusion matrix (computed across all subjects) for dataset `sedf_sc`:

```console
(u-sleep) $ ut cm	--true 'predictions/test_data/sedf_sc/*TRUE.npy' \
					--pred 'predictions/test_data/sedf_sc/majority/*PRED.npy' \
					--ignore 5 \
					--round 2 \
					--wake_trim_min 30
```


## Full Reproduction of U-Sleep
TODO.

## Citations

If you found U-Time and/or U-Sleep useful in your scientific study, please consider citing the paper(s):

#### [1] U-Time

```
@incollection{NIPS2019_8692,
	title = {U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging},
	author = {Perslev, Mathias and Jensen, Michael and Darkner, Sune and Jennum, Poul J\o rgen and Igel, Christian},
	booktitle = {Advances in Neural Information Processing Systems 32},
	editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
	pages = {4415--4426},
	year = {2019},
	publisher = {Curran Associates, Inc.},
	url = {http://papers.nips.cc/paper/8692-u-time-a-fully-convolutional-network-for-time-series-segmentation-applied-to-sleep-staging.pdf}
}
```

#### [2] U-Sleep

```
[IN REVIEW]
```

# U-Sleep

This repository stores code for training and evaluating the U-Sleep sleep staging model. The U-Sleep repository builds upon and significantly extends our [U-Time](https://github.com/perslev/U-Time) repository, published in [[1]](#utime_ref). In the following, we will use the term *U-Sleep* to denote the recilient high frequency sleep staging model, and *u-time* to denote this repository of code used to train and evaluate U-Sleep.

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

```
conda env create --file u-sleep/environment.yaml
conda activate u-sleep
pip install u-sleep
```

This installation process may take up to 10 minutes to complete.

## Demo
In this following we will demonstrate how to launch a short training session of U-Sleep on a significantly limited subset of the datasets used in [[2]](#usleep_ref).

First, we create a project directory that will store all of our data for this demo. The `ut init` command will create a folder and populate it with a set of default hyperparameter values:

```
ut init --name demo --model usleep_demo
```

Entering the newly created project directory we will find a folder storing hyperparameters:

```
cd demo
ls
> hyperparameters
```

We will download 12 PSG studies from the public sleep databases [Sleep-EDF](https://doi.org/10.13026/C2X676) and [DCSM](https://sid.erda.dk/wsgi-bin/ls.py?share_id=fUH3xbOXv8) using the `ut fetch` command.
Note that depending on your internet speed and the current load on each of the two servers, downloading may take anywhere from 5 minutes to multiple hours:

```
ut fetch --dataset sedf_sc --out_dir data/sedf_sc --N_first 12
ut fetch --dataset dcsm --out_dir data/dcsm --N_first 12
```

The raw data has now been downloaded. We split each dataset into fixed train/validation/test splits using the `ut cv_split` command. 
The command must be invoked twice each with a unique set of parameters specifying the naming convention used in each dataset:

```
# Split dataset SEDF-SC
ut cv_split --data_dir data/sedf_sc/ \
            --subject_dir_pattern 'SC*' \
            --CV 1 \
            --validation_fraction 0.10 \
            --test_fraction 0.15 \
            --subject_matching_regex 'SC4(\d{2}).*' \
            --seed 123
            
# Split dataset DCSM
ut cv_split --data_dir data/dcsm/ \
            --subject_dir_pattern 'tp*' \
            --CV 1 \
            --validation_fraction 0.10 \
            --test_fraction 0.15 \
            --seed 123
```

Note that the splitting of SEDF-SC is performed on a per-subject basis. All PSG records from the same subject will be placed into the same dataset split. 
This is not needed for DCSM as all recordings are of unique subjects.

*Please be aware that if you modify any of the above commands to e.g. use different output directory names, you will need to modify paths in dataset hyperparameter files stored under `hyperparameters/dataset_configurations` as appropriate before proceding with the next steps.*

Run the following command to prepare the data for training:

```
ut preprocess --out_path data/processed_data.h5 --dataset_splits train_data val_data
```

We may now start training by invoking the `ut train` command. Note that many optimization hyperparameters have been pre-specified and are located in the `hyperparameters/hparams.yaml` 
file of your project directory. In this demo, we are going to run only a very short training session, but feel free to modify any parameters in the `hparams.yaml` file as you see fit:

```
ut train --num_GPUs=1 --preprocessed --seed 123
```

Following training, a set of candidate models will be available in the folder `models`. Using the best one observed (highest validation mean F1 score), 
we may predict on the testing sets of both `SEDF-SC` and `DCSM` using all channel combinations as well as compute majority votes by invoking the following `ut predict` command:

```
ut predict --num_GPUs=1 \
           --data_split test_data \
           --strip_func strip_to_match \
           --one_shot \
           --save_true \
           --majority \
           --out_dir predictions
```

The predicted hypnograms are now available under directory `predictions/test_data`. 
Finally, let us print a global confusion matrix (computed across all subjects) for dataset `sedf_sc`:

```
ut cm --true 'predictions/test_data/sedf_sc/*TRUE.npy' \
      --pred 'predictions/test_data/sedf_sc/majority/*PRED.npy' \
      --ignore 5 \
      --round 2 \
      --wake_trim_min 30
      
>
>
>
>
>
>
```

If you got the output above, congratulations! 
You have successfully installed, configured, trained and evaluated a U-Sleep model on two different datasets.

## Full Reproduction of U-Sleep
TODO.

## Citations

If you found U-Time and/or U-Sleep useful in your scientific study, please consider citing the paper(s):

#### <a name="utime_ref"> [1] U-Time

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

#### <a name="usleep_ref"> [2] U-Sleep

```
[IN REVIEW]
```

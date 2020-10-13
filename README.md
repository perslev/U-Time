# U-Sleep

Official implementation of the *U-Sleep* model for resilient high-frequency sleep staging.

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Full Reproduction of U-Sleep](#full-reproduction-of-u-sleep)
- [U-Time and U-Sleep References](#references)


## Overview
This document describes the official software package developed for and used to create the free and public sleep staging system `U-Sleep` [[2]](#usleep_ref).
U-Sleep is a fully convolutional deep neural network for automated sleep staging. A single instance of the model may be trained to perform accurate and resilient sleep staging 
across a wide range of clinical populations and polysomnography (PSG) acquisition protocols.

This software allows simultaneous training of U-Sleep across any number of PSG datasets using on-the-fly random selection of input channel configurations. It features a command-line interface for initializing, training and evaluating models without needing to modify the underlying codebase.


#### Pre-trained U-Sleep
In the following we will introduce the software behind U-Sleep in greater detail. Please note that:

* If you are interested to re-implement, extend, or train U-Sleep yourself e.g. on other datasets, you are at the right place!
* If you are looking to use our pre-trained U-Sleep model for automated sleep staging, please refer to https://sleep.ai.ku.dk and follow the introduction steps.


#### U-Time and U-Sleep - What's the Difference?
This repository stores code for training and evaluating the `U-Sleep` sleep staging model. It builds upon and significantly extends our [U-Time](https://github.com/perslev/U-Time) repository, published at NeurIPS 2019 [[1]](#utime_ref). In the following, we will use the term *U-Sleep* to denote the resilient high frequency sleep staging model currently in review [[2]](#usleep_ref), and *U-Time* to denote this repository of code used to train and evaluate the U-Sleep model.

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

#### Requirements
- Completion of the steps outlined in the [Installation Guide](#installation-guide).
- This demo takes approximately `30 minutes` to complete on a typical computer and network connection. The majority of this time is spend downloading the required data from public databases. This step may take significantly longer depending on current database traffic.
- Approximately `11 GiB` of available disk-space on your computer.


#### Preparing a project directory
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

#### Download public PSG data
We will download 6 PSG studies from the public sleep databases [Sleep-EDF](https://doi.org/10.13026/C2X676) and [DCSM](https://sid.erda.dk/wsgi-bin/ls.py?share_id=fUH3xbOXv8) using the `ut fetch` command.
You will need approximately `10 GiB` of free hard-disk space to store the downloaded files.
Note that depending on your internet speed and the current load on each of the two servers, downloading may take anywhere from 5 minutes to multiple hours:

```
ut fetch --dataset sedf_sc --out_dir data/sedf_sc --N_first 6
ut fetch --dataset dcsm --out_dir data/dcsm --N_first 6
```

The raw data that we will consider in this demo has now been downloaded. 
Our `demo` project folder now has roughly the following structure:

```
└─ demo
   ├─ hyperparameters
   └─ data
      ├─ dcsm
      │  ├─ tp005f7e68_a084_46bb_9f0a_b6a084155a1c
      │  │  ├─ hypnogram.ids
      │  │  └─ psg.h5
      │  ├─ ...
      └─ sedf_sc
         ├─ SC4001E0
         │  ├─ SC4001E0-PSG.edf
         │  └─ SC4001EC-Hypnogram.edf
         ├─ ...
```

#### Dataset splitting
Before proceeding to train the U-Sleep model we split each dataset into fixed train/validation/test splits using the `ut cv_split` command. 
The command must be invoked twice each with a unique set of parameters specifying the naming conventions of dataset:

```
# Split dataset SEDF-SC
ut cv_split --data_dir data/sedf_sc/ \
            --subject_dir_pattern 'SC*' \
            --CV 1 \
            --validation_fraction 0.10 \
            --test_fraction 0.25 \
            --subject_matching_regex 'SC4(\d{2}).*' \
            --seed 123
            
# Split dataset DCSM
ut cv_split --data_dir data/dcsm/ \
            --subject_dir_pattern 'tp*' \
            --CV 1 \
            --validation_fraction 0.10 \
            --test_fraction 0.25 \
            --seed 123
```

Note that the splitting of `SEDF-SC` is performed on a per-subject basis. All PSG records from the same subject will be placed into the same dataset split. 
This is not needed for `DCSM` as all recordings are of unique subjects.

*Please be aware that if you modify any of the above commands to e.g. use different output directory names, you will need to modify paths in dataset hyperparameter files stored under `hyperparameters/dataset_configurations` as appropriate before proceding with the next steps.*

#### Data pre-processing
Run the following command to prepare the data for training:

```
ut preprocess --out_path data/processed_data.h5 --dataset_splits train_data val_data
```


#### Training the model
We may now start training by invoking the `ut train` command. Note that many optimization hyperparameters have been pre-specified and are located in the `hyperparameters/hparams.yaml` 
file of your project directory. In this demo, we are going to run only a very short training session, but feel free to modify any parameters in the `hparams.yaml` file as you see fit:

```
ut train --num_GPUs=1 --preprocessed --seed 123
```

Following training, a set of candidate models will be available in the folder `models`. Using the best one observed (highest validation mean F1 score), 
we may predict on the testing sets of both `SEDF-SC` and `DCSM` using all channel combinations as well as compute majority votes by invoking the following `ut predict` command:


#### Predicting and evaluating on the test sets

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

>>>  Looking for files...
>>>  Loading 2 pairs...
>>>  OBS: Wake trimming of 30 minutes (period length 30 sec)
>>>  OBS: Ignoring class(es): [5]
>>>  
>>>  Raw Confusion Matrix:
>>>  
>>>          Pred 0  Pred 1  Pred 2  Pred 3  Pred 4
>>>  True 0       0       0      39     212       0
>>>  True 1       0       0     185      93       0
>>>  True 2       0       0     899      48       0
>>>  True 3       0       0     100     114       0
>>>  True 4       0       0     290      52       0
>>>  
>>>  Raw Metrics:
>>>  
>>>             F1  Precision  Recall/Sens.
>>>  Class 0  0.00       0.00          0.00
>>>  Class 1  0.00       0.00          0.00
>>>  Class 2  0.73       0.95          0.59
>>>  Class 3  0.31       0.53          0.22
>>>  Class 4  0.00       0.00          0.00
>>>  mean     0.21       0.30          0.16 
```

If you received an output similar to the, congratulations! You have successfully installed, configured, trained and evaluated a U-Sleep model on two different datasets.

Two important notes: 
* If you ran the above code on a GPU, you may not obtain the exact same numbers listed here, even if you specified the --seed arguments. This is because some operations are non-deterministic when evaluated on a GPU.

* The performance of the obtained demo model is very low and not suitable for actual sleep staging. The reason is that we trained U-Sleep on a very limited set of data and for a very limited number of epochs.
Please refer to the [Full Reproduction of U-Sleep](#full-reproduction-of-u-sleep) section for details on how to prepare and train a complete version of U-Sleep.


## Full Reproduction of U-Sleep
TODO.


## References

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
*[in review]*
Title:         U-Sleep: Resilient High-Frequency Sleep Staging
Authors:       Mathias Perslev (1), Sune Darkner (1), Lykke Kempfner (2), Miki Nikolic (2), Poul Jørgen Jennum (2) & Christian Igel (1)
Affiliations:  (1) Department of Computer Science, University of Copenhagen, Denmark
               (2) Danish Center for Sleep Medicine, Rigshospitalet, Glostrup, Denmark
```

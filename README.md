# U-Time & U-Sleep

Official implementation of

* The *U-Time* [[1]](#utime_ref) model for general-purpose time-series segmentation.
* The *U-Sleep* [[2]](#usleep_ref) model for resilient high-frequency sleep staging.

This repository may be used to train both the original U-Time and newer U-Sleep models.
However, the repository has been significantly extended since [[1]](#utime_ref) and may gradually 
diverge from the version described in [[2]](#usleep_ref). Earlier versions may be found at:

* [U-Sleep paper version](https://github.com/perslev/U-Time/tree/usleep-paper-version).
* [Latest U-Time only version](https://github.com/perslev/U-Time/tree/utime-latest).
* [U-Time paper version](https://github.com/perslev/U-Time/releases/tag/utime-paper-version).

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [U-Sleep Demo](#demo)
- [Full Reproduction of U-Sleep](#full-reproduction-of-u-sleep)
- [U-Time Example](#u-time-example)
- [U-Time and U-Sleep References](#references)


## Overview
This document describes the official software package developed for and used to create the free and public sleep staging system *U-Sleep* [[2]](#usleep_ref).
U-Sleep is a fully convolutional deep neural network for automated sleep staging. A single instance of the model may be trained to perform accurate and resilient sleep staging 
across a wide range of clinical populations and polysomnography (PSG) acquisition protocols.

This software allows simultaneous training of U-Sleep across any number of PSG datasets using on-the-fly random selection of input channel configurations. It features a command-line interface for initializing, training and evaluating models without needing to modify the underlying codebase.


#### Pre-trained U-Sleep
In the following we will introduce the software behind U-Sleep in greater detail. Please note that:

* If you are interested to re-implement, extend, or train U-Sleep yourself e.g. on other datasets, you are at the right place!
* If you are looking to use our pre-trained U-Sleep model for automated sleep staging, please refer to https://sleep.ai.ku.dk and follow the displayed guide. See also [this repository](https://github.com/perslev/U-Sleep-API-Python-Bindings) for Python bindings to the webserver API.
* U-Sleep is also available under research and commercial licenses on the [youSleep BETA platform](https://yousleep.ai).


#### U-Time and U-Sleep - What's the Difference?
This repository stores code for training and evaluating the *U-Sleep* sleep staging model. It builds upon and significantly extends our [U-Time](https://github.com/perslev/U-Time/tree/utime-latest) repository, published at NeurIPS 2019 [[1]](#utime_ref). In the following, we will use the term *U-Sleep* to denote the resilient high frequency sleep staging model [[2]](#usleep_ref), and *U-Time* to denote this repository of code used to train and evaluate the U-Sleep model.

You can still use this repository to train the older U-Time model, see [U-Time Example](#u-time-example) below.

## System Requirements
**Minimal Hardware Requirements**

Using an already trained U-Sleep model for sleep staging may typically be done on any modern laptop (subject to the software requirements listed below).

For training U-Sleep models from scratch, however, we highly recommend using a Linux based computer with at least the following hardware specifications:

* 4+ CPU cores
* 8+ GiB RAM
* Significant physical storage space*
* 1 CUDA enabled GPU (please refer to [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) for a detailed list).

It is possible to train the model on smaller machines, and without GPUs, but doing so may take considerable time. Likewise, more resources will speed up training. If the considered dataset exceeds the system memory (e.g. the 8 GiB of RAM suggested above), data must be preprocessed and streamed from disk as demonstrated in the [Demo](#demo) section below. On larger machines, one may benefit from maintaining a larger pool of data loaded in memory. For instance, we trained U-Sleep [[2]](#usleep_ref) using 8 CPU cores, 1 GPU and 40 GiB of RAM, please refer to the [Full Reproduction of U-Sleep](#full-reproduction-of-u-sleep) section below.

*The required hard-disk space depends on the number and sizes of datasets considered. For a [Full Reproduction of U-Sleep](#full-reproduction-of-u-sleep) approximately 4 TiB of available storage is needed.

**Software Requirements:**

* A computer with a Linux operating system installed. We have developed and tested the software for Red Hat Enterprise (v7.8) and Ubuntu (v18.04) servers, but any modern distribution should work. The software has also been tested on MacOS Catalina (10.15.6) for CPU-based training and prediction only.
* [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), v4.5 or higher, 64-bit.

If you are going to train a U-Sleep model yourself from scratch, we highly recommend doing so on a GPU. In order to use the `U-Time` package with a GPU, the `tensorflow` (`v2.8.0`) library is required. For this, the following additional software is required on your system:

* [NVIDIA GPU drivers](https://www.nvidia.com/drivers) v450.x or higher.
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) v11.2.
* [cuDNN SDK](https://developer.nvidia.com/cudnn) v8.1.

Please refer to [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) for additional details. You do not need to install TensorFlow yourself (see [Installation Guide](#installation-guide) below), but the above software must be installed before proceeding.

## Installation Guide
On a Linux machine with at least 1 CUDA enabled GPU available and `anaconda` 
or `miniconda` installed, run the following commands to download the software, create a conda environment named `u-sleep` 
and setup the latest U-Time software package and its dependencies:

```
git clone https://github.com/perslev/U-Time.git
conda env create --file U-Time/environment.yaml
conda activate u-sleep
pip install U-Time/
```

Alternatively, you may install the package from [PyPi](https://pypi.org) (may be updated less frequently):

```
pip install utime
```

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
You will need approximately `10 GiB` of free hard-disk space to store the downloaded files. Depending on your internet speed and the current load on each of the two servers, downloading may take anywhere from 5 minutes to multiple hours:

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

The `ut preprocess` script loads and processes all datasets as described by the parameters set in `hyperparameters/hparams.yaml` and all dataset-specific files in the folder `hyperparameters/dataset_configurations`.
Specifically, it loads the needed channels (ignoring the rest), re-samples, scales and clips the data, maps hypnogram stages to interger representations used internally during training and finally saves the processed data to an HDF5 archive.
When training, data may be streamed directly from this archive to significantly reduce the required system memory.

It is also possible to skip this step all together and either **1)** load all data needed for training up front, or **2)** stream and apply preprocessing on-the-fly during training as shown in the [Full Reproduction of U-Sleep](#full-reproduction-of-u-sleep) section below.


#### Training the model
We may now start training by invoking the `ut train` command. A default set of optimization hyperparameters have been pre-specified and are located in the `hyperparameters/hparams.yaml` 
file of your project directory. In this demo, we are going to run only a very short training session, but feel free to modify any parameters in the `hparams.yaml` file as you see fit.
Run the following command:

```
ut train --num_gpus=1 --preprocessed --seed 123
```

You may replace the `--num_gpus=1` parameter in the above command with `--num_gpus=0` if you do not have a GPU available, and wish to train on the CPU. Training on CPU may take up to 30 minutes.

Following training, a set of candidate models will be available in the folder `model`. Using the best one observed (highest validation mean F1 score), 
we may predict on the testing sets of both `SEDF-SC` and `DCSM` using all channel combinations as well as compute majority votes by invoking the following `ut predict` command:


#### Predicting and evaluating on the test sets

```
ut predict --num_gpus=1 \
           --data_split test_data \
           --strip_func strip_to_match \
           --one_shot \
           --save_true \
           --majority \
           --out_dir predictions
```

The predicted hypnograms are now available under directory `predictions/test_data`. 
Finally, let us print a global confusion matrix (computed across all subjects) for dataset `sedf_sc` (replace `sedf_sc` -> `dcsm` for DCSM evaluation):

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
>>>  True 0       0       0      17     234       0
>>>  True 1       0       0     132     146       0
>>>  True 2       0       0     790     157       0
>>>  True 3       0       0      25     189       0
>>>  True 4       0       0     243      99       0
>>>  
>>>  Raw Metrics:
>>>  
>>>             F1  Precision  Recall/Sens.
>>>  Class 0  0.00       0.00          0.00
>>>  Class 1  0.00       0.00          0.00
>>>  Class 2  0.73       0.65          0.83
>>>  Class 3  0.36       0.23          0.88
>>>  Class 4  0.00       0.00          0.00
>>>  mean     0.22       0.18          0.34 
```

If you received an output *similar* to the above, congratulations! You have successfully installed, configured, trained and evaluated a U-Sleep model on two different datasets.

Please note that:
* If you ran the above code on a GPU, you may not obtain the exact same numbers listed here, even if you specified the --seed arguments. This is because some computations used during the training of U-Sleep are fundamentally non-deterministic when evaluated on a GPU. 
However, predicting using a trained U-Sleep model will give deterministic outputs.

* The performance of the obtained demo model is very low and not suitable for actual sleep staging. The reason is that we trained U-Sleep on a very limited set of data and for a very limited number of epochs.
Please refer to the [Full Reproduction of U-Sleep](#full-reproduction-of-u-sleep) section for details on how to prepare and train a complete version of U-Sleep.


## Full Reproduction of U-Sleep
The [Demo](#demo) above in principle describes all steps needed to reproduce U-Sleep as reported in [[2]](#usleep_ref). 
The main - and significant - difference is that in order to reproduce the full model, you will need to 1) be able to access 2) download and 3) preprocess all the required datasets. You may also need a computer with more resources as described in [System Requirements](#system-requirements). 


#### Prepare the datasets
We did our best to make this process as easy as possible. You should take the following steps:

1) Carefully read (at least) the *Methods* and supplementary *Datasets* sections of our paper [[2]](#usleep_ref) to familiarize yourself with the datasets, preprocessing, training pipeline and more.
2) Download all datasets from the [National Sleep Research Resource](https://sleepdata.org), [PhysioNet](https://sleepdata.org) or other sleep repositories as described and referenced in the Supplementary Material's *Dataset* section. For some datasets you must apply to gain access, while others are publicly available. Some datasets may be easily downloaded using the `ut fetch` command. Please invoke `ut fetch --help` to see an up-to-date list of which datasets may be downloaded this way.
3) Place all downloaded data into a single folder `[LOCAL_PATH]` with 1 sub-folder for each dataset.
4) Run `ut extract`, `ut extract_hypno`, and `ut cv_split` on all datasets as specified for each dataset separately in files under the folder `resources/usleep_dataset_pred` of this repository (also found [here](https://sid.erda.dk/wsgi-bin/ls.py?share_id=HE5nA4Xs37)). These commands will extract and place data into a folder-structure and format that U-Time accepts, as well as split the data into subsets.
5) (optional) The `ut extract` command will select the relevant channels, re-sample them to 128 Hz and store the data in HDF5 archives. The original data will not be deleted by default. If you have limited hard-drive space, consider removing the old files before processing the next dataset.
6) Initialize a U-Sleep project: `ut init --name u-sleep --model usleep`.
7) For each dataset configuration file in `u-sleep/hyperparameters/dataset_configurations/` replace the string [LOCAL_PATH] with the `[LOCAL_PATH]` of your data.


#### Train the model
If you have 40+ GiB system memory available, train U-Sleep using the following command:

```
ut train --num_gpus 1 --max_loaded_per_dataset 40 --num_access_before_reload 32 --train_queue_type limitation --val_queue_type lazy --max_train_samples_per_epoch 1000000
```

On systems with less memory, you may either 1) reduce the `--max_loaded_per_dataset` parameter from the current `40` to a lower value (this will keep fewer PSG records in the active memory pool, which will reduce randomness when selecting records), or 2) preprocess the data and stream data during training (as demonstrated in the Demo above) by invoking the following two commands (replacing [LOCAL_PATH] as applicable):

```
ut preprocess --out_path '[LOCAL_PATH]/processed_data.h5' --dataset_splits train_data val_data
ut train --num_gpus 1 --preprocessed --max_train_samples_per_epoch 1000000
```

This will apply all preprocessing, create a data archive suitable for streaming, and train U-Sleep using samples loaded on-the-fly from disk.

Due to the vast size of the dataset considered, training U-Sleep with the default parameters may take very long. 
We suggest increasing the learning rate (from the current `1e-7` to e.g. `1e-6`) unless you are looking to re-create U-Sleep under the exact conditions considered in [[2]](#usleep_ref).

## U-Time Example
You can still use this repository to train the older U-Time model. 
In the following we show an end-to-end example. The commands listed below prepares a project folder, downloads the sleep-edf-153 dataset, fits and evaluates
a U-Time model in a fixed train/val/test dataset split setup. Please note that the below code does not reproduce the sleep-edf-153 experiment of [[1]](#utime_ref) as 10-fold CV was used.
To run a CV experiment, please refer to the `ut cv_split --help` and `ut cv_experiment --help` commands.

<pre>
<b># Obtain a public sleep staging dataset</b>
ut fetch --dataset sedf_sc --out_dir datasets/sedf_sc

<b># Prepare a fixed-split experiment</b>
ut cv_split --data_dir 'datasets/sedf_sc' \
            --subject_dir_pattern 'SC*' \
            --CV 1 \
            --validation_fraction 0.20 \
            --test_fraction 0.20 \
            --subject_matching_regex 'SC4(\d{2}).*' \
            --file_list

<b># Initialize a U-Time project</b>
ut init --name my_utime_project \
        --model utime \
        --data_dir datasets/sedf_sc/views/fixed_split

<b># Start training</b>
cd my_utime_project
ut train --num_gpus=1 --channels 'EEG Fpz-Cz'

<b># Predict and evaluate</b>
ut evaluate --out_dir eval --one_shot

<b># Print a confusion matrix</b>
ut cm --true 'eval/test_data/dataset_1/files/*/true.npz' \
      --pred 'eval/test_data/dataset_1/files/*/pred.npz'
      
<b># Print per-subject summary statistics</b>
ut summary --csv_pattern 'eval/test_data/*/evaluation_dice.csv' \
           --print_all

<b># Output sleep stages for every 3 seconds of 128 Hz signal </b>
<b># Here, the 'folder_regex' matches 2 files in the dataset </b>
ut predict --folder_regex '../datasets/sedf_sc/SC400[1-2]E0' \
           --out_dir high_res_pred \
           --data_per_prediction 384 \
           --one_shot
</pre>


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
U-Sleep: Resilient High-Frequency Sleep Staging
Mathias Perslev (1), Sune Darkner (1), Lykke Kempfner (2), Miki Nikolic (2), Poul Jørgen Jennum (2) & Christian Igel (1)
npj Digital Medicine, 4, 72 (2021)
https://doi.org/10.1038/s41746-021-00440-5

(1) Department of Computer Science, University of Copenhagen, Denmark
(2) Danish Center for Sleep Medicine, Rigshospitalet, Glostrup, Denmark
```

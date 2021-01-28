# U-Time

Implementation of the U-Time model for time-series segmentation as described 
in:

Mathias Perslev, Michael Hejselbak Jensen, Sune Darkner, Poul Jørgen Jennum, 
and Christian Igel. U-Time: A Fully Convolutional Network for Time Series 
Segmentation Applied to Sleep Staging. Advances in Neural Information 
Processing Systems (NeurIPS 2019)

NeurIPS proceedings:
https://proceedings.neurips.cc/paper/2019/file/57bafb2c2dfeefba931bb03a835b1fa9-Paper.pdf

## Code Changes
This codebase is live and may gradually diverge from the version described in the above paper.
Please see to the following [repository tag](https://github.com/perslev/U-Time/releases/tag/paper_version) if you are interested in the original paper implementation.

## An end-to-end example
The commands listed below installs U-Time, prepares a project folder, downloads the sleep-edf-153 dataset, fits and evaluates
a U-Time model in a fixed train/val/test dataset split setup. Please note that this code is only intended as a quick example of how
one may use this codebase. The below code does not reproduce the sleep-edf-153 experiment listed in the paper as 10-fold CV was used.
To run a CV experiment, please refer to the `ut cv_split --help` and `ut cv_experiment --help` commands.

<pre>
<b># Clone repo and install</b>
git clone https://github.com/perslev/U-Time
pip3 install U-Time/

<b># Obtain a public sleep staging dataset</b>
ut fetch --dataset sleep-EDF-153 --out_dir datasets/sleep-EDF-153

<b># Prepare a fixed-split experiment</b>
ut cv_split --data_dir 'datasets/sleep-EDF-153' \
            --subject_dir_pattern 'SC*' \
            --CV 1 \
            --validation_fraction 0.20 \
            --test_fraction 0.20 \
            --common_prefix_length 5 \
            --file_list

<b># Initialize a U-Time project</b>
ut init --name my_utime_project \
        --model utime \
        --data_dir datasets/sleep-EDF-153/views/fixed_split

<b># Start training</b>
cd my_utime_project
ut train --num_GPUs=1 --channels 'EEG Fpz-Cz'

<b># Predict and evaluate</b>
ut evaluate --out_dir eval --one_shot

<b># Print a confusion matrix</b>
ut cm --true 'eval/test_data/dataset_1/files/*/true.npz' \
      --pred 'eval/test_data/dataset_1/files/*/pred.npz'
      
<b># Print per-subject summary statistics</b>
ut summary --csv_pattern 'eval/test_data/*/evaluation_dice.csv' \
           --print_all

<b># Output sleep stages for every 3 seconds of 100 Hz signal </b>
<b># Here, the 'folder_regex' matches 2 files in the dataset </b>
ut predict --folder_regex '../datasets/sleep-EDF-153/SC400[1-2]E0' \
           --out_dir high_res_pred \
           --data_per_prediction 300 \
           --one_shot
</pre>

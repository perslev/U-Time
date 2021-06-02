## Dataset: SEDF-ST

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/sleep-edf-extended/ST*/*PSG.edf' --out_dir '[LOCAL_PATH]/processed/sedf-st/' --resample 128 --channels 'EEG Fpz-Cz' 'EEG Pz-Oz' 'EOG horizontal' --rename Fpz-Cz Pz-Oz EOG
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/sleep-edf-extended/ST*/*Hypnogram.edf' --out_dir '[LOCAL_PATH]/processed/sedf-st/' --fill_blanks 'Sleep stage ?'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/sedf-st/' --subject_dir_pattern 'ST*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex 'ST7(\d{2}).*'
```

Notes: 
- Two sleep studies made on each subject
- 2 groups, files are named in the form ST7ssNJ0-PSG.edf where ss is the subject number, and N is the night.
- example nameing: 'ST7071J0-PSG', 'ST7072J0-PSG' 
- match regex: 'ST7(\d{2}).*'

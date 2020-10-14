## Dataset: SVUH

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/svuh-ucd-sleep-apnea/uc*/*.edf' --out_dir '[LOCAL_PATH]/processed/svuh/' --resample 128 --channels 'C3A2' 'C4A1' 'Lefteye' 'RightEye' --rename_channels 'C3-A2' 'C4-A1' 'EOG(L)' 'EOG(R)'
```

####Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/svuh-ucd-sleep-apnea/uc*/*-HYP.npz' --out_dir '[LOCAL_PATH]/processed/svuh/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/svuh/' --subject_dir_pattern 'uc*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100
```

Notes: 
- No subject relations specified
- example nameing: 'ucddb017'
- match regex: None

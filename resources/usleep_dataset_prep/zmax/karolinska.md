## Dataset: Karolinska

#### Export to USleep format

#### Views command
```
ut cv_split --data_dir [path_to_data_directory] --out_dir [path_to_output_directory] --subject_dir_pattern 'SNZ*' --CV 1 --subject_matching_regex 'SNZ_(\d{3})_.*' --validation_fraction 0.2 --test_fraction 0.0
```

Notes: 
- example nameing: 'SNZ_203_noSnooze'
- Default match regex: 'SNZ_(\d{3})_.*'

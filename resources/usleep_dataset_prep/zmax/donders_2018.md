## Dataset: Donders 2018

#### Export to USleep format

#### Views command
```
ut cv_split --data_dir [path_to_data_directory] --out_dir [path_to_output_directory] --subject_dir_pattern 'P*' --CV 1 --subject_matching_regex 'P(\d{1,2})_.*' --validation_fraction 0.2 --test_fraction 0.0
```

Notes: 
- example nameing: 'P8_night3'
- Default match regex: 'P(\d{1,2})_.*'

## Dataset: Donders 2022

#### Export to USleep format

#### Views command
```
ut cv_split --data_dir [path_to_data_directory] --out_dir [path_to_output_directory] --subject_dir_pattern 's*' --CV 1 --subject_matching_regex 's(\d{1,2})_.*' --validation_fraction 0.2 --test_fraction 0.0
```

Notes: 
- example nameing: 's1_n1'
- Default match regex: 's(\d{1,2})_.*'

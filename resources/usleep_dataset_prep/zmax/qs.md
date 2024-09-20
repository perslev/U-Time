## Dataset: QS

#### Export to USleep format

#### Views command
```
ut cv_split --data_dir [path_to_data_directory] --out_dir [path_to_output_directory] --subject_dir_pattern 's*' --CV 1 --subject_matching_regex 's(\d).*' --validation_fraction 0.0 --test_fraction 0.0
```

Notes: 
- All files are for the same subject, therefore it either can only be used for training/validation or for testing. 
- example nameing: 's1_01_10_2018(somno)'
- Default match regex: 's(\d).*'

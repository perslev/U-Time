## Dataset: DCSM

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/dcsm/*/*.h5' --out_dir '[LOCAL_PATH]/processed/dcsm/' --resample 128 --use_dir_names --channels 'F3-M2' 'F4-M1' 'C3-M2' 'C4-M1' 'O1-M2' 'O2-M1' 'E1-M2' 'E2-M2'
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/dcsm/*/*.ids' --out_dir '[LOCAL_PATH]/processed/dcsm/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/dcsm/' --subject_dir_pattern 'tp*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100
```

Notes:
- No subject relations specified
- 1 group
- example nameing: 'tpdbf4cda5_e571_4f0e_be9a_1a48306b1247'
- match regex: None

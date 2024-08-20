## Dataset: SEDF-SC

#### Extract command
```

```

#### Extract hypno command
```

```

#### Views command
```
ut cv_split --data_dir '/home/sleep/alisab/sleep/zmax-datasets/data/qs/' --out_dir '/home/sleep/alisab/sleep/zmax/data/qs/views/' --subject_dir_pattern 's*' --CV 1 --validation_fraction 0.2 --subject_matching_regex 's(\d).*'
```

Notes: 
- All files are for the same subject
- example nameing: 's1_n01_10_2018(somno)'
- Default match regex: 's(\d)_.*'

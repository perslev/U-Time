## Dataset: MROS

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/mros/polysomnography/edfs/visit*/*.edf' --out_dir '[LOCAL_PATH]/processed/mros/' --resample 128 --channels C3-M2 C4-M1 E1-M2 E2-M1
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/mros/polysomnography/annotations-events-nsrr/visit*/*.xml' --out_dir '[LOCAL_PATH]/processed/mros/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/mros/' --subject_dir_pattern 'mros*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)'
```

Notes:
- 2 visits: 'visit1', 'visit2'
- example nameing: mros-visit1-aa5665, mros-visit2-aa5665
- match regex: .*?-.*?-(.*)

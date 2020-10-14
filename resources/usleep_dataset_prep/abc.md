## Dataset: ABC

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/abc/polysomnography/edfs/*/*.edf' --out_dir '[LOCAL_PATH]/processed/abc/' --resample 128 --channels F3-M2 F4-M1 C3-M2 C4-M1 O1-M2 O2-M1 E1-M2 E2-M1
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/abc/polysomnography/annotations-events-nsrr/*/*.xml' --out_dir '[LOCAL_PATH]/processed/abc/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/abc/' --subject_dir_pattern 'abc*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)'
```

Notes: 
- 3 visits: 'baseline', 'month09', 'month18'
- example nameing: abc-baseline-900001, abc-month18-900001, abc-month09-900001
- match regex: .*?-.*?-(.*)

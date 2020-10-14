## Dataset: SOF

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/sof/polysomnography/edfs/*.edf' --out_dir '[LOCAL_PATH]/processed/sof/' --resample 128 --channels C3-A2 C4-A1 LOC-A2 ROC-A1
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/sof/polysomnography/annotations-events-nsrr/*.xml' --out_dir '[LOCAL_PATH]/processed/sof/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/sof/' --subject_dir_pattern 'sof*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100
```

Notes: 
- No mentioned subject relations
- 1 group: 'visit8'
- example nameing: 'sof-visit-8-07853'
- match regex: None

## Dataset: CCSHS

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/ccshs/polysomnography/edfs/*.edf' --out_dir '[LOCAL_PATH]/processed/ccshs/' --resample 128 --channels C3-A2 C4-A1 LOC-A2 ROC-A1
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/ccshs/polysomnography/annotations-events-nsrr/*.xml' --out_dir '[LOCAL_PATH]/processed/ccshs/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/ccshs/' --subject_dir_pattern 'ccshs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100
```

Notes: 
- No mentioned subject relations
- 1 group: 'trec' (study has 3 longitudinal visits, but only 'trec' is published currently)
- example nameing: 'ccshs-trec-1800005'
- match regex: None

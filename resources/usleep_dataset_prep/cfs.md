## Dataset: CFS

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/cfs/polysomnography/edfs/*.edf' --out_dir '[LOCAL_PATH]/processed/cfs/' --resample 128 --channels C3-A2 C4-A1 LOC-A2 ROC-A1 --overwrite
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/cfs/polysomnography/annotations-events-nsrr/*.xml' --out_dir '[LOCAL_PATH]/processed/cfs/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/cfs/' --subject_dir_pattern 'cfs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*famID(.*)'
```

Notes: 
- BEWARE FAMILY RELATIONS - Split based on FAMILY IDs not NSSR IDs
- Not all subjects in a family are equally far/close genetically (some could be sisters, others cusins etc.)
- 'train_records' above also gives 'train_subjects' as 1 record/subject in all cases.
- 1 group: 'visit5'
- example nameing: 'cfs-visit5-802593-famID{ID_REDACTED}'
                   'cfs-visit5-802626-famID{ID_REDACTED}'
- match regex: .*famID(.*)
- ID regex: .*visit5-(\d+).*

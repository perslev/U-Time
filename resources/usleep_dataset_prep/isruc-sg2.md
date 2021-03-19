## Dataset: ISRUC-SG2

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/ISRUC/multi_visit*/*.edf' --out_dir '[LOCAL_PATH]/processed/isruc-sg2/' --resample 128 --use_dir_names --channels 'F3-M2' 'C3-M2' 'O1-M2' 'F4-M1' 'C4-M1' 'O2-M1' 'E1-M2' 'E2-M1'
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/ISRUC/multi_visit*/*_1-HYP.npz' --out_dir '[LOCAL_PATH]/processed/isruc-sg2/'
```

#### Views command
```
None (all test)
```

Notes: 
- Two visits
- Groups: 'visit_1', 'visit_2'
- example nameing: 'multi_visit_subject_8_visit_1', 'multi_visit_subject_8_visit_2'
- match regex: '.*subject_(\d+).*'

## Dataset: ISRUC-SG3

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/ISRUC/sg3*/*.edf' --out_dir '[LOCAL_PATH]/processed/isruc-sg3/' --resample 128 --use_dir_names --channels 'F3-M2' 'C3-M2' 'O1-M2' 'F4-M1' 'C4-M1' 'O2-M1' 'E1-M2' 'E2-M1'
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/ISRUC/sg3*/*_1-HYP.npz' --out_dir '[LOCAL_PATH]/processed/isruc-sg3/'
```

#### Views command
```
None (all test)
```

Notes: 
- No subject relations specified
- example nameing: 'sg3_subject_3'
- match regex: None

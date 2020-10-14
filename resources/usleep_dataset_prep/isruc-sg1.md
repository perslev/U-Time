## Dataset: ISRUC-SG1

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/ISRUC/subject*/*.edf' --out_dir '[LOCAL_PATH]/processed/isruc-sg1/' --resample 128 --use_dir_names --channels 'F3-M2' 'C3-M2' 'O1-M2' 'F4-M1' 'C4-M1' 'O2-M1' 'E1-M2' 'E2-M1'
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/ISRUC/subject_*/*.npz' --out_dir '[LOCAL_PATH]/processed/isruc-sg1/'
```

#### Views command
```
None (all test)
```

Notes: 
- No subject relations specified
- example nameing: 'subject_82'
- match regex: None

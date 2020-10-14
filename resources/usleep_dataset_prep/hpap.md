## Dataset: HPAP

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/homepap/polysomnography/edfs/lab/*/*.edf' --out_dir '[LOCAL_PATH]/processed/homepap/' --channels F4-M1 C4-M1 O2-M1 C3-M2 F3-M2 O1-M2 E1-M2 E2-M1 E1 E2  --resample 128
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/homepap/polysomnography/annotations-events-nsrr/lab/*/*.xml' --out_dir '[LOCAL_PATH]/processed/homepap/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/hpap/' --subject_dir_pattern 'homepap*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*-(\d+)'
```

Notes: 
- No mentioned subject relations
- 2 group: 'full', 'split'
- example nameing: 'homepap-lab-full-1600039',
                   'homepap-lab-split-1600150'
- match regex (actually not be needed here): .*-(\d+)

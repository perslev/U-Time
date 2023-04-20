## Dataset: SHHS

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/shhs/polysomnography/edfs/shhs*/*.edf' --out_dir '[LOCAL_PATH]/processed/shhs/' --channels 'EEG' 'EEG(sec)' 'EOG(L)' 'EOG(R)' --resample 128 --rename_channels 'C4-A1' 'C3-A2' 'EOG(L)-PG1' 'EOG(R)-PG1'
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/shhs/polysomnography/annotations-events-nsrr/shhs*/*.xml' --out_dir '[LOCAL_PATH]/processed/shhs/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/shhs/' --subject_dir_pattern 'shhs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-(.*)'
```

Notes: 
- 2 visits: 'shhs1', 'shhs2'
- example nameing: 'shhs1-205238', 'shhs2-205238'
- match regex: .*?-(.*)

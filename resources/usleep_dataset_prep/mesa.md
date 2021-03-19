## Dataset: MESA

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/mesa/polysomnography/edfs/*.edf' --out_dir '[LOCAL_PATH]/processed/mesa/' --resample 128 --channels EEG1 EEG2 EEG3 EOG-L EOG-R --rename Fz-Cz Cz-Oz C4-M1 E1-FPz E2-FPz
```

#### Extract hypno command
```
ut extract_hypno --file_regex '[LOCAL_PATH]/mesa/polysomnography/annotations-events-nsrr/*.xml' --out_dir '[LOCAL_PATH]/processed/mesa/'
```

#### Views command
```
ut cv_split --data_dir '[LOCAL_PATH]/processed/mesa/' --subject_dir_pattern 'mesa*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100
```

Notes: 
- No mentioned subject relations
- 1 group: 'sleep'
- example nameing: 'mesa-sleep-5805'
- match regex: None

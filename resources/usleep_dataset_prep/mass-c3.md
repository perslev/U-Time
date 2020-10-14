## Dataset: MASS-C3

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/mass-c3/*PSG.edf' --out_dir '[LOCAL_PATH]/processed/mass-c3/' --resample 128 --channels 'EEG Fp1-LER' 'EEG Fp2-LER' 'EEG F7-LER' 'EEG F8-LER' 'EEG F3-LER' 'EEG F4-LER' 'EEG T3-LER' 'EEG T4-LER' 'EEG C3-LER' 'EEG C4-LER' 'EEG T5-LER' 'EEG T6-LER' 'EEG P3-LER' 'EEG P4-LER' 'EEG O1-LER' 'EEG O2-LER' 'EEG Fz-LER' 'EEG Cz-LER' 'EEG Pz-LER' 'EEG Oz-LER' 'EEG A2-LER' 'EOG Left Horiz' 'EOG Right Horiz' --rename_channels 'Fp1-LER' 'Fp2-LER' 'F7-LER' 'F8-LER' 'F3-LER' 'F4-LER' 'T3-LER' 'T4-LER' 'C3-LER' 'C4-LER' 'T5-LER' 'T6-LER' 'P3-LER' 'P4-LER' 'O1-LER' 'O2-LER' 'Fz-LER' 'Cz-LER' 'Pz-LER' 'Oz-LER' 'A2-LER' 'EOG_left_horizontal' 'EOG_right_horizontal'
```

#### Extract hypno command
```
python convert_mass_hypnograms.py 'SS3_EDF/*Base.edf' 'mass-c3'
```

#### Views command
```
None (all test)
```

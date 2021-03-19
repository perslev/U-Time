## Dataset: MASS-C1

#### Extract command
```
ut extract --file_regex '[LOCAL_PATH]/mass-c1/*PSG.edf' --out_dir '[LOCAL_PATH]/processed/mass-c1/' --resample 128 --channels 'EEG F3-CLE' 'EEG F4-CLE' 'EEG C3-CLE' 'EEG C4-CLE' 'EEG O1-CLE' 'EEG O2-CLE' 'EEG F7-CLE' 'EEG F8-CLE' 'EEG T3-CLE' 'EEG T4-CLE' 'EEG T5-CLE' 'EEG T6-CLE' 'EEG P3-CLE' 'EEG P4-CLE' 'EEG Fz-CLE' 'EEG Cz-CLE' 'EEG Pz-CLE' 'EEG Fp1-CLE' 'EEG Fp2-CLE' 'EEG F3-LER' 'EEG F4-LER' 'EEG C3-LER' 'EEG C4-LER' 'EEG O1-LER' 'EEG O2-LER' 'EEG F7-LER' 'EEG F8-LER' 'EEG T3-LER' 'EEG T4-LER' 'EEG T5-LER' 'EEG T6-LER' 'EEG P3-LER' 'EEG P4-LER' 'EEG Fz-LER' 'EEG Cz-LER' 'EEG Pz-LER' 'EEG Fp1-LER' 'EEG Fp2-LER' 'EOG Left Horiz' 'EOG Right Horiz' --rename_channels 'F3-CLE' 'F4-CLE' 'C3-CLE' 'C4-CLE' 'O1-CLE' 'O2-CLE' 'F7-CLE' 'F8-CLE' 'T3-CLE' 'T4-CLE' 'T5-CLE' 'T6-CLE' 'P3-CLE' 'P4-CLE' 'Fz-CLE' 'Cz-CLE' 'Pz-CLE' 'Fp1-CLE' 'Fp2-CLE' 'F3-LER' 'F4-LER' 'C3-LER' 'C4-LER' 'O1-LER' 'O2-LER' 'F7-LER' 'F8-LER' 'T3-LER' 'T4-LER' 'T5-LER' 'T6-LER' 'P3-LER' 'P4-LER' 'Fz-LER' 'Cz-LER' 'Pz-LER' 'Fp1-LER' 'Fp2-LER' 'EOG_left_horizontal' 'EOG_right_horizontal'
```

#### Extract hypno command
```
python convert_mass_hypnograms.py 'SS1_EDF/*Base.edf' 'mass-c1'
```

#### Views command
```
None (all test)
```

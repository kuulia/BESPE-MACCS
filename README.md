# BESPE-MACCS cheminformatics molecular descriptor
 A proof-of-concept type demo of the Binary Encoded SMARTS Pattern Enumeration - Molecular aCCess System molecular (BESPE-MACCS) descriptor. Includes a toy data set for demonstrative purposes.

# Pipeline
 1. Enumerate SMARTS patterns in the format given by enumerated_smarts_patterns.csv. For the header, SMARTS labels can be arbitrary names, but make sure to name enumerated atom counts accordingly i.e. 'carbon number' and 'oxygen count'. See my other repo that shows one way of doing this https://github.com/kuulia/SMARTS_pattern_enumeration
 2. List molecules as SMILES strings and save them to 'smiles.txt', see example file. No headers.
 3. List target values similar to the SMILES strings, and name file '{target}.txt'. You this name is used later in the 'main.py' script.
 4. Run 'main.py'.

# Script files
 'main.py' - Calls descriptor generation scripts and calls Kernel Ridge Regression (KRR) model training and testing. 

 'generate_MACCS.py' - Generates MACCS descriptor file 'MACCS.txt' from 'smiles.txt' file.
 
 'add_BESPE.py' - Binary encodes SMARTS pattern enumeration data 'from enumerated_smarts_patterns.csv', appends it to the MACCS data from 'MACCS.txt' and saves it to 'BESPE-MACCS.txt'.

 'krr.py' - Trains and tests KRR model. Outputs 'output_KRR_xxx.txt' files that contain training and testing errors for all training sizes and optionally saves plots for learning curve and scatter of predictions, and explicit predictions 'output_predictions_xxx.csv'.


Developed in Python ver. 3.12.0. See requirements.txt for relevant library versions.

cd data

# Small files
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kvpkeIv8LoCqLWcymHqg8_3JayYGnl3l' -O test_set_overexpression_normalized_profiles.csv
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dpUdO7to850ou_rCyCV4g1jRXvviGDch' -O ExCAPE_ligands_agonist_noTrainSet.csv
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gZIOM543cKMu8FgCRMpK7nS3O2zQK_5p' -O ChEMBL_standardized_smiles.csv
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pneVQI1jUTzr_jQ7hLbk1QS3NZ6Jr4Vh' -O sure_chembl_alerts.txt

# Training set needs another syntax for large files
wget --load-cookies /tmpcookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1K4Z-xPZl-p2e6XQR34xiAilyoS0XRmA1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1K4Z-xPZl-p2e6XQR34xiAilyoS0XRmA1" -O train_set_30kcpds_normalized_profiles.csv.gz && rm -rf /tmp/cookies.txt

# prepare_datasets
Prepare datasets for medical images

These scripts prepare data from public medical images dataset, mostly mammograms.
Currently supported: VINDR-MAMMO and CBIS-DDSM.

## Instructions
Instructions on how to run on each script.
Later this readme file will be more informative.

### Convert dicom to PNG preprocessing and resizing
Initially use {cbis, vindr}_select_data.py to convert dicom images to PNG. Make sure you have the original csv files downloaded with original dicom dataset. Also, make sure you edit "TOP" variable inside each script pointing the top folder of dicom files.

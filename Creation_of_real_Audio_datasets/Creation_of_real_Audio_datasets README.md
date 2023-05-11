# Synthetic-to-real Unsupervised Domain Adaptation

This folder is dedicated to software designed to perform:

REAL SOUNDS DATASET(S) CREATION
- load canonical versions of datasets of interest (like FSD50K)
- validate canonical version of downloaded and loaded dataset
- create sub-datasets from the canonical version (like FSD50K with 'water' AudioSet class only)

This repo does not contain the actual created dataset because they are too large to be contained here; nevertheless, you can find a section in this README file for each of the examples used in this Project, with instructions fdor reproducibility
  
## soundata_test.py

This script is used to test the soundata library (loader for some datasets of interest, like FSD50K). It is based on the example provided in the library's documentation.
Also, this script;

- downloads the CANONICAL VERSION OF THE FSD50K dataset (if not already downloaded)
- validates the CANONICAL VERSION OF THE FSD50K dataset
- creates a new subset of the dataset (called "FSD50K_10") containing only the wanted classes from the original 200, like 'water' only

### FSD50K (dataset folder not included in this repo)

The FSD50K dataset is a dataset of 51,197 sound clips from Freesound annotated with labels from the AudioSet Ontology. It is a subset of the Freesound dataset (FSD) created for the task of sound event detection. The dataset is described in the paper: [FSD50K: an Open Dataset of Human-Labeled Sound Events](https://arxiv.org/abs/2010.00475).

I have used soundata to download and validate the canonical version of the dataset, which has to be first downloaded as many .zip files, which have then to be extracted.

<b> This is why we can not effectively partially download the dataset, according to some classes of interest; we need to start by downloading the entire dataset.</b>

A description of the dataset files and their organization can be found here; https://zenodo.org/record/4060432#.ZFykB-xBxge.

<b>AFTER DOWNLOADING THE DATASET, I DELETED THE .zip FILES AND KEPT ONLY THE EXTRACTED FOLDERS</b>, WHICH ARE:

- FSD50K.dev_audio
- FSD50K.doc
- FSD50K.eval_audio
- FSD50K.ground_truth
- FSD50K.metadata

To ensure that no one accidentaly makes some changes on the canonical dataset, I also called the following terminal command (on MacOS) where path is the path to the FSD50K folder:

<b>```sudo chmod -R a-w,u+rx path```</b>

## virtEnv folder

This folder contains the virtual environment used to run the script soundata_test.py.

See requirements.txt for a list of installed packages (in my System, soundata was installed but, when tried to run the script, an error occurred for a missing numpy wheel; I hence updated numpy).
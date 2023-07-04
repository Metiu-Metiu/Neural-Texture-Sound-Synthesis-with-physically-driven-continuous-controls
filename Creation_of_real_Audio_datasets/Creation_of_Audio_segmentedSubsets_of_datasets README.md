# Creation_of_Audio_segmentedSubsets_of_datasets

This folder is dedicated to software designed to take an already existing dataset of audio files and <b>create a new dataset containing only a subset of the original dataset, with the possibility to segment the audio files</b> into subsequent smaller audio files of a given duration (e.g. 10 seconds). Chunks which would be smaller than the specified duration are discarded. Silent chunks are also discarded.

The already existing dataset is called the 'canonical' version of the dataset.
The subset is the dataset produced by the script 'Creation_of_Audio_segmentedSubsets_of_datasets.py', by only selecting the files in the canonical dataset which have the wanted classes (e.g. 'water' only).

You can enter many settings in the dictionary 'realSoundsDataset_Creator_Dict' in the script 'Creation_of_Audio_segmentedSubsets_of_datasets.py' to create the wanted subset of the dataset.

This repo does not contain the actual created dataset because they are too large to be contained here; nevertheless, you can find a section in this README file for each of the examples used in this Project, with instructions for reproducibility.

## Creation_of_Audio_segmentedSubsets_of_datasets.py

You can enter many settings in the dictionary 'realSoundsDataset_Creator_Dict' in the script 'Creation_of_Audio_segmentedSubsets_of_datasets.py' to create the wanted subset of the dataset.

Particular attention has been put on the <b>criteria of labels/tags matching between the original dataset and the subset</b> (the ‘keywords’ tag labels of the canonical dataset which need to also be present in the subset, or not present in the subset, in many combination and permutations). See do_CanonicalAndSubsetTags_Match_AccordingToSubsetTagsPolicy(). The user can in fact, in the JSON configuration file, specify 2 lists of string-type tags, one for the subset tags, one for the excluded tags (the latter can be empty).

The 5 options are:

<b>
- AllAndOnlySubsetTags_ArePresentInCanonicalDatasetFile
  
- AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile
  
- AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot
  
- AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile
  
- AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot
</b>

This algorithm has been tested and evaluated on the FSD50K dataset, but it can be used for any dataset of audio files.
The tags of the FSD50K dataset are the labels from the <b>AudioSet Ontology</b>, and are taken from dev/eval splits .csv files.

The dictionary 'realSoundsDataset_Creator_Dict' is bumped into a .json file for future reference in the same folder as the output subset/segmented dataset.
The same applies to the ground truth (labels/tags) values of the subset/segmented dataset; both .csv files and python dict (.json files) are created in the output dataset folder.

Up to May 16th 2023, the script is designed to work with the <b>soundata library</b> and the <b>FSD50K dataset</b>, but the code is designed to be easily-extensible to other loader libraries and datasets as well.

### FSD50K (canonical dataset folder not included in this repo)

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

This folder contains the virtual environment used to run the script Creation_of_Audio_segmentedSubsets_of_datasets.py.

See requirements.txt for a list of installed packages (in my System, soundata was installed but, when tried to run the script, an error occurred for a missing numpy wheel; I hence updated numpy).
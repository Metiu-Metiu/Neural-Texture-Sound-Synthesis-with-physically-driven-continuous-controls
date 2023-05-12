import soundata
from enum import Enum
import collections

class Loader_Library(Enum):
    SOUNDATA = 1

class Subset_Tags_Policy(Enum): # all computations are made regardless of tags order in all lists
    AllAndOnlySubsetTags_ArePresentInCanonicalDatasetFile = 1 
    AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile = 2
    AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot = 3
    AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile = 4
    AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot = 5

############## INPUT VARIABLES here ##############
realSoundsDataset_Creator_Dict = {
    'canonicalDatasetLoader_Settings': {
        'datasetLoader_Library': Loader_Library.SOUNDATA,
        'datasetLoader_Name': 'fsd50k', # should be supported by the loader library you chose to use (exactly same name as in the loader library)
        # 'canonicalDataset_RemoteURL': 'https://soundata.readthedocs.io/en/latest/source/tutorial.html#accessing-data-remotely', # Google Drive link to the canonical version of the dataset 
        'canonicalDataset_LocationPath': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/FSD50K',
        'download_CanonicalDataset': False, # either True or False
        'validate_CanonicalDataset': False, # either True or False
    },

    'canonicalDatasetAugmentation_Settings': {
        'segment_AudioClips': True, # either True or False
        'segments_Length': float(3.0), # seconds,
    },

    'subset_Settings': {
        'createSubset': True, # either True or False
        'tags_ToExtractFromCanonicalDataset': ['Water'], # these labels will be used to create a partial subset of the canonical dataset with only audio files with these labels
        'tags_ToAvoidFromCanonicalDataset': ['Gurgling'], # these labels will be used to create a partial subset of the canonical dataset with only audio files with these labels
        'subsetTags_Policy': Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot, 
    },

    'outputDataset_Settings': {
        'outputDataset_CreationFolder': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/FSD50K_subset',
    }
}
############################################# end INPUT VARIABLES

datasetTags = ['Water', 'Ocean', 'Gurgling']
subsetTags = ['Ocean', 'Water']
subetTags_ToAvoid = ['Pompe']

if realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AllAndOnlySubsetTags_ArePresentInCanonicalDatasetFile:
    if collections.Counter(subsetTags) == collections.Counter(datasetTags):
        print('AllAndOnlySubsetTags_ArePresentInCanonicalDatasetFile')
elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile:
    if all(tag in datasetTags for tag in subsetTags):
        print('AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile')
elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot:
    if all(tag in datasetTags for tag in subsetTags):
        if not any(tag in datasetTags for tag in subetTags_ToAvoid):
            print('AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot')
elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile:
    if any(tag in datasetTags for tag in subsetTags):
        print('AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile')
elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot:
    if any(tag in datasetTags for tag in subsetTags):
        if not any(tag in datasetTags for tag in subetTags_ToAvoid):
            print('AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot')



'''
if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['datasetLoader_Library'] == Loader_Library.SOUNDATA:
    dataset_Loader = soundata.initialize(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['datasetLoader_Name'], data_home = realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath'])
    if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['download_CanonicalDataset']:
        dataset_Loader.download(cleanup = True)  # download the canonical version of the dataset
    if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['validate_CanonicalDataset']:
        dataset_Loader.validate()  # validate that all the expected files are there

    # example_clip = dataset_Loader.choice_clip()  # choose a random example clip
    # print(example_clip.audio_path)  
    # print(example_clip.audio[1]) 

    subsetDataset_NoAugm_Size = 0
    devCsvFile_Dict = dataset_Loader.load_ground_truth('/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/FSD50K/FSD50K.ground_truth/dev.csv')[0] # load the ground truth labels for the dataset
    for key, value in devCsvFile_Dict.items():
        if realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AllSubsetTagsArePresentInCanonicalDatasetFile or realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AllSubsetTagsArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot:
            if collections.Counter(value['tags']) == collections.Counter(realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToExtractFromCanonicalDataset']):
                if realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AllSubsetTagsArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot:
                    for tagToAvoid in realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToAvoidFromCanonicalDataset']:
                        if tagToAvoid in value['tags']:
                            break
                    print(key, value)
                    subsetDataset_NoAugm_Size += 1
                    break
        elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTagIsPresentInCanonicalDatasetFile or realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTagIsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot:
            skipThisClip = False
            for tagToExtract in realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToExtractFromCanonicalDataset']:
                if realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTagIsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot:
                    for tagToAvoid in realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToAvoidFromCanonicalDataset']:
                        if tagToAvoid in value['tags']:
                            print(f'{key} : {Subset_Tags_Policy.AtLeastOneSubsetTagIsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot.name}')
                            skipThisClip = True
                            break
                    if skipThisClip:
                        break
                if tagToExtract in value['tags']:
                    print(key, value)
                    subsetDataset_NoAugm_Size += 1
                    break
    extractedTags = realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToExtractFromCanonicalDataset']
    print(f'Found {subsetDataset_NoAugm_Size} Audio files with tags {extractedTags}')

# go through each clip in the dataset and check its label
# if the label is one of the labels we want to extract, then get the clip's audio
# if segment_AudioClips is True, segment the audio
# finally, copy all the segments in the subset dataset folder
# continue the loop until all clips in the dataset have been processed 
'''
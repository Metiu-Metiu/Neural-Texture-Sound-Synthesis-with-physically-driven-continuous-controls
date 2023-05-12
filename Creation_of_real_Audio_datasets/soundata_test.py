import soundata
from enum import Enum
import collections
import os
import shutil # for copying files in case no Audio segmentation is needed
import essentia.standard as essentia

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
        'audio_Files_Format' : '.wav'
    },

    'canonicalDatasetAugmentation_Settings': {
        'segment_AudioClips': True, # either True or False
        'segments_Length_Secs': float(3.0), # seconds,
    },

    'subset_Settings': {
        'createSubset': True, # either True or False
        'tags_ToExtractFromCanonicalDataset': list(['Water']), # these labels will be used to create a partial subset of the canonical dataset with only audio files with these labels
        'tags_ToAvoidFromCanonicalDataset': list(['Rain']), # these labels will be used to create a partial subset of the canonical dataset with only audio files with these labels
        'subsetTags_Policy': Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot, 
    },

    'outputDataset_Settings': {
        'outputDataset_CreationFolder': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/FSD50K_subset',
    }
}
############################################# end INPUT VARIABLES

##########
def do_CanonicalAndSubsetTags_Match_AccordingToSubsetTagsPolicy(datasetTags, subsetTags, subetTags_ToAvoid):
    if realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AllAndOnlySubsetTags_ArePresentInCanonicalDatasetFile:
        if collections.Counter(subsetTags) == collections.Counter(datasetTags):
            # print('AllAndOnlySubsetTags_ArePresentInCanonicalDatasetFile')
            return True
    elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile:
        if all(tag in datasetTags for tag in subsetTags):
            # print('AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile')
            return True
    elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot:
        if all(tag in datasetTags for tag in subsetTags):
            if not any(tag in datasetTags for tag in subetTags_ToAvoid):
                # print('AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot')
                return True
    elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile:
        if any(tag in datasetTags for tag in subsetTags):
            # print('AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile')
            return True
    elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot:
        if any(tag in datasetTags for tag in subsetTags):
            if not any(tag in datasetTags for tag in subetTags_ToAvoid):
                # print('AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot')
                return True
    else:
        return False
##########

if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['datasetLoader_Library'] == Loader_Library.SOUNDATA:
    dataset_Loader = soundata.initialize(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['datasetLoader_Name'], data_home = os.path.abspath(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath']))
    if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['download_CanonicalDataset']:
        dataset_Loader.download(cleanup = True)  # download the canonical version of the dataset
    if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['validate_CanonicalDataset']:
        dataset_Loader.validate()  # validate that all the expected files are there

    # example_clip = dataset_Loader.choice_clip()  # choose a random example clip
    # print(example_clip.audio_path)  
    # print(example_clip.audio[1]) 

    # Create output dataset destination directory if it doesn't exist
    os.makedirs(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_CreationFolder']), exist_ok=True)

    subsetDataset_NoAugm_Size = 0
    subsetDataset_Augm_Size = 0
    canonicalDataset_DevAudioFilesFolder_Path = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath']), 'FSD50K.dev_audio')
    groundTruth_Dev_CsvFile_SubPath = 'FSD50K.ground_truth/dev.csv' # relative to canonicalDataset_LocationPath
    new_pgroundTruth_Dev_CsvFile_Path = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath']), groundTruth_Dev_CsvFile_SubPath)
    devCsvFile_Dict = dataset_Loader.load_ground_truth(new_pgroundTruth_Dev_CsvFile_Path)[0] # load the ground truth labels for the dataset
    if realSoundsDataset_Creator_Dict['subset_Settings']['createSubset']:
        for key, value in devCsvFile_Dict.items():
            canonicalFileName = key + realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['audio_Files_Format']
            if do_CanonicalAndSubsetTags_Match_AccordingToSubsetTagsPolicy(value['tags'], realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToExtractFromCanonicalDataset'], realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToAvoidFromCanonicalDataset']):
                if realSoundsDataset_Creator_Dict['canonicalDatasetAugmentation_Settings']['segment_AudioClips']:
                    # segment the audio
                    canonicalFileAudioWaveF_AndSR = dataset_Loader.load_audio(os.path.join(canonicalDataset_DevAudioFilesFolder_Path, canonicalFileName))
                    segmentSize_Samp = int(realSoundsDataset_Creator_Dict['canonicalDatasetAugmentation_Settings']['segments_Length_Secs'] * canonicalFileAudioWaveF_AndSR[1])
                    audioSegments = essentia.FrameGenerator(canonicalFileAudioWaveF_AndSR[0], frameSize = segmentSize_Samp, hopSize = segmentSize_Samp, startFromZero=True)
                    segmentNum = 1
                    if len(canonicalFileAudioWaveF_AndSR[0]) >= segmentSize_Samp:
                        for segment in audioSegments:
                            if len(segment) == segmentSize_Samp:
                                output_file_path = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_CreationFolder']), (str(key) + str('_') + str(segmentNum) + str(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['audio_Files_Format'])))
                                essentia.MonoWriter(filename = output_file_path)(segment)
                                segmentNum += 1
                                subsetDataset_Augm_Size += 1
                else:
                    shutil.copy2(os.path.join(canonicalDataset_DevAudioFilesFolder_Path, canonicalFileName), os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_CreationFolder']), canonicalFileName))
                    subsetDataset_NoAugm_Size += 1
    else:
        print('Why should you copy the entire caonical dataset if you don\'t want to create a subset of it?')
        exit()

    extractedTags = realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToExtractFromCanonicalDataset']
    if realSoundsDataset_Creator_Dict['canonicalDatasetAugmentation_Settings']['segment_AudioClips']:
        print(f'Found {subsetDataset_Augm_Size} Audio files with tags {extractedTags}')
    else:
        print(f'Found {subsetDataset_NoAugm_Size} Audio files with tags {extractedTags}')

# go through each clip in the dataset and check its label
# if the label is one of the labels we want to extract, then get the clip's audio
# if segment_AudioClips is True, segment the audio
# finally, copy all the segments in the subset dataset folder
# continue the loop until all clips in the dataset have been processed 

import soundata
from enum import Enum
import collections
import os
import shutil # for copying files in case no Audio segmentation is needed
import essentia.standard as essentia
import csv
import json

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
        'datasetLoader_Library': Loader_Library.SOUNDATA.name,
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
        'tags_ToExtractFromCanonicalDataset': list(['Water', 'Stream']), # these labels will be used to create a partial subset of the canonical dataset with only audio files with these labels
        'tags_ToAvoidFromCanonicalDataset': list(['Rain', 'Ocean']), # these labels will be used to create a partial subset of the canonical dataset with only audio files with these labels
        'subsetTags_Policy': Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot.name, 
    },

    'outputDataset_Settings': {
        'outputDataset_ParentFolder': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets',
        'outputDataset_FolderName': 'FSD50K_Water_Stream_subset', # also name of the .json file with this dictionary
        'outputDataset_DevSplit_SubFolderName' : '', # subfolder of outputDataset_FolderName, '' not to put the dev split in a subfolder
        'outputDataset_EvalSplit_SubFolderName' : '', # subfolder of outputDataset_FolderName,'' not to put the dev split in a subfolder
        'outputDataset_GroundTruthCsvFiles_SubFolderName' : '', # subfolder of outputDataset_FolderName,'' not to put the .csv files in a subfolder
    }
}
############################################# end INPUT VARIABLES

##########
def do_CanonicalAndSubsetTags_Match_AccordingToSubsetTagsPolicy(datasetTags, subsetTags, subetTags_ToAvoid):
    if realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AllAndOnlySubsetTags_ArePresentInCanonicalDatasetFile.name:
        if collections.Counter(subsetTags) == collections.Counter(datasetTags):
            # print('AllAndOnlySubsetTags_ArePresentInCanonicalDatasetFile')
            return True
    elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile.name:
        if all(tag in datasetTags for tag in subsetTags):
            # print('AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile')
            return True
    elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot.name:
        if all(tag in datasetTags for tag in subsetTags):
            if not any(tag in datasetTags for tag in subetTags_ToAvoid):
                # print('AtLeastAllSubsetTags_ArePresentInCanonicalDatasetFile_AndExcludedTagsAreNot')
                return True
    elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile.name:
        if any(tag in datasetTags for tag in subsetTags):
            # print('AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile')
            return True
    elif realSoundsDataset_Creator_Dict['subset_Settings']['subsetTags_Policy'] == Subset_Tags_Policy.AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot.name:
        if any(tag in datasetTags for tag in subsetTags):
            if not any(tag in datasetTags for tag in subetTags_ToAvoid):
                # print('AtLeastOneSubsetTag_IsPresentInCanonicalDatasetFile_AndExcludedTagsAreNot')
                return True
    else:
        return False
##########

if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['datasetLoader_Library'] == Loader_Library.SOUNDATA.name:
    dataset_Loader = soundata.initialize(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['datasetLoader_Name'], data_home = os.path.abspath(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath']))
    canonicalDevSplit_SubFolder_Name = 'FSD50K.dev_audio'
    canonicalEvalSplit_SubFolder_Name = 'FSD50K.eval_audio'
    if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['download_CanonicalDataset']:
        dataset_Loader.download(cleanup = True)  # download the canonical version of the dataset
    if realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['validate_CanonicalDataset']:
        dataset_Loader.validate()  # validate that all the expected files are there

    canonicalGroundTruthCsvFile_SubFolder_Name = 'FSD50K.ground_truth' # relative to canonicalDataset_LocationPath
    canonicalGroundTruthDevCsvFile_Name = 'dev.csv'
    canonicalGroundTruthEvalCsvFile_Name = 'eval.csv'

    os.makedirs(os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName'])), exist_ok=True)
    os.makedirs(os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_DevSplit_SubFolderName'])), exist_ok=True)
    os.makedirs(os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_EvalSplit_SubFolderName'])), exist_ok=True)
    os.makedirs(os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_GroundTruthCsvFiles_SubFolderName'])), exist_ok=True)

    if realSoundsDataset_Creator_Dict['canonicalDatasetAugmentation_Settings']['segment_AudioClips'] or realSoundsDataset_Creator_Dict['subset_Settings']['createSubset']:
        realSoundsDataset_Creator_Dict['outputDataset_Statistics'] = {}
    
    numberOfSplitsInDataset = 2
    for split in range(numberOfSplitsInDataset):
        subsetDataset_NoAugm_Size = 0
        subsetDataset_Augm_Size = 0
        if split == 0: # dev split
            groundTruth_CsvFile_Path = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath']), canonicalGroundTruthCsvFile_SubFolder_Name, canonicalGroundTruthDevCsvFile_Name)
            canonicalDataset_AudioFilesFolder_Path = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath']), canonicalDevSplit_SubFolder_Name)
            outputSubFolderSplitName = realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_DevSplit_SubFolderName']
            outputCsvFileName = str('dev') + str(".csv")
            outputCsvFileFieldnames = ['fname', 'labels', 'mids', 'split']
        elif split == 1: # eval split
            groundTruth_CsvFile_Path = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath']), canonicalGroundTruthCsvFile_SubFolder_Name, canonicalGroundTruthEvalCsvFile_Name)
            canonicalDataset_AudioFilesFolder_Path = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['canonicalDataset_LocationPath']), canonicalEvalSplit_SubFolder_Name)
            outputSubFolderSplitName = realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_EvalSplit_SubFolderName']
            outputCsvFileName = str('eval') + str(".csv")
            outputCsvFileFieldnames = ['fname', 'labels', 'mids']

        outputCsvFilePath = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_GroundTruthCsvFiles_SubFolderName']), outputCsvFileName)

        devCsvFile_Dict = dataset_Loader.load_ground_truth(groundTruth_CsvFile_Path)[0] # load the ground truth labels for the dataset
        with open(outputCsvFilePath, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = outputCsvFileFieldnames, dialect='excel')
            writer.writeheader()
            if realSoundsDataset_Creator_Dict['subset_Settings']['createSubset']:
                for key, value in devCsvFile_Dict.items():
                    canonicalFileName = key + realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['audio_Files_Format']
                    if do_CanonicalAndSubsetTags_Match_AccordingToSubsetTagsPolicy(value['tags'], realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToExtractFromCanonicalDataset'], realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToAvoidFromCanonicalDataset']):
                        if realSoundsDataset_Creator_Dict['canonicalDatasetAugmentation_Settings']['segment_AudioClips']:
                            canonicalFileAudioWaveF_AndSR = dataset_Loader.load_audio(os.path.join(canonicalDataset_AudioFilesFolder_Path, canonicalFileName))
                            segmentSize_Samp = int(realSoundsDataset_Creator_Dict['canonicalDatasetAugmentation_Settings']['segments_Length_Secs'] * canonicalFileAudioWaveF_AndSR[1])
                            audioSegments = essentia.FrameGenerator(canonicalFileAudioWaveF_AndSR[0], frameSize = segmentSize_Samp, hopSize = segmentSize_Samp, startFromZero=True, validFrameThresholdRatio = 1)
                            segmentNum = 1
                            if len(canonicalFileAudioWaveF_AndSR[0]) >= segmentSize_Samp:
                                for segment in audioSegments:
                                    outpuFileName = str(str(key) + str('_') + str(segmentNum))
                                    output_file_path = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName']), str(outputSubFolderSplitName), (str(outpuFileName) + str(realSoundsDataset_Creator_Dict['canonicalDatasetLoader_Settings']['audio_Files_Format'])))
                                    if os.path.exists(output_file_path):
                                        print(f'ERROR:  An Audio file you are trying to create ALREADY EXISTS at path {output_file_path} ; Exiting...')
                                        exit()
                                    essentia.MonoWriter(filename = output_file_path)(segment)
                                    csvRowDict = {'fname': str(outpuFileName), 'labels': value['tags'], 'mids': value['mids']}
                                    if split == 0: # only for dev split
                                        csvRowDict.update({'split': value['split']})
                                    writer.writerow(csvRowDict)
                                    segmentNum += 1
                                    subsetDataset_Augm_Size += 1
                        else:
                            shutil.copy2(os.path.join(canonicalDataset_AudioFilesFolder_Path, canonicalFileName), os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName']), str(outputSubFolderSplitName), canonicalFileName))
                            csvRowDict = {'fname': str(key), 'labels': value['tags'], 'mids': value['mids']}
                            if split == 0: # only for dev split
                                csvRowDict.update({'split': value['split']})
                            writer.writerow(csvRowDict)
                            subsetDataset_NoAugm_Size += 1
            else:
                print('Why should you copy the entire canonical dataset if you don\'t want to create a subset of it?')
                exit()

        extractedTags = realSoundsDataset_Creator_Dict['subset_Settings']['tags_ToExtractFromCanonicalDataset']
        if realSoundsDataset_Creator_Dict['canonicalDatasetAugmentation_Settings']['segment_AudioClips'] and realSoundsDataset_Creator_Dict['subset_Settings']['createSubset']: ##########
            if split == 0:
                print(f'CREATED DEV SPLIT OF SEGMENTED SUBSET DATASET')
                print(f'    Created {subsetDataset_Augm_Size} Audio files with tags {extractedTags}')
                realSoundsDataset_Creator_Dict['outputDataset_Statistics']['numberOfFilesCreated_In_Segmented_DevSplit_Subset_Dataset'] = subsetDataset_Augm_Size
            elif split == 1:
                print(f'CREATED EVAL SPLIT OF SEGMENTED SUBSET DATASET')
                print(f'    Created {subsetDataset_Augm_Size} Audio files with tags {extractedTags}')
                realSoundsDataset_Creator_Dict['outputDataset_Statistics']['numberOfFilesCreated_In_Segmented_EvalSplit_Subset_Dataset'] = subsetDataset_Augm_Size
        elif realSoundsDataset_Creator_Dict['canonicalDatasetAugmentation_Settings']['segment_AudioClips']: ##########
            if split == 0:
                print(f'CREATED DEV SPLIT OF SEGMENTED DATASET')
                print(f'    Created {subsetDataset_Augm_Size} Audio files with tags {extractedTags}')
                realSoundsDataset_Creator_Dict['outputDataset_Statistics']['numberOfFilesCreated_In_Segmented_DevSplit_Dataset'] = subsetDataset_Augm_Size
            elif split == 1:
                print(f'CREATED EVAL SPLIT OF SEGMENTED DATASET')
                print(f'    Created {subsetDataset_Augm_Size} Audio files with tags {extractedTags}')
                realSoundsDataset_Creator_Dict['outputDataset_Statistics']['numberOfFilesCreated_In_Segmented_EvalSplit_Dataset'] = subsetDataset_Augm_Size
        elif realSoundsDataset_Creator_Dict['subset_Settings']['createSubset']: ##########
            if split == 0:
                print(f'CREATED DEV SPLIT OF SUBSET DATASET')
                print(f'    Created {subsetDataset_Augm_Size} Audio files with tags {extractedTags}')
                realSoundsDataset_Creator_Dict['outputDataset_Statistics']['numberOfFilesCreated_In_DevSplit_Subset_Dataset'] = subsetDataset_NoAugm_Size
            elif split == 1:
                print(f'CREATED EVAL SPLIT OF SUBSET DATASET')
                print(f'    Created {subsetDataset_Augm_Size} Audio files with tags {extractedTags}')
                realSoundsDataset_Creator_Dict['outputDataset_Statistics']['numberOfFilesCreated_In_EvalSplit_Subset_Dataset'] = subsetDataset_NoAugm_Size

# create .json file with realSoundsDataset_Creator_Dict
os.makedirs(os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName'])), exist_ok=True)

jsonFileName = str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName']) + str(".json")
jsonFilePath = os.path.join(os.path.abspath(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_ParentFolder']), str(realSoundsDataset_Creator_Dict['outputDataset_Settings']['outputDataset_FolderName']), jsonFileName)
with open(jsonFilePath, 'w') as jsonfile:
    json.dump(realSoundsDataset_Creator_Dict, jsonfile, indent=4)
print(f'Finished writing {jsonFileName} .json file with realSoundsDataset_Creator_Dict')
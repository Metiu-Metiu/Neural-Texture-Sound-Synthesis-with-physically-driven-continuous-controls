import csv
import json
import os # for getting synthetic datasets' paths
import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram
from torch import nn

from torchsummary import summary

from Dataset_Wrapper import Dataset_Wrapper
from Neural_Networks import Convolutional_DynamicNet, train, test, perform_inference_byExtractingSynthesisControlParameters

############### INPUT VARIABLES ###############
configDict_JSonFilePath = '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/Neural Networks/Old/2D_CNN_SynthParamExtractor_June11_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm/2D_CNN_SynthParamExtractor_June11_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm.json'
###############################################

with open(configDict_JSonFilePath) as configDict_JSonFile:
    configDict = json.load(configDict_JSonFile)

torch.manual_seed(configDict['pyTorch_General_Settings']['manual_seed'])
device = configDict['pyTorch_General_Settings']['device']
print(f'Using device: {device}')

with open(configDict['paths']['synthDataset_JSonFile_Path']) as synthDataset_JSonFile:
    synthDatasetGenerator_DescriptorDict = json.load(synthDataset_JSonFile)
syntheticDataset_LabelsNames = synthDatasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys()
synthDataset_AudioFiles_Directory = os.path.abspath(synthDatasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'])
synthDataset_CsvFilePath = os.path.join(synthDataset_AudioFiles_Directory, str(synthDatasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + '.csv'))

with open(configDict['paths']['realDataset_JSonFile_Path']) as realDataset_JSonFile:
    realDatasetGenerator_DescriptorDict = json.load(realDataset_JSonFile)
realDataset_AudioFiles_Directory_ParentFold = os.path.abspath(realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_ParentFolder'])
realDataset_AudioFiles_Directory = os.path.join(realDataset_AudioFiles_Directory_ParentFold, realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_FolderName'])
realDataset_CsvFilePath = os.path.join(realDataset_AudioFiles_Directory, str(realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_FolderName'] + '.csv'))

realDataset = Dataset_Wrapper(realDataset_AudioFiles_Directory, realDataset_CsvFilePath, configDict, transform = exec(configDict['neuralNetwork_Settings']['input_Transforms']))
realDataset_DL = DataLoader(realDataset, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = False)

# test inference with synthetic data (only eval split for speeding up the inference process)
synthDataset = Dataset_Wrapper(synthDataset_AudioFiles_Directory, synthDataset_CsvFilePath, None, device, transform = exec(configDict['neuralNetwork_Settings']['input_Transforms']))

numSamplesTrainSet = int(configDict['syntheticDataset_Settings']['splits']['train'] * len(synthDataset))
numSamplesValidationSet = int(configDict['syntheticDataset_Settings']['splits']['val'] * len(synthDataset))
numSamplesTestSet = int(configDict['syntheticDataset_Settings']['splits']['test'] * len(synthDataset))
if (numSamplesTrainSet + numSamplesValidationSet + numSamplesTestSet) < len(synthDataset):
    numSamplesTrainSet += len(synthDataset) - (numSamplesTrainSet + numSamplesValidationSet + numSamplesTestSet)

synthDS_TrainSplit, synthDS_EvalSplit, synthDS_TestSplit = torch.utils.data.random_split(synthDataset, [numSamplesTrainSet, numSamplesValidationSet, numSamplesTestSet])

synthDataset_Train, synthDataset_Valid, synthDataset_Test = torch.utils.data.random_split(synthDataset, [numSamplesTrainSet, numSamplesValidationSet, numSamplesTestSet])
synthDataset_DL = DataLoader(synthDataset_Train, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = False)


inputSignalLength = configDict['inputTransforms_Settings']['resample']['new_freq'] * int(configDict['validation']['nominal_AudioDurationSecs'])

inputTensor = torch.randn(1, 1, 1000, 1000)
inputTensor = realDataset.__getitem__(0)[0].unsqueeze(0) # unsqueeze adds a dimension of size 1 at the specified position
print(f'Input test from real dataset, shape : {inputTensor.shape}')

# expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
conv_1D_Net_PreTrained = Convolutional_DynamicNet(inputTensor.shape,
                        4,
                        numberOfFeaturesToExtract_IncremMultiplier_FromLayer1 = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFeaturesToExtract_IncremMultiplier_FromLayer1'],
                        numberOfConvLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfConvLayers'],
                        kernelSizeOfConvLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['kernelSizeOfConvLayers'],
                        strideOfConvLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['strideOfConvLayers'],
                        kernelSizeOfPoolingLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['kernelSizeOfPoolingLayers'],
                        strideOfPoolingLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['strideOfPoolingLayers'],
                        numberOfFullyConnectedLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFullyConnectedLayers'],
                        fullyConnectedLayers_InputSizeDecreaseFactor = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['fullyConnectedLayers_InputSizeDecreaseFactor']).to(device)     

print(f'Model output shape : {conv_1D_Net_PreTrained(inputTensor).shape}')
# summary(conv_1D_Net_PreTrained, inputTensor.shape)

pre_trained_model_path_parentFold = os.path.abspath(configDict['outputFilesSettings']['outputFolder_Path'])
checkpointFile_path = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + str('.pth')))
checkpointDictionary = torch.load(checkpointFile_path, map_location = torch.device(device))
conv_1D_Net_PreTrained.load_state_dict(checkpointDictionary['model_state_dict'])
print(conv_1D_Net_PreTrained)

syntheticDataset_LabelsNames = list(syntheticDataset_LabelsNames)
labelled_AudioFilesDict = perform_inference_byExtractingSynthesisControlParameters(synthDataset_DL, conv_1D_Net_PreTrained, syntheticDataset_LabelsNames)

if len(labelled_AudioFilesDict.keys()) == len(realDataset):
    print(f'{len(labelled_AudioFilesDict.keys())} files -exactly the size of the dataset- have been labelled.')
else:
    print(f'{len(labelled_AudioFilesDict.keys())} files have been labelled, but the dataset size is {len(realDataset)}.')

# COMMENT/UNCOMMENT THE FOLLOWING LINES FOR ACTUAL INFERENCE ON REAL DATASET
# dump .json and .csv file with labelled audio files to pre_trained_model_path_parentFold
# labelled_AudioFilesDict_StrIdentifier = str('_ExtractedAudioFilesLabels')
# labelled_AudioFilesDict_JSON_FilePath = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + labelled_AudioFilesDict_StrIdentifier + str('.json')))
# labelled_AudioFilesDict_CSV_FilePath = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + labelled_AudioFilesDict_StrIdentifier + str('.csv')))
# # test create .csv file with labelled synthetic audio files
# with open(labelled_AudioFilesDict_JSON_FilePath, 'w') as labelled_AudioFilesDict_JSON_File:
#     json.dump(labelled_AudioFilesDict, labelled_AudioFilesDict_JSON_File, indent = 4)

# COMMENT/UNCOMMENT THIS LINE FOR TESTS ON SYNTHETIC DATASET
labelled_SynthAudioFilesDict_CSV_FilePath = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + str('_ExtractedAudioFilesLabels') + str('_Synth') + str('.csv')))

csvFileFieldnames = ['AudioFileName'] # .csv file header name for audio files names column
csvFileFieldnames += syntheticDataset_LabelsNames # add synthesis control parameters names to the .csv file header

labelled_AudioFiles_ListOfDict = list()
for key in labelled_AudioFilesDict.keys():
    thisFileDict = dict()
    thisFileDict['AudioFileName'] = key
    for i in range(len(syntheticDataset_LabelsNames)):
        thisFileDict[syntheticDataset_LabelsNames[i]] = labelled_AudioFilesDict[key][syntheticDataset_LabelsNames[i]]
    labelled_AudioFiles_ListOfDict.append(thisFileDict)
# CHANGE THIS LINE FOR TESTS ON SYNTHETIC DATASET OR ACTUAL INFERENCE ON REAL DATASET
with open(labelled_SynthAudioFilesDict_CSV_FilePath, 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames=csvFileFieldnames, dialect='excel')
    writer.writeheader()
    for dict in labelled_AudioFiles_ListOfDict:
        writer.writerow(dict)
print(f'Finished writing .csv file with Audio Files labels.')

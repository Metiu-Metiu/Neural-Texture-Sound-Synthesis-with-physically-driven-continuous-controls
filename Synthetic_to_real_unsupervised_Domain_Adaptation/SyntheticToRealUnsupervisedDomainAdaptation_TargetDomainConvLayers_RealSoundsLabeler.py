import csv
import json
import os # for getting synthetic datasets' paths
import torch
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.transforms import Spectrogram
from torch import nn

from torchsummary import summary

from Dataset_Wrapper import Dataset_Wrapper
from Neural_Networks import Convolutional_DynamicNet, perform_inference_byExtractingSynthesisControlParameters

############### INPUT VARIABLES ###############
configDict_JSonFilePath = '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/Neural Networks/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm_DBScale/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm_DBScale_ConfigDict.json'
labelSyntheticDataset_RatherThanRealDataset = False # if True, the model will be inferenced on the synthetic dataset, otherwise on the real dataset
###############################################

with open(configDict_JSonFilePath) as configDict_JSonFile:
    configDict = json.load(configDict_JSonFile)

if configDict['pyTorch_General_Settings']['dtype'] == "torch.float32":
    configDict['pyTorch_General_Settings']['dtype'] = torch.float32

def jSonDecodeInputTransforms():
    transformsList = list()
    for transform_string in configDict['neuralNetwork_Settings']['input_Transforms']:
        if transform_string == 'torchaudio.transforms.Resample':
            transformsList.append(torchaudio.transforms.Resample(orig_freq = configDict['validation']['nominal_SampleRate'],
                                                                 new_freq = configDict['inputTransforms_Settings']['resample']['new_freq']))
        elif transform_string == 'torchaudio.transforms.MelSpectrogram':
            transformsList.append(torchaudio.transforms.MelSpectrogram(normalized = True,
                                                                       n_mels = configDict['inputTransforms_Settings']['spectrogram']['n_mels'],
                                                                       sample_rate = configDict['inputTransforms_Settings']['resample']['new_freq']))
    return transformsList
configDict['neuralNetwork_Settings']['input_Transforms'] = jSonDecodeInputTransforms()

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

# prepare synthetic dataset
synthDataset = Dataset_Wrapper(synthDataset_AudioFiles_Directory, synthDataset_CsvFilePath, configDict, transform = configDict['neuralNetwork_Settings']['input_Transforms'], applyNoise = True, supervised_Task = False)

# numSamplesTrainSet = int(configDict['syntheticDataset_Settings']['splits']['train'] * len(synthDataset))
# numSamplesValidationSet = int(configDict['syntheticDataset_Settings']['splits']['val'] * len(synthDataset))
# numSamplesTestSet = int(configDict['syntheticDataset_Settings']['splits']['test'] * len(synthDataset))
# if (numSamplesTrainSet + numSamplesValidationSet + numSamplesTestSet) < len(synthDataset):
#     numSamplesTrainSet += len(synthDataset) - (numSamplesTrainSet + numSamplesValidationSet + numSamplesTestSet)

# synthDataset_Train, synthDataset_Valid, synthDataset_Test = torch.utils.data.random_split(synthDataset, [numSamplesTrainSet, numSamplesValidationSet, numSamplesTestSet])
synthDataset_DL = DataLoader(synthDataset, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = False)

# prepare real dataset
realDataset = Dataset_Wrapper(realDataset_AudioFiles_Directory, realDataset_CsvFilePath, configDict, transform = configDict['neuralNetwork_Settings']['input_Transforms'], supervised_Task = False)
realDataset_DL = DataLoader(realDataset, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = False)

inputSignalLength = configDict['inputTransforms_Settings']['resample']['new_freq'] * int(configDict['validation']['nominal_AudioDurationSecs'])

inputTensor = torch.randn(1, 1, 1000, 1000)
inputTensor = realDataset.__getitem__(0)[0].unsqueeze(0) # unsqueeze adds a dimension of size 1 at the specified position
print(f'Input test from real dataset, shape : {inputTensor.shape}')

# expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
conv_PreTrained_2D_Net = Convolutional_DynamicNet(inputTensor.shape,
                        4, # synthDataset.numberOfLabels,
                        configDict).to(device)     

print(f'Model output shape : {conv_PreTrained_2D_Net(inputTensor).shape}')
# summary(conv_PreTrained_2D_Net, inputTensor.shape)

pre_trained_model_path_parentFold = os.path.abspath(configDict['outputFilesSettings']['outputFolder_Path'])
checkpointFile_path = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + str('.pth')))
checkpointDictionary = torch.load(checkpointFile_path, map_location = torch.device(device))

targetDomainConvLayersAdaptation_checkpointFile_path = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + str('_ConvLayers_TargetDomainAdaptation') + str('.pth')))
targetDomainConvLayersAdaptation_checkpointDict = torch.load(targetDomainConvLayersAdaptation_checkpointFile_path, map_location = torch.device(device))

pretrained_dict_conv = targetDomainConvLayersAdaptation_checkpointDict['model_state_dict']
print(f'    Pretrained dict conv layers : {pretrained_dict_conv.keys()}')

pretrained_dict_fc = {k: v for k, v in checkpointDictionary['model_state_dict'].items() if k.startswith('fc_blocks')}
print(f'    Pretrained dict fc layers : {pretrained_dict_fc.keys()}')

pretrained_dict_conv.update(pretrained_dict_fc) 
pretrained_fullModel_dict = pretrained_dict_conv
print(f'    Pretrained full model : {pretrained_fullModel_dict.keys()}')

conv_PreTrained_2D_Net.load_state_dict(pretrained_fullModel_dict)

print(conv_PreTrained_2D_Net)

syntheticDataset_LabelsNames = list(syntheticDataset_LabelsNames)
datasetLength = 0

if labelSyntheticDataset_RatherThanRealDataset:
    datasetLength = len(synthDataset_DL)
    labelled_AudioFilesDict = perform_inference_byExtractingSynthesisControlParameters(synthDataset_DL, conv_PreTrained_2D_Net, syntheticDataset_LabelsNames, configDict)
else:
    datasetLength = len(realDataset_DL)
    labelled_AudioFilesDict = perform_inference_byExtractingSynthesisControlParameters(realDataset_DL, conv_PreTrained_2D_Net, syntheticDataset_LabelsNames, configDict)

if len(labelled_AudioFilesDict.keys()) == datasetLength:
    print(f'{len(labelled_AudioFilesDict.keys())} files -exactly the size of the dataset- have been labelled.')
else:
    print(f'{len(labelled_AudioFilesDict.keys())} files have been labelled, but the dataset size is {len(realDataset)}.')

labelled_AudioFilesDict_StrIdentifier = str('_ExtractedAudioFilesLabels__DA')
if labelSyntheticDataset_RatherThanRealDataset:
    datasetTypeIdentifier = str('_SynthDataset')
else:
    datasetTypeIdentifier = str('_RealDataset')
labelled_AudioFilesDict_JSON_FilePath = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + labelled_AudioFilesDict_StrIdentifier + datasetTypeIdentifier + str('.json')))
labelled_AudioFilesDict_CSV_FilePath = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + labelled_AudioFilesDict_StrIdentifier + datasetTypeIdentifier + str('.csv')))
with open(labelled_AudioFilesDict_JSON_FilePath, 'w') as labelled_AudioFilesDict_JSON_File:
        json.dump(labelled_AudioFilesDict, labelled_AudioFilesDict_JSON_File, indent = 4)

csvFileFieldnames = ['AudioFileName'] # .csv file header name for audio files names column
csvFileFieldnames += syntheticDataset_LabelsNames # add synthesis control parameters names to the .csv file header

labelled_AudioFiles_ListOfDict = list()
for key in labelled_AudioFilesDict.keys():
    thisFileDict = dict()
    thisFileDict['AudioFileName'] = key
    for i in range(len(syntheticDataset_LabelsNames)):
        thisFileDict[syntheticDataset_LabelsNames[i]] = labelled_AudioFilesDict[key][syntheticDataset_LabelsNames[i]]
    labelled_AudioFiles_ListOfDict.append(thisFileDict)
with open(labelled_AudioFilesDict_CSV_FilePath, 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames=csvFileFieldnames, dialect='excel')
    writer.writeheader()
    for dict in labelled_AudioFiles_ListOfDict:
        writer.writerow(dict)
print(f'Finished writing .csv file with Audio Files labels.')

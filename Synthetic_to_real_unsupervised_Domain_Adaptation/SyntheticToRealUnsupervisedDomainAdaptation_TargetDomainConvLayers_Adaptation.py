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
from Neural_Networks import Convolutional_DynamicNet, SyntheticAndReal_Sounds_Classifier_FullyConnectedLayers, train_FCLayers_withFrozenConvLayers, test_FCLayers_withFrozenConvLayers, train_ConvLayers_withFrozenFCLayers, test_ConvLayers_withFrozenFCLayers
import time
import datetime

############### INPUT VARIABLES ###############
configDict_JSonFilePath = '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/Neural Networks/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm_DBScale/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm_DBScale_ConfigDict.json'
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

with open(configDict['paths']['realDataset_JSonFile_Path']) as realDataset_JSonFile:
    realDatasetGenerator_DescriptorDict = json.load(realDataset_JSonFile)
realDataset_AudioFiles_Directory_ParentFold = os.path.abspath(realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_ParentFolder'])
realDataset_AudioFiles_Directory = os.path.join(realDataset_AudioFiles_Directory_ParentFold, realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_FolderName'])
realDataset_CsvFilePath = os.path.join(realDataset_AudioFiles_Directory, str(realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_FolderName'] + '.csv'))

# prepare real dataset
realDataset = Dataset_Wrapper(realDataset_AudioFiles_Directory, realDataset_CsvFilePath, configDict, transform = configDict['neuralNetwork_Settings']['input_Transforms'], supervised_Task = False)

numSamplesTrainSet = int(configDict['syntheticDataset_Settings']['splits']['train'] * len(realDataset))
numSamplesValidationSet = int(configDict['syntheticDataset_Settings']['splits']['val'] * len(realDataset))
numSamplesTestSet = int(configDict['syntheticDataset_Settings']['splits']['test'] * len(realDataset))
if (numSamplesTrainSet + numSamplesValidationSet + numSamplesTestSet) < len(realDataset):
    numSamplesTrainSet += len(realDataset) - (numSamplesTrainSet + numSamplesValidationSet + numSamplesTestSet)

realDataset_Train, realDataset_Valid, realDataset_Test = torch.utils.data.random_split(realDataset, [numSamplesTrainSet, numSamplesValidationSet, numSamplesTestSet])

realDataset_Train_DL = DataLoader(realDataset_Train, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)
realDataset_Valid_DL = DataLoader(realDataset_Valid, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)
realDataset_Test_DL = DataLoader(realDataset_Test, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)

inputSignalLength = configDict['inputTransforms_Settings']['resample']['new_freq'] * int(configDict['validation']['nominal_AudioDurationSecs'])
inputTensor = torch.randn(1, 1, 1000, 1000)
inputTensor = realDataset.__getitem__(0)[0].unsqueeze(0) # unsqueeze adds a dimension of size 1 at the specified position
print(f'Input test from real dataset, shape : {inputTensor.shape}')

# expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
targetDomainConvLayersAdaptation_ConvLayers = Convolutional_DynamicNet(inputTensor.shape,
                        4, # synthDataset.numberOfLabels,
                        configDict,
                        createOnlyConvLayers = True).to(device)     

# print(f'Model output shape : {targetDomainConvLayersAdaptation_ConvLayers(inputTensor).shape}')
# summary(conv_PreTrained_2D_Net, inputTensor.shape)

pre_trained_model_path_parentFold = os.path.abspath(configDict['outputFilesSettings']['outputFolder_Path'])
checkpointConvLayersFile_path = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + str('.pth')))
checkpointConvLayersDictionary = torch.load(checkpointConvLayersFile_path, map_location = torch.device(device))
# filter out fully connected layers from model_state_dict
modelStateDict_OnlyConvLayers = {k: v for k, v in checkpointConvLayersDictionary['model_state_dict'].items() if k.startswith('conv_blocks')}
targetDomainConvLayersAdaptation_ConvLayers.load_state_dict(modelStateDict_OnlyConvLayers)

syntheticAndReal_Sound_Classifier_FCLayers = SyntheticAndReal_Sounds_Classifier_FullyConnectedLayers([1, 1, 256],
                        8,
                        2).to(device)     
print(syntheticAndReal_Sound_Classifier_FCLayers)

checkpointFCLayersFile_path = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + str('_FCLayers_SyntheticAndRealAudioClassifier') + str('.pth')))
checkpointFCLayersDictionary = torch.load(checkpointFCLayersFile_path, map_location = torch.device(device))
syntheticAndReal_Sound_Classifier_FCLayers.load_state_dict(checkpointFCLayersDictionary['model_state_dict'])
syntheticAndReal_Sound_Classifier_FCLayers.eval() # freeze fully connected layers

loss_Function = nn.BCELoss()
optimizer = torch.optim.Adam(targetDomainConvLayersAdaptation_ConvLayers.parameters(), lr = configDict['neuralNetwork_Settings']['learning_Rate'])

startTime = time.time()
train_ConvLayers_withFrozenFCLayers(syntheticAndReal_Sound_Classifier_FCLayers, # frozen model
                      targetDomainConvLayersAdaptation_ConvLayers, # model to be trained
                      realDataset_Train_DL,
                      realDataset_Valid_DL,
                      loss_Function,
                      optimizer,
                      device,
                      50, # number of epochs
                      configDict)
endTime = time.time()
trainingTimeElapsed = round(endTime - startTime)
trainingTimeElapsed = str(datetime.timedelta(seconds = trainingTimeElapsed))
print(f'Finished training.')

test_ConvLayers_withFrozenFCLayers(realDataset_Test_DL,
                     syntheticAndReal_Sound_Classifier_FCLayers,
                     targetDomainConvLayersAdaptation_ConvLayers,
                     loss_Function,
                     configDict)
print(f'Finished testing.')
import json
import os # for getting synthetic datasets' paths
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram
from torch import nn

import datetime
import time
from torchsummary import summary

from Dataset_Wrapper import Dataset_Wrapper
from Neural_Networks import Convolutional_DynamicNet, train, test
from Configuration_Dictionary import configDict

torch.manual_seed(configDict['pyTorch_General_Settings']['manual_seed'])
device = configDict['pyTorch_General_Settings']['device']
print(f'Using device: {device}')

os.makedirs(os.path.abspath(configDict['outputFilesSettings']['outputFolder_Path']), exist_ok=True)

########### processing input variables ###########
# create dict data structure out of the synth dataset descriptor .json file
with open(configDict['paths']['synthDataset_JSonFile_Path']) as synthDataset_JSonFile:
    synthDatasetGenerator_DescriptorDict = json.load(synthDataset_JSonFile)
    # print("Type:", type(synthDatasetGenerator_DescriptorDict))
    # print("\Dataset_General_Settings:", synthDatasetGenerator_DescriptorDict['Dataset_General_Settings'])
    # print("\Audio_Files_Settings:", synthDatasetGenerator_DescriptorDict['Audio_Files_Settings'])
synthDataset_AudioFiles_Directory = os.path.abspath(synthDatasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'])
synthDataset_GroundTruth_CsvFIlePath = os.path.join(synthDataset_AudioFiles_Directory, synthDatasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + ".csv")
##################################################

# Since the whole Thesis Project should be as transparent as possible with respect to the Audio Synthesis Engine used to
# synthesize the synthetic datasets, a synthetic dataset with uniform joint probability distribution is created, and
# train, validation and test splits are made out of it, rather than creating a new synthetic dataset for each split.
# This is to ensure that the validation and test splits are not biased towards the training split in any way (we know for sure they are expected to be different than the train split).
synthDataset = Dataset_Wrapper(synthDataset_AudioFiles_Directory, synthDataset_GroundTruth_CsvFIlePath, configDict['syntheticDataset_Settings']['rangeOfColumnNumbers_ToConsiderInCsvFile'], device, transform = configDict['neuralNetwork_Settings']['input_Transforms'])
synthDS_TrainSplit, synthDS_EvalSplit, synthDS_TestSplit = torch.utils.data.random_split(synthDataset, [int(configDict['syntheticDataset_Settings']['splits']['train'] * len(synthDataset)), int(configDict['syntheticDataset_Settings']['splits']['val'] * len(synthDataset)), int(configDict['syntheticDataset_Settings']['splits']['test'] * len(synthDataset))])
print(f'Number of samples in train split : {len(synthDS_TrainSplit)}')
print(f'Number of samples in validation split : {len(synthDS_EvalSplit)}')
print(f'Number of samples in test split : {len(synthDS_TestSplit)}')

synthDS_TrainDL = DataLoader(synthDS_TrainSplit, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)
synthDS_ValDL = DataLoader(synthDS_EvalSplit, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True, drop_last = True)
synthDS_TestDL = DataLoader(synthDS_TestSplit, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)

inputSignalLength = configDict['inputTransforms_Settings']['resample']['new_freq'] * int(configDict['validation']['nominal_AudioDurationSecs'])
# Example input tensor shape for 1D Convolutions-based NN: (batch_size, channels, width)
# Example input tensor shape for 2D Convolutions-based NN: (batch_size, channels, height, width)
inputTensor = torch.randn(1, 1, 1000, 1000)
inputTensor = synthDataset.__getitem__(0)[0].unsqueeze(0) # unsqueeze adds a dimension of size 1 at the specified position
print(f'Input test from synthetic dataset, shape : {inputTensor.shape}')

configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['inputTensor_Shape'] = inputTensor.shape
configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFeatures_ToExtract'] = synthDataset.numberOfLabels

# expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
conv_1D_Net = Convolutional_DynamicNet(inputTensor.shape,
                        synthDataset.numberOfLabels,
                        numberOfFeaturesToExtract_IncremMultiplier_FromLayer1 = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFeaturesToExtract_IncremMultiplier_FromLayer1'],
                        numberOfConvLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfConvLayers'],
                        kernelSizeOfConvLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['kernelSizeOfConvLayers'],
                        strideOfConvLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['strideOfConvLayers'],
                        kernelSizeOfPoolingLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['kernelSizeOfPoolingLayers'],
                        strideOfPoolingLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['strideOfPoolingLayers'],
                        numberOfFullyConnectedLayers = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFullyConnectedLayers'],
                        fullyConnectedLayers_InputSizeDecreaseFactor = configDict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['fullyConnectedLayers_InputSizeDecreaseFactor']).to(device)     
print(f'Model output shape : {conv_1D_Net(inputTensor).shape}')
print(f'Labels data from dataset, shape : {synthDataset.__getitem__(0)[1].shape}')
# summary(conv_1D_Net, inputTensor.shape)

loss_Function = nn.L1Loss(reduction='mean') # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss (reduction -mean or sum- is applied over the batch size)
optimizer = torch.optim.Adam(conv_1D_Net.parameters(), lr=0.001)

startTime = time.time()
train(conv_1D_Net, synthDS_TrainDL, synthDS_ValDL, loss_Function, optimizer, device, configDict['neuralNetwork_Settings']['number_Of_Epochs'])
endTime = time.time()
trainingTimeElapsed = round(endTime - startTime)
trainingTimeElapsed = str(datetime.timedelta(seconds = trainingTimeElapsed))
print(f'Finished training.')

# TODO make sure execution does not stop here because of Error KeyNotFound in configDict
configDict['statistics']['dateAndTime_WhenTrainingFinished_dd/mm/YY H:M:S'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
configDict['statistics']['elapsedTime_WhileTraining'] = trainingTimeElapsed

test(synthDS_TestDL, conv_1D_Net, loss_Function)
print(f'Finished testing.')

# dump output files
# CAREFUL HERE THIS CAN CAUSE PROBLEMS TO SCRIPTS GETTING configDict['neuralNetwork_Settings']['input_Transforms'] AFTER TO STRING CONVERSION
# TODO: PROBABLY IT IS BETTER TO COPY configDict AT THE START OF EACH SCRIPT AND ONLY MODIFY THE COPY BEFORE DUMPING IT TO JSON FILES
configDict['pyTorch_General_Settings']['device'] = str(configDict['pyTorch_General_Settings']['device'])
configDict['pyTorch_General_Settings']['dtype'] = str(configDict['pyTorch_General_Settings']['dtype'])
configDict['neuralNetwork_Settings']['input_Transforms'] = str(configDict['neuralNetwork_Settings']['input_Transforms'])
jsonFileName = configDict['outputFilesSettings']['jSonFile_WithThisDict_Name'] + str(".json")
jsonFilePath = os.path.join(configDict['outputFilesSettings']['outputFolder_Path'], jsonFileName)
with open(jsonFilePath, 'w') as jsonfile:
    json.dump(configDict, jsonfile, indent=4)

print('Finished training and dumping output files.')
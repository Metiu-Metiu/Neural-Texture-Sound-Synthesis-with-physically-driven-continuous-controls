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

config_Dict = configDict # COPY THE DICTIONARY STORED IN THE FILE IN ORDER TO AVOID OVERWRITING IT BY MISTAKE

torch.manual_seed(config_Dict['pyTorch_General_Settings']['manual_seed'])
device = config_Dict['pyTorch_General_Settings']['device']
print(f'Using device: {device}')

os.makedirs(os.path.abspath(config_Dict['outputFilesSettings']['outputFolder_Path']), exist_ok=True)

########### processing input variables ###########
# create dict data structure out of the synth dataset descriptor .json file
with open(config_Dict['paths']['synthDataset_JSonFile_Path']) as synthDataset_JSonFile:
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
synthDataset = Dataset_Wrapper(synthDataset_AudioFiles_Directory, synthDataset_GroundTruth_CsvFIlePath, config_Dict['syntheticDataset_Settings']['rangeOfColumnNumbers_ToConsiderInCsvFile'], device, transform = config_Dict['neuralNetwork_Settings']['input_Transforms'])
synthDS_TrainSplit, synthDS_EvalSplit, synthDS_TestSplit = torch.utils.data.random_split(synthDataset, [int(config_Dict['syntheticDataset_Settings']['splits']['train'] * len(synthDataset)), int(config_Dict['syntheticDataset_Settings']['splits']['val'] * len(synthDataset)), int(config_Dict['syntheticDataset_Settings']['splits']['test'] * len(synthDataset))])
print(f'Number of samples in train split : {len(synthDS_TrainSplit)}')
print(f'Number of samples in validation split : {len(synthDS_EvalSplit)}')
print(f'Number of samples in test split : {len(synthDS_TestSplit)}')

synthDS_TrainDL = DataLoader(synthDS_TrainSplit, batch_size = config_Dict['neuralNetwork_Settings']['batch_size'], shuffle = True)
synthDS_ValDL = DataLoader(synthDS_EvalSplit, batch_size = config_Dict['neuralNetwork_Settings']['batch_size'], shuffle = True, drop_last = True)
synthDS_TestDL = DataLoader(synthDS_TestSplit, batch_size = config_Dict['neuralNetwork_Settings']['batch_size'], shuffle = True)

# Example input tensor shape for 1D Convolutions-based NN: (batch_size, channels, width)
# Example input tensor shape for 2D Convolutions-based NN: (batch_size, channels, height, width)
inputTensor_WithBatchDim = synthDataset.__getitem__(0)[0].unsqueeze(0) # unsqueeze adds a dimension of size 1 at the specified position (in this case simulates the batch size dimension)

# expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
conv_1D_Net = Convolutional_DynamicNet(inputTensor_WithBatchDim.shape,
                        synthDataset.numberOfLabels,
                        numberOfFeaturesToExtract_IncremMultiplier_FromLayer1 = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFeaturesToExtract_IncremMultiplier_FromLayer1'],
                        numberOfConvLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfConvLayers'],
                        kernelSizeOfConvLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['kernelSizeOfConvLayers'],
                        strideOfConvLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['strideOfConvLayers'],
                        kernelSizeOfPoolingLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['kernelSizeOfPoolingLayers'],
                        strideOfPoolingLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['strideOfPoolingLayers'],
                        numberOfFullyConnectedLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFullyConnectedLayers'],
                        fullyConnectedLayers_InputSizeDecreaseFactor = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['fullyConnectedLayers_InputSizeDecreaseFactor']).to(device)     
summary(conv_1D_Net, synthDataset.__getitem__(0)[0].shape)

config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['inputTensor_Shape'] = inputTensor_WithBatchDim.shape
config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFeatures_ToExtract'] = synthDataset.numberOfLabels

loss_Function = nn.L1Loss(reduction = config_Dict['neuralNetwork_Settings']['loss']['reduction']) # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss (reduction -mean or sum- is applied over the batch size)
optimizer = torch.optim.Adam(conv_1D_Net.parameters(), lr = config_Dict['neuralNetwork_Settings']['learning_Rate'])

startTime = time.time()
train(conv_1D_Net, synthDS_TrainDL, synthDS_ValDL, loss_Function, optimizer, device, config_Dict['neuralNetwork_Settings']['number_Of_Epochs'])
endTime = time.time()
trainingTimeElapsed = round(endTime - startTime)
trainingTimeElapsed = str(datetime.timedelta(seconds = trainingTimeElapsed))
print(f'Finished training.')

config_Dict['statistics']['dateAndTime_WhenTrainingFinished_dd/mm/YY H:M:S'] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
config_Dict['statistics']['elapsedTime_WhileTraining'] = trainingTimeElapsed

test(synthDS_TestDL, conv_1D_Net, loss_Function)
print(f'Finished testing.')

config_Dict['pyTorch_General_Settings']['device'] = str(config_Dict['pyTorch_General_Settings']['device'])
config_Dict['pyTorch_General_Settings']['dtype'] = str(config_Dict['pyTorch_General_Settings']['dtype'])
config_Dict['neuralNetwork_Settings']['input_Transforms'] = str(config_Dict['neuralNetwork_Settings']['input_Transforms'])
jsonFileName = config_Dict['outputFilesSettings']['jSonFile_WithThisDict_Name'] + str(".json")
jsonFilePath = os.path.join(config_Dict['outputFilesSettings']['outputFolder_Path'], jsonFileName)
with open(jsonFilePath, 'w') as jsonfile:
    json.dump(config_Dict, jsonfile, indent=4)

print('Finished training and dumping output files.')
import json
import os # for getting synthetic datasets' paths
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram
from torch import nn

import time
from torchsummary import summary

from Dataset_Wrapper import Dataset_Wrapper
from Neural_Networks import Convolutional_DynamicNet, train, train_single_epoch
from Configuration_Dictionary import configDict

torch.manual_seed(configDict['pyTorch_General_Settings']['manual_seed'])
device = configDict['pyTorch_General_Settings']['device']
print(f'Using device: {device}')

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

synthDS_TrainDL = DataLoader(synthDS_TrainSplit, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)
synthDS_EvalDL = DataLoader(synthDS_EvalSplit, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)
synthDS_TestDL = DataLoader(synthDS_TestSplit, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)

inputSignalLength = 44100 * int(configDict['validation']['nominal_AudioDurationSecs'])
# Example input tensor shape for 1D Convolutions-based NN: (batch_size, channels, width)
# Example input tensor shape for 2D Convolutions-based NN: (batch_size, channels, height, width)
inputTensor = torch.randn(1, 1, 1000, 1000)
inputTensor = synthDataset.__getitem__(0)[0].unsqueeze(0) # unsqueeze adds a dimension of size 1 at the specified position
print(f'Input test from synthetic dataset, shape : {inputTensor.shape}')
# expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
conv_1D_Net = Convolutional_DynamicNet(inputTensor.shape,
                        synthDataset.numberOfLabels,
                        numberOfFeaturesToExtract_IncremMultiplier_FromLayer1 = 1,
                        numberOfConvLayers = 4,
                        kernelSizeOfConvLayers = 5,
                        strideOfConvLayers = 1,
                        kernelSizeOfPoolingLayers = 2,
                        strideOfPoolingLayers = 2,
                        numberOfFullyConnectedLayers = 8,
                        fullyConnectedLayers_InputSizeDecreaseFactor = 2).to(device)     
print(f'Model output shape : {conv_1D_Net(inputTensor).shape}')
print(f'Labels data from dataset, shape : {synthDataset.__getitem__(0)[1].shape}')
summary(conv_1D_Net, inputTensor)

loss_Function = nn.L1Loss()
optimizer = torch.optim.Adam(conv_1D_Net.parameters(), lr=0.001)
train(conv_1D_Net, synthDS_TrainDL, loss_Function, optimizer, device, configDict['neuralNetwork_Settings']['number_Of_Epochs'])

torch.save(conv_1D_Net.state_dict(), 'SynthesisControlParameters_Extractor_Network.pth')
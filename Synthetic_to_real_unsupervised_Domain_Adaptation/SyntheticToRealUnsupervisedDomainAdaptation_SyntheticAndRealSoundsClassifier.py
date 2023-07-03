import csv
import json
import os # for getting synthetic datasets' paths
import torch
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.transforms import Spectrogram
from torch import nn

from torchsummary import summary

from Dataset_Wrapper import Dataset_Wrapper, Mixed_Dataset_Wrapper
from Neural_Networks import Convolutional_DynamicNet, SyntheticAndReal_Sounds_Classifier_FullyConnectedLayers, train_FCLayers_withFrozenConvLayers, test_FCLayers_withFrozenConvLayers
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

# prepare mixed dataset
synthAndRealDataset = Mixed_Dataset_Wrapper(configDict, transform = configDict['neuralNetwork_Settings']['input_Transforms'])

numSamplesTrainSet = int(configDict['syntheticDataset_Settings']['splits']['train'] * len(synthAndRealDataset))
numSamplesValidationSet = int(configDict['syntheticDataset_Settings']['splits']['val'] * len(synthAndRealDataset))
numSamplesTestSet = int(configDict['syntheticDataset_Settings']['splits']['test'] * len(synthAndRealDataset))
if (numSamplesTrainSet + numSamplesValidationSet + numSamplesTestSet) < len(synthAndRealDataset):
    numSamplesTrainSet += len(synthAndRealDataset) - (numSamplesTrainSet + numSamplesValidationSet + numSamplesTestSet)

synthAndRealDataset_Train, synthAndRealDataset_Valid, synthAndRealDataset_Test = torch.utils.data.random_split(synthAndRealDataset, [numSamplesTrainSet, numSamplesValidationSet, numSamplesTestSet])
synthAndRealDataset_Train_DL = DataLoader(synthAndRealDataset_Train, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)
synthAndRealDataset_Valid_DL = DataLoader(synthAndRealDataset_Valid, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)
synthAndRealDataset_Test_DL = DataLoader(synthAndRealDataset_Test, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)

inputSignalLength = configDict['inputTransforms_Settings']['resample']['new_freq'] * int(configDict['validation']['nominal_AudioDurationSecs'])
inputTensor = torch.randn(1, 1, 1000, 1000)
inputTensor = synthAndRealDataset_Test.__getitem__(0)[0].unsqueeze(0) # unsqueeze adds a dimension of size 1 at the specified position
print(f'Input test from real dataset, shape : {inputTensor.shape}')

# expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
syntheticAndReal_Sound_Classifier_ConvLayers = Convolutional_DynamicNet(inputTensor.shape,
                        4, # synthDataset.numberOfLabels,
                        configDict,
                        createOnlyConvLayers = True).to(device)     

print(f'Model output shape : {syntheticAndReal_Sound_Classifier_ConvLayers(inputTensor).shape}')
# summary(conv_PreTrained_2D_Net, inputTensor.shape)

pre_trained_model_path_parentFold = os.path.abspath(configDict['outputFilesSettings']['outputFolder_Path'])
checkpointFile_path = os.path.join(pre_trained_model_path_parentFold, (configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name'] + str('.pth')))
checkpointDictionary = torch.load(checkpointFile_path, map_location = torch.device(device))
# filter out fully connected layers from model_state_dict
modelStateDict_OnlyConvLayers = {k: v for k, v in checkpointDictionary['model_state_dict'].items() if k.startswith('conv_blocks')}
syntheticAndReal_Sound_Classifier_ConvLayers.load_state_dict(modelStateDict_OnlyConvLayers)
syntheticAndReal_Sound_Classifier_ConvLayers.eval() # freeze convolutional layers

syntheticAndReal_Sound_Classifier_FCLayers = SyntheticAndReal_Sounds_Classifier_FullyConnectedLayers([1, 1, 256],
                        8,
                        2).to(device)     
print(syntheticAndReal_Sound_Classifier_FCLayers)

loss_Function = nn.BCELoss()
optimizer = torch.optim.Adam(syntheticAndReal_Sound_Classifier_FCLayers.parameters(), lr = configDict['neuralNetwork_Settings']['learning_Rate'])

startTime = time.time()
train_FCLayers_withFrozenConvLayers(syntheticAndReal_Sound_Classifier_ConvLayers, # frozen model
                      syntheticAndReal_Sound_Classifier_FCLayers, # model to be trained
                      synthAndRealDataset_Train_DL,
                      synthAndRealDataset_Valid_DL,
                      loss_Function,
                      optimizer,
                      device,
                      50, # number of epochs
                      configDict)
endTime = time.time()
trainingTimeElapsed = round(endTime - startTime)
trainingTimeElapsed = str(datetime.timedelta(seconds = trainingTimeElapsed))
print(f'Finished training.')

test_FCLayers_withFrozenConvLayers(synthAndRealDataset_Test_DL,
                     syntheticAndReal_Sound_Classifier_ConvLayers,
                     syntheticAndReal_Sound_Classifier_FCLayers,
                     loss_Function,
                     configDict)
print(f'Finished testing.')
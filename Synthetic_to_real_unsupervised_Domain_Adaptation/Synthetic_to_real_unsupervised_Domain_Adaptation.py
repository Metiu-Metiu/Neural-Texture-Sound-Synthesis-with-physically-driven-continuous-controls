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
from Neural_Networks import FF_NN, SynthesisControlParameters_Extractor_Network, train, train_single_epoch
from Configuration_Dictionary import configDict

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
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

# model = FF_NN(24000)
# x = torch.randn(24000)
# print(model(x).shape)

# model = SynthesisControlParameters_Extractor_Network()
# x = torch.randn(1, 1, 401, 61)
# print(model(x).shape)

synthDataset = Dataset_Wrapper(synthDataset_AudioFiles_Directory, synthDataset_GroundTruth_CsvFIlePath, configDict['syntheticDataset_Settings']['rangeOfColumnNumbers_ToConsiderInCsvFile'], device, transform = configDict['neuralNetwork_Settings']['input_Transforms'])
train_dataloader = DataLoader(synthDataset, batch_size = configDict['neuralNetwork_Settings']['batch_size'], shuffle = True)
item = synthDataset.__getitem__(0)
print(item[0].shape)
print(item[1].shape)
print(type(item))

# Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

'''
# nn_Model = SynthesisControlParameters_Extractor_Network().to(device)     
nn_Model = FF_NN(24000).to(device)     

print(synthDataset.__getitem__(0)[0].shape)
# print(len(synthDataset.__getitem__(0)[1]))
# summary(nn_Model, 24000)

loss_Function = nn.MSELoss()
optimizer = torch.optim.Adam(nn_Model.parameters(), lr=0.01)

train(nn_Model, train_dataloader, loss_Function, optimizer, device, configDict['neuralNetwork_Settings']['number_Of_Epochs'])

torch.save(nn_Model.state_dict(), 'SynthesisControlParameters_Extractor_Network.pth')
'''
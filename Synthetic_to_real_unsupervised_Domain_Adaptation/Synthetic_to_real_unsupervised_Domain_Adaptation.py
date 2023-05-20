import json
import os # for getting synthetic datasets' paths
import pandas # for reading .csv files

import torch
import torchaudio
from torch.utils.data import DataLoader

from torchaudio.transforms import Spectrogram
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from IPython.display import Audio, display

########### input variables #############################################
syntheticToReal_Unsupervised_DA = {
    'paths': {
        # Path of the .json file containing the descriptor dictionary of the synthetic dataset
        'synthDataset_JSonFile_Path': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/SDT_FluidFlow_dataset/SDT_FluidFlow.json',
    },

    'syntheticDataset_Settings': {
        # The first and last column number to consider in the .csv file containing the ground truth of interest
        # Audio file name is always column n. 0
        'rangeOfColumnNumbers_ToConsiderInCsvFile': [2, 7],
    },

    'realDataset_Settings': {
    # The first and last column number to consider in the .csv file containing the ground truth of interest
    # Audio file name is always column n. 0
    'rangeOfColumnNumbers_ToConsiderInCsvFile': None,
    },

    'validation': {
        'validate_AudioDatasets': True, # either true or false, checks if the variables below are valid
        'nominal_SampleRate': 44100, # int
        'nominal_NumOfAudioChannels': 1, # int
        'nominal_AudioFileExtension': '.wav', # string
        'nominal_BitQuantization': 16, # int
        'nominal_AudioDurationSecs': 3.0, # float
    }
}
#########################################################################

########### processing input variables ###########
# create dict data structure out of the synth dataset descriptor .json file
with open(syntheticToReal_Unsupervised_DA['paths']['synthDataset_JSonFile_Path']) as synthDataset_JSonFile:
    synthDatasetGenerator_DescriptorDict = json.load(synthDataset_JSonFile)
    # print("Type:", type(synthDatasetGenerator_DescriptorDict))
    # print("\Dataset_General_Settings:", synthDatasetGenerator_DescriptorDict['Dataset_General_Settings'])
    # print("\Audio_Files_Settings:", synthDatasetGenerator_DescriptorDict['Audio_Files_Settings'])
synthDataset_AudioFiles_ParentFolderPath = os.path.abspath(synthDatasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'])
synthDataset_GroundTruth_CsvFIlePath = os.path.join(synthDataset_AudioFiles_ParentFolderPath, synthDatasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + ".csv")
##################################################

################################################## UTILS ##################################################
def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate, autoplay = True, embed = True))
    else:
        raise ValueError("Waveform with more than 1 channel are not supported.")

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=True)

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=True)
######################################################################################################################################################
    
# DATASET CLASS (extends torch.utils.data.Dataset) 
class Sounds_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, audioFiles_ParentFolderPath_, groundTruth_CsvFIlePath_, rangeOfColumnNumbers_ToConsiderInCsvFile_, transform = Spectrogram(n_fft=800), target_transform = None):
        self.labels = pandas.read_csv(groundTruth_CsvFIlePath_)
        self.rangeOfColumnNumbers_ToConsiderInCsvFile = rangeOfColumnNumbers_ToConsiderInCsvFile_
        self.audioFiles_ParentFolderPath = audioFiles_ParentFolderPath_
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audioFile_path = os.path.join(self.audioFiles_ParentFolderPath, self.labels.iloc[idx, 0])
        if os.path.exists(audioFile_path):
            if syntheticToReal_Unsupervised_DA['validation']['validate_AudioDatasets']:
                assert audioFile_path.endswith(syntheticToReal_Unsupervised_DA['validation']['nominal_AudioFileExtension']), f"Error while loading {audioFile_path} : Audio file extension is not valid"
                audioFile_Metadata = torchaudio.info(audioFile_path)
                assert audioFile_Metadata.sample_rate == syntheticToReal_Unsupervised_DA['validation']['nominal_SampleRate'], f"Error while loading {audioFile_path} : Sample rate is not valid"
                assert audioFile_Metadata.num_frames == syntheticToReal_Unsupervised_DA['validation']['nominal_AudioDurationSecs'] * syntheticToReal_Unsupervised_DA['validation']['nominal_SampleRate'], f"Error while loading {audioFile_path} : Audio duration is not valid"
                assert audioFile_Metadata.num_channels == syntheticToReal_Unsupervised_DA['validation']['nominal_NumOfAudioChannels'], f"Error while loading {audioFile_path} : Number of audio channels is not valid"
                assert audioFile_Metadata.bits_per_sample == syntheticToReal_Unsupervised_DA['validation']['nominal_BitQuantization'], f"Error while loading {audioFile_path} : Bit quantization is not valid"
                # print(audioFile_Metadata.encoding)
            waveform, sample_rate = torchaudio.load(audioFile_path)
            # print(f'Loaded {audioFile_path}')
            # play_audio(waveform, sample_rate)
            # plot_waveform(waveform, sample_rate)
            # plot_specgram(waveform, sample_rate)
            if self.rangeOfColumnNumbers_ToConsiderInCsvFile is not None:
                labels = self.labels.iloc[idx, self.rangeOfColumnNumbers_ToConsiderInCsvFile[0]:self.rangeOfColumnNumbers_ToConsiderInCsvFile[1]].to_numpy()
                labels = torch.tensor(list(labels), dtype=torch.float64)
                print(labels)
            else:
                labels = torch.empty(0)
            if self.transform:
                spectrogram = self.transform(waveform)
            if self.target_transform and labels is not None:
                labels = self.target_transform(labels)
            return spectrogram, labels
        else:
            return torch.empty(0), torch.empty(0)

synthDataset = Sounds_Dataset(synthDataset_AudioFiles_ParentFolderPath, synthDataset_GroundTruth_CsvFIlePath, syntheticToReal_Unsupervised_DA['syntheticDataset_Settings']['rangeOfColumnNumbers_ToConsiderInCsvFile'])
# synthDataset.__getitem__(0)

# DATALOADER (torch.utils.data.DataLoader)
train_dataloader = DataLoader(synthDataset, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
import pandas 
import os
import json

from Configuration_Dictionary import configDict

######################################################################################################################################################
# DATASET CLASS (extends torch.utils.data.Dataset) 
class Dataset_Wrapper(Dataset):
    def __init__(self, audioFiles_Directory_, groundTruth_CsvFIlePath_, rangeOfColumnNumbers_ToConsiderInCsvFile_, device_, transform = None, target_transform = None):
        self.device = device_
        self.labels = pandas.read_csv(groundTruth_CsvFIlePath_)
        self.rangeOfColumnNumbers_ToConsiderInCsvFile = rangeOfColumnNumbers_ToConsiderInCsvFile_
        self.numberOfLabels = (self.rangeOfColumnNumbers_ToConsiderInCsvFile[1] - self.rangeOfColumnNumbers_ToConsiderInCsvFile[0]) + 1
        self.audioFiles_Directory = audioFiles_Directory_
        if transform:
            self.transforms = transform
            for transform in self.transforms:
                transform = transform.to(self.device)
        else:
            self.transforms = None
        if target_transform:
            self.target_transform = target_transform.to(self.device)
        else:
            self.target_transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audioFile_path = os.path.join(self.audioFiles_Directory, self.labels.iloc[idx, 0])
        if os.path.exists(audioFile_path):
            assert audioFile_path.endswith(configDict['validation']['nominal_AudioFileExtension']), f"Error while loading {audioFile_path} : Audio file extension is not valid"
            if configDict['validation']['validate_AudioDatasets']:
                audioFile_Metadata = torchaudio.info(audioFile_path)
                assert audioFile_Metadata.sample_rate == configDict['validation']['nominal_SampleRate'], f"Error while loading {audioFile_path} : Sample rate is not valid"
                assert audioFile_Metadata.num_frames == configDict['validation']['nominal_AudioDurationSecs'] * configDict['validation']['nominal_SampleRate'], f"Error while loading {audioFile_path} : Audio duration is not valid"
                assert audioFile_Metadata.num_channels == configDict['validation']['nominal_NumOfAudioChannels'], f"Error while loading {audioFile_path} : Number of audio channels is not valid"
                assert audioFile_Metadata.bits_per_sample == configDict['validation']['nominal_BitQuantization'], f"Error while loading {audioFile_path} : Bit quantization is not valid"
            audioSignal, sample_rate = torchaudio.load(audioFile_path)
            if self.rangeOfColumnNumbers_ToConsiderInCsvFile:
                labels = self.labels.iloc[idx, self.rangeOfColumnNumbers_ToConsiderInCsvFile[0]:self.rangeOfColumnNumbers_ToConsiderInCsvFile[1]].to_numpy()
                labels = torch.tensor(list(labels), dtype=torch.float64)
            else:
                labels = torch.empty(self.numberOfLabels)
            if self.transforms:
                for transform in self.transforms:
                    audioSignal = transform(audioSignal)
            if self.target_transform and labels:
                labels = self.target_transform(labels)
            # print(f'Audio signal shape : {audioSignal.shape}')
            # print(f'Labels shape : {labels.shape}')
            return audioSignal, labels
        else:
            return torch.empty(0), torch.empty(0)
        
    def getAnnotations_ColumnsNames(self):
        # return the column names of the .csv file containing the ground truth
        return list(self.labels.columns[self.rangeOfColumnNumbers_ToConsiderInCsvFile[0]:self.rangeOfColumnNumbers_ToConsiderInCsvFile[1]])
######################################################################################################################################################

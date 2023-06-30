import torch
import torchaudio
from torch.utils.data.dataset import Dataset
import pandas 
import os
import matplotlib.pyplot as plt
import librosa
import random
import json

######################################################################################################################################################
# DATASET CLASS (extends torch.utils.data.Dataset) 

class Dataset_Wrapper(Dataset):
    def __init__(self,
                audioFiles_Directory_,
                groundTruth_CsvFIlePath_,
                configDict,
                transform = None,
                target_transform = None,
                applyNoise = False,
                supervised_Task = True):
        '''
        For supervised tasks, rangeOfColumnNumbers_ToConsiderInCsvFile_ must not be None. 
        See __getItem()__ documentation for more details.
        '''
        # print(f'Initializing Dataset_Wrapper object.')
        # print(f'    Audio files directory : {audioFiles_Directory_}')
        # print(f'    Ground truth .csv file path : {groundTruth_CsvFIlePath_}')
        # print(f'    Range of column numbers to consider in the .csv file : {rangeOfColumnNumbers_ToConsiderInCsvFile_}')   
        self.configDict = configDict
        random.seed(self.configDict['pyTorch_General_Settings']['manual_seed'])

        self.applyNoise = applyNoise
        self.device = self.configDict['pyTorch_General_Settings']['device']
        self.labels = pandas.read_csv(groundTruth_CsvFIlePath_)
        if self.configDict['syntheticDataset_Settings']['rangeOfColumnNumbers_ToConsiderInCsvFile'] is not None:
            self.rangeOfColumnNumbers_ToConsiderInCsvFile = self.configDict['syntheticDataset_Settings']['rangeOfColumnNumbers_ToConsiderInCsvFile']
            self.rangeOfColumnNumbers_ToConsiderInCsvFile[1] += 1 # to include the last column
            self.numberOfLabels = (self.rangeOfColumnNumbers_ToConsiderInCsvFile[1] - self.rangeOfColumnNumbers_ToConsiderInCsvFile[0])
            # print(f'    self.numberOfLabels : {self.numberOfLabels}')
        else:
            self.rangeOfColumnNumbers_ToConsiderInCsvFile = None
            self.numberOfLabels = None
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
        if not supervised_Task:
            self.rangeOfColumnNumbers_ToConsiderInCsvFile = None
            self.numberOfLabels = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        If self.configDict['syntheticDataset_Settings']['rangeOfColumnNumbers_ToConsiderInCsvFile'] is not None in the constructor, returns a tuple (audioFile, target) where target is a list of labels
        Else, returns a tuple (audioFile, target) where target is the name of the audioFile
        '''
        verbose = False

        audioFile_path = os.path.join(self.audioFiles_Directory, self.labels.iloc[idx, 0])
        if os.path.exists(audioFile_path):
            assert audioFile_path.endswith(self.configDict['validation']['nominal_AudioFileExtension']), f"Error while loading {audioFile_path} : Audio file extension is not valid"
            if self.configDict['validation']['validate_AudioDatasets']:
                audioFile_Metadata = torchaudio.info(audioFile_path)
                assert audioFile_Metadata.sample_rate == self.configDict['validation']['nominal_SampleRate'], f"Error while loading {audioFile_path} : Sample rate is not valid"
                assert audioFile_Metadata.num_frames == self.configDict['validation']['nominal_AudioDurationSecs'] * self.configDict['validation']['nominal_SampleRate'], f"Error while loading {audioFile_path} : Audio duration is not valid"
                assert audioFile_Metadata.num_channels == self.configDict['validation']['nominal_NumOfAudioChannels'], f"Error while loading {audioFile_path} : Number of audio channels is not valid"
                assert audioFile_Metadata.bits_per_sample == self.configDict['validation']['nominal_BitQuantization'], f"Error while loading {audioFile_path} : Bit quantization is not valid"
            audioSignal, sample_rate = torchaudio.load(audioFile_path)
            audioSignal_Abs = torch.abs(audioSignal)
            audioSignal_Max = torch.max(audioSignal_Abs)
            if audioSignal_Max == 0.:
                print(f'ERROR : {self.labels.iloc[idx, 0]} waveform is all 0-valued')
                exit()
                audioSignal_Norm = torch.zeros(audioSignal.shape)
            else:
                audioSignal_Norm = torch.div(audioSignal, audioSignal_Max) # normalize audio waveform between -1. and 1.
            if verbose:
                plot_waveform(audioSignal_Norm, sample_rate = self.configDict['validation']['nominal_SampleRate'], title = self.labels.iloc[idx, 0])
            if self.applyNoise:
                # print('applying noise')
                audioSignal_Norm = add_noise(audioSignal_Norm,
                                            self.labels.iloc[idx, 0],
                                            sample_rate,
                                            self.configDict['inputTransforms_Settings']['addNoise']['minimum_LowPassFilter_FreqThreshold'],
                                            self.configDict['inputTransforms_Settings']['addNoise']['maximum_LowPassFilter_FreqThreshold'],
                                            self.configDict['inputTransforms_Settings']['addNoise']['minimumNoiseAmount'],
                                            self.configDict['inputTransforms_Settings']['addNoise']['maximumNoiseAmount'],
                                            verbose)
            if verbose:
                plot_waveform(audioSignal_Norm, sample_rate = self.configDict['validation']['nominal_SampleRate'], title = self.labels.iloc[idx, 0])
            audioSignal_Norm = audioSignal_Norm.to(self.device)
            if self.rangeOfColumnNumbers_ToConsiderInCsvFile:
                labels = self.labels.iloc[idx, self.rangeOfColumnNumbers_ToConsiderInCsvFile[0]:self.rangeOfColumnNumbers_ToConsiderInCsvFile[1]].to_numpy()
                labels = torch.tensor(list(labels), dtype = self.configDict['pyTorch_General_Settings']['dtype'])
            else:
                if self.numberOfLabels:
                    labels = torch.empty(self.numberOfLabels)
                else:
                    # labels = torch.empty(0)
                    labels = self.labels.iloc[idx, 0]
            if self.transforms:
                for trans_Num, transform in enumerate(self.transforms):
                    audioSignal_Norm = transform(audioSignal_Norm)
                    if verbose:
                        if trans_Num == 0:
                            plot_waveform(audioSignal_Norm, sample_rate = self.configDict['inputTransforms_Settings']['resample']['new_freq'], title = self.labels.iloc[idx, 0])
                        elif trans_Num == 1:
                            plot_spectrogram(audioSignal_Norm[0], title = self.labels.iloc[idx, 0])
            if self.target_transform and labels:
                labels = self.target_transform(labels)

            # print(f'audioSignal_Norm min : {audioSignal_Norm.min()}')
            # print(f'audioSignal_Norm max : {audioSignal_Norm.max()}')
            # print(f'audioSignal_Norm shape : {audioSignal_Norm.shape}')
            # print(f'audioSignal_Norm : {audioSignal_Norm}')

            aToDbTrans = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
            audioSignal_Norm = aToDbTrans(audioSignal_Norm)

            # print(f'audio DB min : {audioSignal_Norm.min()}')
            # print(f'audio DB max : {audioSignal_Norm.max()}')
            # print(f'audio DB shape : {audioSignal_Norm.shape}')
            # print(f'audio DB : {audioSignal_Norm}')

            threshold = torch.nn.Threshold(-80., -80.)
            audioSignal_Norm = threshold(audioSignal_Norm)

            audioSignal_Norm = audioSignal_Norm + 80.
            audioSignal_Norm = audioSignal_Norm / audioSignal_Norm.max()

            # print(f'audio DB norm min : {audioSignal_Norm.min()}')
            # print(f'audio DB norm max : {audioSignal_Norm.max()}')
            # print(f'audio DB norm shape : {audioSignal_Norm.shape}')
            # print(f'audio DB norm: {audioSignal_Norm}')

            return audioSignal_Norm, labels
        else:
            return torch.empty(0), torch.empty(0)
        
    def getAnnotations_ColumnsNames(self):
        # return the column names of the .csv file containing the ground truth
        if self.rangeOfColumnNumbers_ToConsiderInCsvFile is not None:
            return list(self.labels.columns[self.rangeOfColumnNumbers_ToConsiderInCsvFile[0]:self.rangeOfColumnNumbers_ToConsiderInCsvFile[1]])
        else:
            return list()
######################################################################################################################################################

######################################################################################################################################################
# MIXED DATASET CLASS (extends torch.utils.data.Dataset) 

class Mixed_Dataset_Wrapper(Dataset):
    def __init__(self,
                configDict,
                transform = None):
        
        self.configDict = configDict
        random.seed(self.configDict['pyTorch_General_Settings']['manual_seed'])

        self.device = self.configDict['pyTorch_General_Settings']['device']

        with open(self.configDict['paths']['synthDataset_JSonFile_Path']) as synthDataset_JSonFile:
            synthDatasetGenerator_DescriptorDict = json.load(synthDataset_JSonFile)
        self.synthDataset_AudioFiles_Directory = os.path.abspath(synthDatasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'])
        print(f'Synthetic dataset directory : {self.synthDataset_AudioFiles_Directory}')
        synthDataset_CsvFilePath = os.path.join(self.synthDataset_AudioFiles_Directory, str(synthDatasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + '.csv'))
        self.syntheticAudioFilesLabels = pandas.read_csv(synthDataset_CsvFilePath)

        with open(self.configDict['paths']['realDataset_JSonFile_Path']) as realDataset_JSonFile:
            realDatasetGenerator_DescriptorDict = json.load(realDataset_JSonFile)
        realDataset_AudioFiles_Directory_ParentFold = os.path.abspath(realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_ParentFolder'])
        self.realDataset_AudioFiles_Directory = os.path.join(realDataset_AudioFiles_Directory_ParentFold, realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_FolderName'])
        print(f'Real dataset directory : {self.realDataset_AudioFiles_Directory}')
        realDataset_CsvFilePath = os.path.join(self.realDataset_AudioFiles_Directory, str(realDatasetGenerator_DescriptorDict['outputDataset_Settings']['outputDataset_FolderName'] + '.csv'))
        self.realAudioFilesLabels = pandas.read_csv(realDataset_CsvFilePath)

        if transform:
            self.transforms = transform
            for transform in self.transforms:
                transform = transform.to(self.device)
        else:
            self.transforms = None

    def __len__(self):
        # MAKE SURE len(self.syntheticAudioFilesLabels) >= len(self.realAudioFilesLabels)
        return len(self.realAudioFilesLabels) * 2

    def __getitem__(self, idx):
        verbose = False

        if idx < len(self.realAudioFilesLabels):
            # chosenToReturnRealAudioFile = random.choice([True, False])
            chosenToReturnRealAudioFile = True
        else:
            chosenToReturnRealAudioFile = False
        if chosenToReturnRealAudioFile:
            # print(f'{self.realAudioFilesLabels.iloc[idx, 0]}')
            audioFile_path = os.path.join(self.realDataset_AudioFiles_Directory, self.realAudioFilesLabels.iloc[idx, 0])
            audioFIleName = self.realAudioFilesLabels.iloc[idx, 0]
            label = torch.ones(1, dtype=torch.float32)
        else:
            # print(f'{self.syntheticAudioFilesLabels.iloc[idx, 0]}')
            audioFile_path = os.path.join(self.synthDataset_AudioFiles_Directory, self.syntheticAudioFilesLabels.iloc[idx, 0])
            audioFIleName = self.syntheticAudioFilesLabels.iloc[idx, 0]
            label = torch.zeros(1, dtype=torch.float32)
        if os.path.exists(audioFile_path):
            audioSignal, sample_rate = torchaudio.load(audioFile_path)
            audioSignal_Abs = torch.abs(audioSignal)
            audioSignal_Max = torch.max(audioSignal_Abs)
            if audioSignal_Max == 0.:
                print(f'ERROR : {audioFIleName} waveform is all 0-valued')
                exit()
                audioSignal_Norm = torch.zeros(audioSignal.shape)
            else:
                audioSignal_Norm = torch.div(audioSignal, audioSignal_Max) # normalize audio waveform between -1. and 1.
            if verbose:
                plot_waveform(audioSignal_Norm, sample_rate = self.configDict['validation']['nominal_SampleRate'], title = audioFIleName)
            if not chosenToReturnRealAudioFile:
                audioSignal_Norm = add_noise(audioSignal_Norm,
                                            audioFIleName,
                                            sample_rate,
                                            self.configDict['inputTransforms_Settings']['addNoise']['minimum_LowPassFilter_FreqThreshold'],
                                            self.configDict['inputTransforms_Settings']['addNoise']['maximum_LowPassFilter_FreqThreshold'],
                                            self.configDict['inputTransforms_Settings']['addNoise']['minimumNoiseAmount'],
                                            self.configDict['inputTransforms_Settings']['addNoise']['maximumNoiseAmount'],
                                            verbose)
            if verbose:
                plot_waveform(audioSignal_Norm, sample_rate = self.configDict['validation']['nominal_SampleRate'], title = audioFIleName)
            audioSignal_Norm = audioSignal_Norm.to(self.device)
            if self.transforms:
                for trans_Num, transform in enumerate(self.transforms):
                    audioSignal_Norm = transform(audioSignal_Norm)
                    if verbose:
                        if trans_Num == 0:
                            plot_waveform(audioSignal_Norm, sample_rate = self.configDict['inputTransforms_Settings']['resample']['new_freq'], title = audioFIleName)
                        elif trans_Num == 1:
                            plot_spectrogram(audioSignal_Norm[0], title = audioFIleName)
            return audioSignal_Norm, label # label is true if the audio file is real, false if it is synthetic
        else:
            return torch.empty(0), torch.empty(0)
######################################################################################################################################################

# UTILS
######################################################################################################################################################
def plot_waveform(waveform, sample_rate, title = "waveform"):
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
    figure.suptitle(str(title + ' sampled at ' +  str(sample_rate) + ' Hz'))
    plt.show(block = True)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block = True)


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block = True)

def add_noise(audio_waveform,
                audio_file_name,
                in_sample_rate,
                minimum_LowPassFilter_FreqThreshold,
                maximum_LowPassFilter_FreqThreshold,
                minimumNoiseAmount,
                maximumNoiseAmount,
                verbose):

    # TEST save nosy Audio file for evaluation
    # torchaudio.save(os.path.join('/Users/matthew/Downloads/Test_Noisy_Audio', str('original_' + audio_file_name)), audio_waveform, in_sample_rate)

    specrogramTransform = torchaudio.transforms.Spectrogram(n_fft=1024, power=2)

    noise = torch.randn_like(audio_waveform)
    noise_abs = torch.abs(noise)
    noise_max = torch.max(noise_abs)
    noise_norm = torch.div(noise, noise_max) # normalize audio waveform between -1. and 1.

    # Define effects
    threshold = torch.randint(minimum_LowPassFilter_FreqThreshold, maximum_LowPassFilter_FreqThreshold, (1,))
    # print(str(int(threshold)))
    effects = [
        ["lowpass", str(int(threshold))],  # apply single-pole lowpass filter
    ]
    filteredNoise, filteredNoiseSampRate = torchaudio.sox_effects.apply_effects_tensor(noise_norm, in_sample_rate, effects)
    if filteredNoiseSampRate != in_sample_rate:
        print('WARNING : add_noise -> filteredNoiseSampRate != in_sample_rate')

    if verbose:
        plot_spectrogram(specrogramTransform(audio_waveform[0]), title = 'Original audio')
    noiseAmount = random.uniform(minimumNoiseAmount, maximumNoiseAmount)
    noisy_waveform = audio_waveform + (filteredNoise * noiseAmount)
    if verbose:
        plot_spectrogram(specrogramTransform(noisy_waveform[0]), title = 'Audio with noise')

    # re-normalize
    noisy_waveform_abs = torch.abs(noisy_waveform)
    noisy_waveform_max = torch.max(noisy_waveform_abs)
    noisy_waveform_norm = torch.div(noisy_waveform, noisy_waveform_max) # normalize audio waveform between -1. and 1.

    # TEST save nosy Audio file for evaluation
    # torchaudio.save(os.path.join('/Users/matthew/Downloads/Test_Noisy_Audio', audio_file_name), audio_waveform, in_sample_rate)
    # torchaudio.save(os.path.join('/Users/matthew/Downloads/Test_Noisy_Audio', str('noisy_' + audio_file_name)), noisy_waveform_norm, filteredNoiseSampRate)

    return noisy_waveform_norm
######################################################################################################################################################
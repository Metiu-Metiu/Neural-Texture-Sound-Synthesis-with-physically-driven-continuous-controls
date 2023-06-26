import torch
import torchaudio

########### input variables #############################################
configDict = {
    'paths': {
        # Path of the .json file containing the descriptor dictionary of the synthetic dataset
        # /content/drive/MyDrive/Master Thesis Project/Datasets/SDT_FluidFlow_dataset/SDT_FluidFlow.json
        # '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/SDT_FluidFlow_dataset/SDT_FluidFlow.json
        'synthDataset_JSonFile_Path': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/Datasets/SDT_FluidFlow_dataset_10000_1sec/SDT_FluidFlow.json',
        # /content/drive/MyDrive/Master Thesis Project/Datasets/FSD50K_Water_Stream_subset/FSD50K_Water_Stream_subset_creatorDescriptorDict.json
        # /Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/FSD50K_Water_Stream_subset/FSD50K_Water_Stream_subset_creatorDescriptorDict.json
        'realDataset_JSonFile_Path': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/Datasets/FSD50K_Water_Stream_subset_1sec/FSD50K_Water_Stream_subset_1sec_creatorDescriptorDict.json',
    },

    'syntheticDataset_Settings': {
        # The first and last column number to consider in the .csv file containing the ground truth of interest
        # Audio file name is always column n. 0
        'rangeOfColumnNumbers_ToConsiderInCsvFile': [1, 4],
        'splits' : { # they need to add up to 1.
            'train' : 0.85,
            'val' : 0.05,
            'test' : 0.1
        }
    },

    'realDataset_Settings': {
        # The first and last column number to consider in the .csv file containing the ground truth of interest
        # Audio file name is always column n. 0
        'rangeOfColumnNumbers_ToConsiderInCsvFile': None, # None for an 'unsupervised' task, with no ground truth
    },

    'validation': {
        'validate_AudioDatasets': True, # either true or false, checks if the variables below are valid
        'nominal_SampleRate': 44100, # int
        'nominal_NumOfAudioChannels': 1, # int
        'nominal_AudioFileExtension': '.wav', # string
        'nominal_BitQuantization': 16, # int
        'nominal_AudioDurationSecs': 1.0, # float
    },

    'pyTorch_General_Settings': {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'dtype': torch.float32,
        'manual_seed': 42,
    },

    'inputTransforms_Settings': {
        'resample' : {
            'new_freq' : 32000
        },

        'addNoise' : {
            'perform' : True, # either True or False
            'minimum_LowPassFilter_FreqThreshold' : 6000,
            'maximum_LowPassFilter_FreqThreshold' : 14000,
            'minimumNoiseAmount' : 0.005,
            'maximumNoiseAmount' : 0.02
        },

        'spectrogram' : {
            'n_fft' : 1024,
            'n_mels' : 56,
        },
    },

    'neuralNetwork_Settings': {
        'number_Of_Epochs': 1,
        'batch_size': 128, # try to decide a batch_size so that the total number of samples in the dataset is divisible by the batch size
        'learning_Rate': 0.0005, # 0.0005
        'dropout_Probability': 0.4,
        'arguments_For_Convolutional_DynamicNet_Constructor': {
            'numberOfFeaturesToExtract_IncremMultiplier_FromLayer1': 2,
            'numberOfConvLayers': 4,
            'kernelSizeOfConvLayers': 3,
            'strideOfConvLayers': 1,
            'kernelSizeOfPoolingLayers': 2,
            'strideOfPoolingLayers': 2,
            'numberOfFullyConnectedLayers': 3,
            'fullyConnectedLayers_InputSizeDecreaseFactor': 4
        },
        'early_Stopping': True,
        'minimum_NumberOfEpochsToTrain_RegardlessOfEarlyStoppingBeingActive': 1,
        'loss' : {
            # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
            'reduction' : 'mean'
        },
        'activation_Function' : {
            'negative_slope' : 0.9
        }
    },

    'outputFilesSettings': {
        # /content/drive/MyDrive/Master Thesis Project/Trained_Neural_Networks/2D_CNN_SynthParamExtractor_June1_2023
        # /Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/Neural Networks/2D_CNN_SynthParamExtractor_June9_2023
        'outputFolder_Path': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/Neural Networks/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm',
        'jSonFile_WithThisDict_Name': '2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm',
        'pyTorch_NN_StateDict_File_Name': '2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm'
    },

    'statistics': {
        'elapsedTime_WhileTraining' : None,
        'dateAndTime_WhenTrainingFinished_dd/mm/YY H:M:S' : None,
    }
}

# MUST BE A TORCHAUDIO TRANSFORM, see https://pytorch.org/audio/stable/transforms.html for available transforms
configDict['neuralNetwork_Settings']['input_Transforms'] = [
    torchaudio.transforms.Resample(
        orig_freq = configDict['validation']['nominal_SampleRate'],
        new_freq = configDict['inputTransforms_Settings']['resample']['new_freq']),
    torchaudio.transforms.MelSpectrogram(
        # n_fft = configDict['inputTransforms_Settings']['spectrogram']['n_fft'],
        normalized = True,
        n_mels = configDict['inputTransforms_Settings']['spectrogram']['n_mels'], # with 128 -default-: UserWarning: At least one mel filterbank has all zero values. The value for `n_mels` (128) may be set to
        sample_rate = configDict['inputTransforms_Settings']['resample']['new_freq'])
        ]
#########################################################################
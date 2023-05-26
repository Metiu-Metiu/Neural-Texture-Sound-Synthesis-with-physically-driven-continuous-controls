import torch
import torchaudio

########### input variables #############################################
configDict = {
    'paths': {
        # Path of the .json file containing the descriptor dictionary of the synthetic dataset
        'synthDataset_JSonFile_Path': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/SDT_FluidFlow_dataset/SDT_FluidFlow.json',
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
        'nominal_AudioDurationSecs': 3.0, # float
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

        'spectrogram' : {
            'n_fft' : 1024,
        },
    },

    'neuralNetwork_Settings': {
        'number_Of_Epochs': 30,
        'batch_size': 128 # try to decide a batch_size so that the total number of samples in the dataset is divisible by the batch size
    },

    'outputFilesSettings': {
        'outputFolder_Path': '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/repo/SMC_thesis/Synthetic_to_real_unsupervised_Domain_Adaptation/Trained_Neural_Networks/1D_CNN_SynthParamExtractor_May26_2023/',
        'jSonFile_WithThisDict_Name': '1D_CNN_SynthParamExtractor_May26_2023',
        'pyTorch_NN_StateDict_File_Name': '1D_CNN_SynthParamExtractor_May26_2023'
    }
}

# MUST BE A TORCHAUDIO TRANSFORM, see https://pytorch.org/audio/stable/transforms.html for available transforms
configDict['neuralNetwork_Settings']['input_Transforms'] = [
    torchaudio.transforms.Resample(
        orig_freq = configDict['validation']['nominal_SampleRate'],
        new_freq = configDict['inputTransforms_Settings']['resample']['new_freq'])
        ]
#########################################################################
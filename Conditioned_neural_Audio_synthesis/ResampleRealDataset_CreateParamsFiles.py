# INPUT VARIABLES
##############################################################################################################
originalRealDatasetPath = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/Segmented_20Minutes_shower_withDifferentFlowRates with params files'
resampledRealDatasetPath = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/Segmented_20Minutes_shower_withDifferentFlowRates with params files'

realDatasetLabels_CsvFilePath = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/Segmented_20Minutes_shower_withDifferentFlowRates with params files/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm_DBScale_ExtractedAudioFilesLabels__DA_runningwater.16K.csv'
numOfSynthContrParamValues_ForEachAudioFile = 11
target_sample_rate = 16000

perform_Resampling = False # if False, set resampledRealDatasetPath equal to originalRealDatasetPath
    # only params files will be created if False
##############################################################################################################

import csv
import json
import os

import soundfile
import librosa

import time

numberOfSoundFiles_InDataset_Resampled = 0
numberOfParamsFiles_Created = 0

# RESAMPLE .WAV FILES
if perform_Resampling:
    if not os.path.exists(resampledRealDatasetPath):
        os.makedirs(resampledRealDatasetPath)

    for filename in os.listdir(originalRealDatasetPath):
        if filename.endswith(".wav"):
            input_path = os.path.join(originalRealDatasetPath, filename)
            output_path = os.path.join(resampledRealDatasetPath, filename)
            try:
                x, sr = librosa.load(input_path, sr=None)
                # print(f'Original sample rate: {sr} Hz')
                # time.sleep(1)
                x_copy = x.copy()
                resampled_audio = librosa.resample(x_copy, orig_sr = sr, target_sr = target_sample_rate)
                soundfile.write(output_path, resampled_audio, target_sample_rate)
                y, sr = soundfile.read(output_path)
                # print(f'Converted sample rate: {sr} Hz')
                # print(f'Converted Audio data shape: {y.shape}')
                print(f"Resampled '{input_path}' and saved to '{output_path}' at {target_sample_rate} Hz.")
                # time.sleep(1)
                numberOfSoundFiles_InDataset_Resampled += 1
            except Exception as e:
                print(f"Error processing '{input_path}': {e}")
                exit()

# CREATE .PARAMS FILES
if not os.path.exists(resampledRealDatasetPath):
    os.makedirs(resampledRealDatasetPath)

realDatasetLabels_ColumnNames = []
with open(realDatasetLabels_CsvFilePath, 'r') as realDatasetLabelsCSVFile:
            csvReader = csv.DictReader(realDatasetLabelsCSVFile)
            for csvReaderRow in csvReader:
                paramsFileDict = dict()
                paramsFileDict['meta'] = dict()
                paramsFileDict['meta']['filename'] = csvReaderRow['AudioFileName']
                for csvReaderRowKey in csvReaderRow.keys():
                    if csvReaderRowKey != 'AudioFileName':
                        paramsFileDict[csvReaderRowKey] = dict()
                        paramsFileDict[csvReaderRowKey]['times'] = [0, 1]
                        paramsFileDict[csvReaderRowKey]['values'] = [csvReaderRow[csvReaderRowKey], csvReaderRow[csvReaderRowKey]]
                        paramsFileDict[csvReaderRowKey]['units'] = 'norm'
                        paramsFileDict[csvReaderRowKey]['nvals'] = numOfSynthContrParamValues_ForEachAudioFile
                        paramsFileDict[csvReaderRowKey]['minval'] = 0
                        paramsFileDict[csvReaderRowKey]['maxval'] = 1
                paramsFilePath = os.path.join(os.path.abspath(resampledRealDatasetPath), str(os.path.splitext(csvReaderRow['AudioFileName'])[0] + '.params'))
                with open(paramsFilePath, 'w') as paramsfile:
                    json.dump(paramsFileDict, paramsfile, indent=4)
                    numberOfParamsFiles_Created += 1

if perform_Resampling:
    print(f'Finished creating .params files for real dataset.')
    print(f'    Number of sound files in dataset: {numberOfSoundFiles_InDataset_Resampled}')
    print(f'    Number of .params files created: {numberOfParamsFiles_Created}')
    if numberOfSoundFiles_InDataset_Resampled == numberOfParamsFiles_Created:
        print(f'    Number of sound files in dataset = Number of .params files created.')
        print('NO ERRORS ENCOUNTERED')
    else:
        print(f'    Number of sound files in dataset != Number of .params files created.')
        print('SOME ERRORS OCCURRED')
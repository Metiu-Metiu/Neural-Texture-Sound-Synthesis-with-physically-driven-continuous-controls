# INPUT VARIABLES
##############################################################################################################
# path of the .csv file containing the labels extracted from the Real Dataset
synthContrParam_Labels_CSVFilePath = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/Segmented_20Minutes_shower_withDifferentFlowRates/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm_DBScale_ExtractedAudioFilesLabels__DA_runningwater.16K.csv'
# synthContrParam_Labels_CSVFilePath = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/SDT_FluidFlow_dataset_10000_1sec/SDT_FluidFlow.csv'

deleteNegativeValues = False
normaliseData = True

variable_ToCreateApproxUniformDistributionFrom = 'expRadius'
numberOfHistogramBins = 10
plot = True

inputAudioFilesParentFolder = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/Segmented_20Minutes_shower_withDifferentFlowRates'
outputAudioFilesParentFolder = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/Segmented_20Minutes_shower_withDifferentFlowRates_UniformDistrFor_expRadius'
##############################################################################################################

import pandas
import matplotlib.pyplot as plt
import os
import shutil

df = pandas.read_csv(synthContrParam_Labels_CSVFilePath)
df = df.drop(df.columns[0], axis=1)

if deleteNegativeValues:
    negative_counts = (df < 0).sum()
    print("\nNumber of negative values in each column:")
    print(negative_counts)
    df[df < 0] = float('nan')
    print('\nDeleted negative values from dataframe... .. .')

if normaliseData:
    df=(df-df.min())/(df.max()-df.min())
    print('\nNormalised data... .. .')

min_values = df.min()
max_values = df.max()
mean_values = df.mean()
std_values = df.std()

print("\nMinimum values:")
print(min_values)
print("\nMaximum values:")
print(max_values)
print("\nMean values:")
print(mean_values)
print("\nStandard deviation values:")
print(std_values)

# Create a scatter plot using Matplotlib
# plt.scatter(df['avgRate'], [df['expRadius']])
# plt.xlabel('avgRate')
# plt.ylabel('expRadius')
# plt.title('Scatter Plot of avgRate vs Y')
# plt.show()

# # Create a box plot using Matplotlib
# plt.boxplot(df['avgRate'], notch = True)
# plt.xlabel('avgRate')
# plt.title('Box Plot of avgRate')
# plt.show()

if plot:
    hist = df.hist(bins = numberOfHistogramBins, column = variable_ToCreateApproxUniformDistributionFrom)
    plt.show()

    plt.hist(df[variable_ToCreateApproxUniformDistributionFrom], bins = numberOfHistogramBins)
    plt.xlabel(variable_ToCreateApproxUniformDistributionFrom)
    plt.title(f'Histogram of {variable_ToCreateApproxUniformDistributionFrom}')
    plt.show()

totalNumOfSamples = df[variable_ToCreateApproxUniformDistributionFrom].count()
print(f'\nTotal number of samples for variable {variable_ToCreateApproxUniformDistributionFrom}: {totalNumOfSamples}')

medianNumberOfSamples = int(totalNumOfSamples / numberOfHistogramBins)
print(f'\nMedian number of samples for each histogram bin: {medianNumberOfSamples}')

df = pandas.read_csv(synthContrParam_Labels_CSVFilePath)
audioFileNames_UniformDistrSubset = list()
histogram_bins = pandas.cut(df[variable_ToCreateApproxUniformDistributionFrom], bins = numberOfHistogramBins)
print(histogram_bins)

# Count occurrences in each bin
bin_counts = histogram_bins.value_counts().sort_index()

filtered_data = list()
for bin_range, count in zip(bin_counts.index, bin_counts.values):
    lower, upper = bin_range.left, bin_range.right
    fileNamesInThisBin = df[(df[variable_ToCreateApproxUniformDistributionFrom] >= lower) & (df[variable_ToCreateApproxUniformDistributionFrom] < upper)]['AudioFileName'].values.tolist()
    print(f"Bin Range: {bin_range}, Count: {count}")
    # print(filtered_data)
    if len(fileNamesInThisBin) > medianNumberOfSamples:
        fileNamesInThisBin = fileNamesInThisBin[:medianNumberOfSamples]
        print(f'Selected {medianNumberOfSamples} samples from this bin.')
    else:
        print(f'Not enough samples in this bin. Selected {len(fileNamesInThisBin)} samples from this bin.')
    print("-" * 40)
    filtered_data.append(fileNamesInThisBin)

numberOfSamplesInUniformDistr = 0
for binList in filtered_data:
    numberOfSamplesInUniformDistr += len(binList)
print(f'\nTotal number of samples in uniform distribution: {numberOfSamplesInUniformDistr}')

# COPY FILES TO NEW FOLDER
if not os.path.exists(outputAudioFilesParentFolder):
    os.makedirs(outputAudioFilesParentFolder)

inputFiles = os.listdir(inputAudioFilesParentFolder)

audioFileNames_UniformDistrSubset_ = list()
for list in filtered_data:
    for audio_file_name in list:
        audioFileNames_UniformDistrSubset_.append(audio_file_name)

for inputFile in inputFiles:
    if inputFile in audioFileNames_UniformDistrSubset_:
        source_file_path = os.path.join(inputAudioFilesParentFolder, inputFile)
        destination_file_path = os.path.join(outputAudioFilesParentFolder, inputFile)
        shutil.copy(source_file_path, destination_file_path)

print("Audio files copied successfully!")
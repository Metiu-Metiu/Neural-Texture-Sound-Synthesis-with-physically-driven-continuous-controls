# INPUT VARIABLES
##############################################################################################################
# path of the .csv file containing the labels extracted from the Real Dataset
synthContrParam_Labels_CSVFilePath = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/Segmented_20Minutes_shower_withDifferentFlowRates/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm_DBScale_ExtractedAudioFilesLabels__DA_runningwater.16K.csv'
# synthContrParam_Labels_CSVFilePath = '/Users/matthew/Downloads/Synthetic-and-real-sounds-datasets-for-SMC_Thesis/SDT_FluidFlow_dataset_10000_1sec/SDT_FluidFlow.csv'
##############################################################################################################

import pandas

df = pandas.read_csv(synthContrParam_Labels_CSVFilePath)
df = df.drop(df.columns[0], axis=1)

negative_counts = (df < 0).sum()
min_values = df.min()
max_values = df.max()
mean_values = df.mean()
std_values = df.std()

print("\nNumber of negative values in each column:")
print(negative_counts)
print("\nMinimum values:")
print(min_values)
print("\nMaximum values:")
print(max_values)
print("\nMean values:")
print(mean_values)
print("\nStandard deviation values:")
print(std_values)

##############################################################################################################
# delete negative values
df[df < 0] = float('nan')
print('\nDeleted negative values from dataframe... .. .')

negative_counts = (df < 0).sum()
min_values = df.min()
max_values = df.max()
mean_values = df.mean()
std_values = df.std()

print("\nNumber of negative values in each column:")
print(negative_counts)
print("\nMinimum values:")
print(min_values)
print("\nMaximum values:")
print(max_values)
print("\nMean values:")
print(mean_values)
print("\nStandard deviation values:")
print(std_values)
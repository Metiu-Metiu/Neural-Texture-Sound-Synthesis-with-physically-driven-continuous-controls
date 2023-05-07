from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
import argparse
import random
import time
import csv
import os
# import threading
import json
import numpy
from enum import Enum
from datetime import datetime
import math
from itertools import product # to compute all combinations of synth contr param values for LINEAR_UNIFORM_ALL_COMBINATIONS distribution
 
class Distribution_Of_Synthesis_Control_Parameters_Values(Enum):
    # RANDOM_UNIFORM: for each audio file, for each synth contr param, 2 stochastic processes are involved;
        # a binary choice is randomly taken -with random.choice()-, to decide whether to generate a new synth contr param value or to re-use the same one used in the previous file
        # if a new synth contr param value has to be generated, a value is generated randomly -with random.uniform()- within the given numerical ranges
    # RANDOM_UNIFORM is uniform as the number of audio files to be generated approaches infinity
    RANDOM_UNIFORM = 1
    # LINEAR_UNIFORM_NO_REPETITIONS:
        #  for each synth contr param, a list of linearly spaced values is created (with given numerical ranges included), with 1 synth contr param value for each audio file to be generated
        #  for each audio file, for each synth contr param, 1 stochastic process is involved; a synth contr param is randomly chosen -with random.choice()- from the 
        #   corresponding set, and then that value is deleted from the set
    # LINEAR_UNIFORM_NO_REPETITIONS is guaranteed to be uniform, with no repetitions of the same value for the same synt contr param:
    # the number of unique values for each synth contr param is computed automatically and the same for all synth contr param
    LINEAR_UNIFORM_NO_REPETITIONS = 2 # NO MORE THAN 1 SAME VALUE FOR EACH SYNTH CONTR PARAM (for each synth contr param, a value appears only once in the generated dataset)
    # LINEAR_UNIFORM_ALL_COMBINATIONS:
    LINEAR_UNIFORM_ALL_COMBINATIONS = 3

################################# INPUT VARIABLES ####################################
# Only make changes here !! These dict will be dumped in a .json file for future reference
datasetGenerator_DescriptorDict = {

    'Dataset_General_Settings' : {
    
        'absolute_Path' : '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/repo/SMC_thesis/Creation_of_synthetic_Audio_datasets/SDT_FluidFlow_dataset', # Audio, .json and .csv files will be stored here
        'audio_Files_Extension' : '.wav', # if you change this, also change the object 'prepend writewave' in Max_8_OSC_receiver.maxpat
        'number_Of_AudioFiles_ToBeGenerated' : int(10), # audio dataset size, MUST be an integer
        'random_Seed' : 0, # for reproducibility
        'distribution_Of_Synthesis_Control_Parameters_Values' : Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_NO_REPETITIONS.name,
        'includeInCSVFile_ParametersValues_ScaledForMaxPDRanges' : False, # either True or False
        'dateAndTime_WhenGenerationFinished_dd/mm/YY H:M:S' : datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        },

    'Audio_Files_Settings' : {
    
        'sample_Rate' : int(44100), # problems in Max with values < 44100
        'file_Duration_Secs' : float(3), # secs 
        'quantization_Bits' : int(16), # (not used)
        'file_Names_Prefix' : 'SDT_FluidFlow', # increasing numbers will be appended (1, 2, ..., up to 'number_Of_AudioFiles_ToBeGenerated')
        
        'volume' : {
            'normalizedRandomRange_Min' : 0.9, # float, min value for generating random volume, normalized between 0. and 1.
            'normalizedRandomRange_Max' : 0.65, # float, max value for generating random volume, normalized between 0. and 1.
            'chance_Generating_New_Volume' : 100, # int, chances of generating new volume values at each file, cumulative to 100
            'chance_Retaining_Previous_File_Volume' : 0, # int, chances of not generating new volume values at each file, cumulative to 100
            'maxPDScaledRanges_Min' : 0., # min value expected in the Max/PD patch for volume control
            'maxPDScaledRanges_Max' : 158. # max value expected in the Max/PD patch for volume control
            }
        },

    'Synthesis_Control_Parameters_Settings' : {
    
        'settings' : {
            'decimalPrecisionPoints' : 2, # number of decimal points precisions for normalized 0. <-> 1. synthesis control parameters
            },

        'Synthesis_Control_Parameters' : {
    
            'avgRate' : {
                'normMinValue' : 0.05, 
                'normMaxValue' : 0.75,
                'scaledMinValue' : 0.,
                'scaledMaxValue' : 100.,
                'chance_Generating_New_Value' : 100, # only for Distribution_Of_Synthesis_Control_Parameters_Values.RANDOM_UNIFORM
                'chance_Retaining_Previous_File_Value' : 0, # only for Distribution_Of_Synthesis_Control_Parameters_Values.RANDOM_UNIFORM
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS
                },
            'minRadius' : {
                'normMinValue' : 0.1,
                'normMaxValue' : 0.2,
                'scaledMinValue' : 0.,
                'scaledMaxValue' : 100.,
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS
                },
            'maxRadius' : {
                'normMinValue' : 0.25,
                'normMaxValue' : 0.4,
                'scaledMinValue' : 0.,
                'scaledMaxValue' : 100.,
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS
                },
            'expRadius' : {
                'normMinValue' : 0.3,
                'normMaxValue' : 0.6,
                'scaledMinValue' : 0.,
                'scaledMaxValue' : 100.,
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS
                },
            'minDepth' : {
                'normMinValue' : 0.2,
                'normMaxValue' : 0.3,
                'scaledMinValue' : 0.,
                'scaledMaxValue' : 100.,
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS
                },
            'maxDepth' : {
                'normMinValue' : 0.5,
                'normMaxValue' : 0.6,
                'scaledMinValue' : 0.,
                'scaledMaxValue' : 100.,
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                 # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS
                },
            'expDepth' : {
                'normMinValue' : 0.4,
                'normMaxValue' : 0.55,
                'scaledMinValue' : 0.,
                'scaledMaxValue' : 100.,
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 2 # only for Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS
                }
            }
        },

    'OSC_Communication_Settings'    : { 
        'oscComm_IPNumber' : '127.0.0.1',
        'oscComm_PyToMaxPD_PortNumber' : 8000,
        'oscComm_MaxPDToPy_PortNumber' : 8001 # can not be the same as oscComm_PyToMaxPD_PortNumber
        }
}
###################################################################################################

###################################################################################################
######## Dataset_General_Settings ########
random.seed(datasetGenerator_DescriptorDict['Dataset_General_Settings']['random_Seed']) # for reproducibility

################ Synthesis_Control_Parameters_Settings ################
decimalPrecPoints = str('{:.') + str(datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints']) + str('f}')
synthContrParam_names = list()
synthContrParam_minMax = list()
synthContrParam_ranges = list()
synthContrParam_chanceNewVal = list()
######## Distribution_Of_Synthesis_Control_Parameters_Values == LINEAR_UNIFORM_NO_REPETITIONS ########
# each parameter has a corresponding set with unique and equally spaced values, as many as the number of audio files to be generated
synthContrParam_ForceRandDistr_ListOfLists = list()
# synthContrParam_ForceRandDistr_ListOfLists contains n elements where n is the number of synthesis control parameters specified,
# and each element is a list of generated numeric values -out of an uniform distribution- 
# when, for each new audio file to be synthesised, a value is -randomly- chosen, that value will be deleted from the list
for synthContParam in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys():
    synthContrParam_names.append(synthContParam)
    synthContrParam_minMax.append([datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMinValue'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMaxValue']])
    synthContrParam_ranges.append([datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['scaledMinValue'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['scaledMaxValue']])
    synthContrParam_chanceNewVal.append([datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['chance_Generating_New_Value'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['chance_Retaining_Previous_File_Value']])
    ######## Distribution_Of_Synthesis_Control_Parameters_Values == LINEAR_UNIFORM_NO_REPETITIONS ######## 
    listWithForcedUniformDistr_ForThisParam = numpy.linspace(datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMinValue'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMaxValue'], datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated'])
    for i in range(len(listWithForcedUniformDistr_ForThisParam)):
        listWithForcedUniformDistr_ForThisParam[i] = float(decimalPrecPoints.format(listWithForcedUniformDistr_ForThisParam[i]))
    listWithForcedUniformDistr_ForThisParam = sorted(listWithForcedUniformDistr_ForThisParam)
    # print(f'List for linear uniform distribution -no repetitions- for parameter {synthContParam}: {listWithForcedUniformDistr_ForThisParam}')
    synthContrParam_ForceRandDistr_ListOfLists.append(listWithForcedUniformDistr_ForThisParam)
######## end of Distribution_Of_Synthesis_Control_Parameters_Values == LINEAR_UNIFORM_NO_REPETITIONS ########

######## Distribution_Of_Synthesis_Control_Parameters_Values == LINEAR_UNIFORM_ALL_COMBINATIONS ########
# calculate, for all synth contr param, how many unique values there are gonna be.
def approx_factorize(numberToFactorize, listOfFactorsRatios):
    '''
    x is the positive int number (> 0) to approximately factorize
    y is a list of weights representing the wanted relative size of each factor (integers and > 0)
    returns 2 objects: a list of factors (int and > 0), and the actual number resulting from the product of the list of factors
    '''
    approxFactors = listOfFactorsRatios
    haveBestApproxFactors_BeenFound = False
    approxNumberToFactorize = 0
    loopCounter = 1
    lastDistanceToX = numberToFactorize

    while (haveBestApproxFactors_BeenFound == False):
        approxFactors = [item * loopCounter for item in listOfFactorsRatios]
        approxNumberToFactorize = numpy.prod(approxFactors)
        currentDistanceToX = abs(numberToFactorize - approxNumberToFactorize)

        # print(f'    loop counter = {loopCounter}')
        # print(f'approxFactors = {approxFactors}')
        # print(f'approxNumberToFactorize = {approxNumberToFactorize}')
        # print(f'currentDistanceToX = {currentDistanceToX}')
        # time.sleep(1)

        if currentDistanceToX == 0 :
            return approxFactors, numpy.prod(approxFactors)
        elif currentDistanceToX > lastDistanceToX:
            lastApproxFactors = [item * (loopCounter - 1) for item in listOfFactorsRatios]
            return lastApproxFactors, numpy.prod(lastApproxFactors)
        elif currentDistanceToX < lastDistanceToX:
            lastDistanceToX = currentDistanceToX # continue
            loopCounter += 1

if datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Synthesis_Control_Parameters_Values'] == Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS.name:
    promptedNumAudioFiles = datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']
    paramRelativeVariance = [datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][key]['number_Of_Minimum_Unique_SynthContrParam_Values'] for key in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys()]
    numUniqueValues_ForEachParameter, actualNumAudioFilesToGenerate_WithLINEAR_UNIFORM_ALL_COMBINATIONSDistr = approx_factorize(promptedNumAudioFiles, paramRelativeVariance)

    print('Computed number of unique synth contr param values:')
    i = 0
    for key in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys():
        print(f'    Param {key}: {numUniqueValues_ForEachParameter[i]}')
        i += 1

    print(f'You asked to generate {promptedNumAudioFiles} files.')
    if promptedNumAudioFiles != actualNumAudioFilesToGenerate_WithLINEAR_UNIFORM_ALL_COMBINATIONSDistr:
        print(f'{actualNumAudioFilesToGenerate_WithLINEAR_UNIFORM_ALL_COMBINATIONSDistr} files would instead satisfy all the combinations for the {i} synth contr param(s) with the prompted variance values.')
    userInput = ''
    while userInput != 'y' and userInput != 'n':
        userInput = input(f'Go ahead and generate {actualNumAudioFilesToGenerate_WithLINEAR_UNIFORM_ALL_COMBINATIONSDistr} Audio files  ? y = yes, n = abort program')
    if userInput == 'n':
        exit()

    # for each synth contr param, generate the unique values
    synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_Unique_Values_ListOfLists = list()
    synthContParamIterator = 0
    for synthContParam in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys():
        listWithLinearUniformAllCombinationsDistr_ForThisParam = numpy.linspace(datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMinValue'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMaxValue'], numUniqueValues_ForEachParameter[synthContParamIterator])
        for i in range(len(listWithLinearUniformAllCombinationsDistr_ForThisParam)):
            listWithLinearUniformAllCombinationsDistr_ForThisParam[i] = float(decimalPrecPoints.format(listWithLinearUniformAllCombinationsDistr_ForThisParam[i]))
        listWithLinearUniformAllCombinationsDistr_ForThisParam = sorted(listWithLinearUniformAllCombinationsDistr_ForThisParam)
        print(f'List for linear uniform distribution -all combinations- for parameter {synthContParam}: {listWithLinearUniformAllCombinationsDistr_ForThisParam}')
        synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_Unique_Values_ListOfLists.append(listWithLinearUniformAllCombinationsDistr_ForThisParam)
        synthContParamIterator += 1

    # '*' unpacks the list and passes each element as a separate argument
    synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_ListOfLists = list(product(*synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_Unique_Values_ListOfLists)) 
    for i in range(len(synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_ListOfLists)): # convert tuples into lists
        synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_ListOfLists[i] = list(synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_ListOfLists[i])

    print(f'Created {len(synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_ListOfLists)} combinations of different synth contr param values')
    print(synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_ListOfLists)
######## end of Distribution_Of_Synthesis_Control_Parameters_Values == LINEAR_UNIFORM_ALL_COMBINATIONS ########

############################################
class OscMessageReceiver(): # class OscMessageReceiver(threading.Thread):
    def __init__(self, ip, receive_from_port):
        # super(OscMessageReceiver, self).__init__()
        self.ip = ip
        self.receiving_from_port = receive_from_port

        # dispatcher is used to assign a callback to a received osc message
        self.dispatcher = Dispatcher()
        self.dispatcher.set_default_handler(self.default_handler)

        # python-osc method for establishing the UDP communication with pd
        self.server = BlockingOSCUDPServer((self.ip, self.receiving_from_port), self.dispatcher)

        self.oscMessageReceived_Flag = False
        self.count = 0

    '''
    def run(self):
        print("OscMessageReceiver Started ---")
        while 1:
            self.server.handle_request()
            if self.isOSCMessageReceiverNeeded == False:
                break
            time.sleep(0.1)
        print('OscMessageReceiver Stopped ---')
    '''

    def default_handler(self, address, *args):
        if address == '/audioFileRecording_Ended':
            self.oscMessageReceived_Flag = True
            self.count += 1
        # print(f'OSC message received: {address} {args}')

    def get_ip(self):
        return self.ip

    def get_receiving_from_port(self):
        return self.receiving_from_port

    def get_server(self):
        return self.server
    
    def change_ip_port(self, ip, port):
        self.ip = ip
        self.receiving_from_port = port
        self.server = BlockingOSCUDPServer(self.ip, self.receiving_from_port)
############################################

csvFileFieldnames = ['AudioFileName'] # .csv file header name for audio files names column
csvFileFieldnames += synthContrParam_names # add synthesis control parameters names to the .csv file header
csvFileFieldNameSuffix_ScaledParamValues = str('_Scaled')
if datasetGenerator_DescriptorDict['Dataset_General_Settings']['includeInCSVFile_ParametersValues_ScaledForMaxPDRanges']:
    for scpName in synthContrParam_names:
        csvFileFieldnames += [scpName + csvFileFieldNameSuffix_ScaledParamValues]
# initialize audio file volume last values with random values
newVolumeNorm = float(decimalPrecPoints.format(random.uniform(datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['normalizedRandomRange_Min'], datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['normalizedRandomRange_Max'])))
newVolume_MaxPDMap = round(newVolumeNorm * (datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Max'] - datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Min']) + datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Min'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints'])
print(f'Generated normalized random volume : {newVolumeNorm}')
print(f'Generated Max/PD mapped random volume : {newVolume_MaxPDMap}')
audioFilesVolume_lastValuesNorm = newVolumeNorm
audioFilesVolume_lastValues = newVolume_MaxPDMap
# initialize synthesis control parameters' last values with random values
synthContrParam_lastValues = list()
synthContrParam_lastValuesNorm = list()
for scp in range(len(synthContrParam_names)): 
    newValNorm = float(decimalPrecPoints.format(random.uniform(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1])))
    newVal_MaxPDMap = round(newValNorm * (synthContrParam_ranges[scp][1] - synthContrParam_ranges[scp][0]) + synthContrParam_ranges[scp][0], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints'])
    print(f'Generated normalized random value : {newValNorm}')
    print(f'Generated Max/PD mapped random value : {newVal_MaxPDMap}')
    synthContrParam_lastValuesNorm.append(newValNorm)
    synthContrParam_lastValues.append(newVal_MaxPDMap)

################################# OSC #################################
# Parse command line arguments
parser = argparse.ArgumentParser(description='Send OSC messages to a Max MSP 8 patch')
parser.add_argument('--host', type=str, default='localhost',
                    help='the IP address or hostname of the OSC server (default: localhost)')
parser.add_argument('--port', type=int, default=8000,
                    help='the port number of the OSC server (default: 8000)')
args = parser.parse_args()

# OSC ips / ports
ip = datasetGenerator_DescriptorDict['OSC_Communication_Settings']['oscComm_IPNumber']
# Create OSC sender
sending_to_max_pd_port = datasetGenerator_DescriptorDict['OSC_Communication_Settings']['oscComm_PyToMaxPD_PortNumber']
oscSender = udp_client.SimpleUDPClient(ip, sending_to_max_pd_port)
print(f'Started OSC sender server with Host: {ip}, and port: {sending_to_max_pd_port}')
# Create OSC receiver
receiving_from_max_pd_port = datasetGenerator_DescriptorDict['OSC_Communication_Settings']['oscComm_MaxPDToPy_PortNumber']
oscReceiver = OscMessageReceiver(ip, receiving_from_max_pd_port)
# oscReceiver.start()
print(f'Started OSC receiver server with Host: {ip}, and port: {receiving_from_max_pd_port}')
#########################################################################

'''
print(f'Prompted synthesis control parameters names and ranges:')
for scp in range(len(synthContrParam_names)):
    print(f'{synthContrParam_names[scp]} : {synthContrParam_minMax[scp][0]} to {synthContrParam_minMax[scp][1]}')
'''

oscSender.send_message('maxPDToPy_OSCPortNumber', receiving_from_max_pd_port)
oscSender.send_message('sampleRate', datasetGenerator_DescriptorDict['Audio_Files_Settings']['sample_Rate'])
oscSender.send_message('bufferLength_ms', int(datasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Duration_Secs'] * 1000))

# list of dictionaries, each dictionary represents a .csv line which will
# be saved to a file. Each dictionary/line represents an audio file name and
# synthesis control parameters used to generate the corresopnding audio file
synthContrParam_Dictlist = list()

############################################################################################################
number_Of_Files_To_Be_Generated = 0
if datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Synthesis_Control_Parameters_Values'] == Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_NO_REPETITIONS.name:
    number_Of_Files_To_Be_Generated = datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']
elif datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Synthesis_Control_Parameters_Values'] == Distribution_Of_Synthesis_Control_Parameters_Values.RANDOM_UNIFORM.name:
    number_Of_Files_To_Be_Generated = datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']
elif datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Synthesis_Control_Parameters_Values'] == Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS.name:
    number_Of_Files_To_Be_Generated = actualNumAudioFilesToGenerate_WithLINEAR_UNIFORM_ALL_COMBINATIONSDistr

# generate audio files and save synthesis control parameters
for fileNumber in range(number_Of_Files_To_Be_Generated):
    oscSender.send_message('clearBuffer', True)

    # generate audio file name and path, send message
    audioFileName = datasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + '_' + str(fileNumber + 1) + datasetGenerator_DescriptorDict['Dataset_General_Settings']['audio_Files_Extension']
    audioFilePath = os.path.join(datasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'], audioFileName)
    oscSender.send_message('filePath', audioFilePath)
    print(f'File: {audioFileName}')

    # initialise dictionary for storing data with this audio file's name
    thisAudioFile_Dict = {csvFileFieldnames[0] : audioFileName}

    # for each synthesis control parameter, generate (conditionally) a new value or use the last one,
    # then send mapped value to Max/PD and add it do dict
    for scp in range(len(synthContrParam_names)):
        if datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Synthesis_Control_Parameters_Values'] == Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_NO_REPETITIONS.name:
            newValNorm = float(decimalPrecPoints.format(random.choice(synthContrParam_ForceRandDistr_ListOfLists[scp])))
            # newValNorm = float(decimalPrecPoints.format(list(synthContrParam_ForceRandDistr_ListOfLists[scp])[0])) # test: get the first element in the list to generate values in ascending order
            synthContrParam_ForceRandDistr_ListOfLists[scp].remove(newValNorm) # if no KeyError is raised, operation was performed successfully
            newVal_MaxPDMap = round(newValNorm * (synthContrParam_ranges[scp][1] - synthContrParam_ranges[scp][0]) + synthContrParam_ranges[scp][0], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints'])
        elif datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Synthesis_Control_Parameters_Values'] == Distribution_Of_Synthesis_Control_Parameters_Values.RANDOM_UNIFORM.name:
            if random.choices([True, False], weights=synthContrParam_chanceNewVal[scp], cum_weights=None, k=1)[0]: # chose to generate  new value  
                newValNorm = float(decimalPrecPoints.format(random.uniform(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1])))
                newVal_MaxPDMap = round(newValNorm * (synthContrParam_ranges[scp][1] - synthContrParam_ranges[scp][0]) + synthContrParam_ranges[scp][0], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints'])
                synthContrParam_lastValuesNorm[scp] = newValNorm
                synthContrParam_lastValues[scp] = newVal_MaxPDMap
            else: # chose not to generate  new value  
                newValNorm = synthContrParam_lastValuesNorm[scp]
                newVal_MaxPDMap = synthContrParam_lastValues[scp]
        elif datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Synthesis_Control_Parameters_Values'] == Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS.name:
            newValNorm = synthContrParam_LINEAR_UNIFORM_ALL_COMBINATIONS_ListOfLists[fileNumber][scp]
            newVal_MaxPDMap = round(newValNorm * (synthContrParam_ranges[scp][1] - synthContrParam_ranges[scp][0]) + synthContrParam_ranges[scp][0], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints'])

        oscSender.send_message(synthContrParam_names[scp], newVal_MaxPDMap)
        thisAudioFile_Dict.update({synthContrParam_names[scp] : newValNorm})
        if datasetGenerator_DescriptorDict['Dataset_General_Settings']['includeInCSVFile_ParametersValues_ScaledForMaxPDRanges']:
            thisAudioFile_Dict.update({synthContrParam_names[scp] + csvFileFieldNameSuffix_ScaledParamValues : newVal_MaxPDMap})
        print(f'    Norm {synthContrParam_names[scp]} : {newValNorm}')
        print(f'    Max/PD map {synthContrParam_names[scp]} : {newVal_MaxPDMap}')

    # when values for all parameters have been sent,
    # generate (conditionally) and send audio file volume value
    if random.choices([True, False], weights=[datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['chance_Generating_New_Volume'], datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['chance_Retaining_Previous_File_Volume']], cum_weights=None, k=1)[0]: # chose to generate new volume or not for this file  
        newVolumeNorm = float(decimalPrecPoints.format(random.uniform(datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['normalizedRandomRange_Min'], datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['normalizedRandomRange_Max'])))
        newVolume_MaxPDMap = round(newVolumeNorm * (datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Max'] - datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Min']) + datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Min'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints'])
        print(f'    Norm volume : {newVolumeNorm}')
        print(f'    Max/PD map volume : {newVolume_MaxPDMap}')
        audioFilesVolume_lastValuesNorm = newVolumeNorm
        audioFilesVolume_lastValues = newVolume_MaxPDMap
    else: # chose not to generate  new value  
        newVolumeNorm = audioFilesVolume_lastValuesNorm
        newVolume_MaxPDMap = audioFilesVolume_lastValues
    oscSender.send_message('audioFileVolume', newVolume_MaxPDMap)

    # when values for all parameters and audio volume have been sent,
    # save the info dictionary for this audio file to the list
    synthContrParam_Dictlist.append(thisAudioFile_Dict)

    # trigger start recording, then wait until a flag message is received
    # back from Max/PD to indicate that the recording has finished
    oscSender.send_message('startRecordAudioToBuffer', True)
    while(oscReceiver.oscMessageReceived_Flag == False): # wait until the server receives the OSC message
        oscReceiver.server.handle_request()
        time.sleep(0.1)
    # message received (flag set to True by OscMessageReceiver), reset flag
    oscReceiver.oscMessageReceived_Flag = False
    print(f'Finished recording file: {audioFileName}')
################################################################################################ finished generating audio files

# print(synthContrParam_Dictlist)
if oscReceiver.count == number_Of_Files_To_Be_Generated:
    print('Finished creating synthetic dataset, no errors encountered')
else:
    print('Finished creating synthetic dataset, some errors were encountered')
'''
if datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Synthesis_Control_Parameters_Values'] == Distribution_Of_Synthesis_Control_Parameters_Values.LINEAR_UNIFORM_ALL_COMBINATIONS.name:
    if oscReceiver.count == number_Of_Files_To_Be_Generated:
        print('Finished creating synthetic dataset, no errors encountered')
    else:
        print('Finished creating synthetic dataset, some errors were encountered')
else:
    if oscReceiver.count == datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']:
        print('Finished creating synthetic dataset, no errors encountered')
    else:
        print('Finished creating synthetic dataset, some errors were encountered')
'''

# generate .csv file with audio file names and synthesis control parameters
csvFileName = datasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + str(".csv")
csvFilePath = os.path.join(datasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'], csvFileName)
with open(csvFilePath, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csvFileFieldnames, dialect='excel')
    writer.writeheader()
    for dict in synthContrParam_Dictlist:
        writer.writerow(dict)
print(f'Finished writing {csvFileName} .csv file with synthesis control parameters')

# create .json file
jsonFileName = datasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + str(".json")
jsonFilePath = os.path.join(datasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'], jsonFileName)
with open(jsonFilePath, 'w') as jsonfile:
    json.dump(datasetGenerator_DescriptorDict, jsonfile, indent=4)
print(f'Finished writing {jsonFileName} .json file with synthesis control parameters')
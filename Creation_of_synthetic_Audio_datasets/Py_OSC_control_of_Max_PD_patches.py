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
import datetime
import math
from itertools import product # to compute all combinations of synth contr param values outputs for UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION distribution
 
'''
README
This script allows you to generate a synthetic Audio dataset, by controlling a Max/PD patch via OSC messages.
The Max/PD patch is a synthesiser, which -in the context of this Project- takes as input a set of synthesis control parameters (synth contr param), and outputs an audio file.
The synth contr param, which in a Procedural Audio context represent physically-driven variables (e.g. mass, stiffness in a membrane percussion sound),
    are controlled via OSC messages sent from this script to the Max/PD patch.
    All synth contr param values for all Audio files -usable as ground truth for Machine Learning models- will also be stored in a separate .csv file.
    All synth contr param values are normalized between 0. and 1. in this script (again, useful if used as ground truth in ML models),
    and then mapped to the expected ranges -settable in this scripts' dictionary- in the Max/PD patch.

You can set some global settings for the generated dataset (e.g. number of audio files to be generated, audio files duration, path to store the files into, files names, etc.),
as well as the specific synth contr param variables (e.g. ranges and distribution), in the datasetGenerator_DescriptorDict dictionary
(which will be dumped in a .json file for future reference).
Specifically, defining the Distribution_Of_Values_For_Each_Synthesis_Control_Parameter enum data structure below,
you can control how the synth contr param values are distributed across the generated dataset.
You can only set one unique distribution type for the entire dataset, which is valid for all the marginal distributions
(the marginal distributions are the distributions of each of the individual variables, a.k.a. each of the individual synth contr param).

Probability theory concepts: summary of Joint, Marginal, and Conditional Probability Distributions (https://en.wikipedia.org/wiki/Joint_probability_distribution )
Given two random variables that are defined on the same probability space, the joint probability distribution is the corresponding probability distribution
on all possible pairs of outputs. The joint distribution can just as well be considered for any given number of random variables.
The joint distribution encodes the marginal distributions, i.e. the distributions of each of the individual random variables.
It also encodes the conditional probability distributions, which deal with how the outputs of one random variable are distributed
when given information on the outputs of the other random variable(s).
'''

 # This enum represents the possible settings for the marginal distributions, i.e. the distributions of each of the individual variables (in this case, the synthesis control parameters). 
    # The marginal distributions determine the joint and conditional distributions as well,
    # but only in UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION you can directly control the marginal and joint distributions
 # Set the enum value for the marginal distributions in datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter']
class Distribution_Of_Values_For_Each_Synthesis_Control_Parameter(Enum):
    ####################################################################
    # RANDOM_UNIFORM involves non-deterministic (except for the seed number, made for reproducibility) stochastic processes
    # yields a marginal uniform distribution, for each synth contr param, as the number of audio files to be generated approaches infinity
    # for each audio file, for each synth contr param, 2 stochastic processes are involved;
        # a binary choice is randomly taken -with random.choice()-, to decide whether to generate a new synth contr param value or to re-use the same one used in the previous file
        # if a new synth contr param value has to be generated, a value is generated randomly -with random.uniform()- within the given numerical ranges
    RANDOM_UNIFORM = 1
    ####################################################################
    # in UNIFORM_LINEARLY_SPACED_VALUES, marginal distributions are guaranteed to be uniform, but the joint distribution does not take into account all combinations of outputs between synth contr param variables
    # in order to generate synth contr param values, no stochastic process is involved:
        # n linearly spaced values are generated (min and max values included) for each synth contr param, with n = number of audio files to be generated,
        # so, for each synth contr param, there is 1 different value for each audio file and thus none of the values is repeated more than once across the entire dataset
    # the only stochastic process involved is the choice of which value to use for each audio file (in other words, a random choice is made for choosing the order of values with respect to the series of files)
    # UNIFORM_LINEARLY_SPACED_VALUES:
        #  for each synth contr param, a list of linearly spaced values is created (the given numerical ranges are included), producing 1 different synth contr param value for each audio file to be generated
        #  for each audio file, for each synth contr param, only 1 stochastic process is involved; a synth contr param is randomly chosen -with random.choice()- from the corresponding pool of possible values, and then that value is deleted from the pool of future possible values
    UNIFORM_LINEARLY_SPACED_VALUES = 2 
    ####################################################################
    # in UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION, both marginal and joint distributions are guaranteed to be uniform
        # the joint distribution is guaranteed to take into account all combinations of outputs between synth contr param variables
        # priority is given to your choice of deciding arbitrary ratios between the number of unique values for each synth contr param, useful when a synth contr param needs to have different variance than others
            # unique values are a set of non-repeated values for each synth contr param, which will be repeated in the joint distribution as many time as needed so that every possible combinatorial match between variables outputs is covered
            # this is why the prompted number of audio files to be generated is not necessarily respected, since it is computed automatically and
            # it is equal to the product of the number of unique values for each synth contr param (which is computed automatically as well, by respecting the ratios between the number of unique values for each synth contr param you prompted)
        # in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][YOUR_PARAM_NAME]['number_Of_Minimum_Unique_SynthContrParam_Values'], you can set the minimum number of unique values for each synth contr param,
        # which will be used to compute the ratios between the number of unique values for each synth contr param
        # (they will be respectively multiplied by an incremental int number, and the product of the enlarged numbers is checked against the prompted number of audio files to be generated (x): this process terminates when the closest possible match to x -with the prompted ratios- is reached)
    # UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION:
        # no stochastic process at all is involved in the generation of the synth contr param values
        # the number of unique values for each synth contr param is computed automatically by respecting the prompted ratios between the number of unique values for each synth contr param
        # you are asked to confirm the resultant number of audio files that will be generated (equal to all the possible combinations of unique variables values)
        # if you decide to proceed, for each synth contr param, a list of n linearly spaced values is created (with given numerical ranges included), where n is the computed number of unique values for that synth contr param
        # then, all combinations of unique values for all synth contr param are computed, and for each audio file, a combination of unique values is chosen
    UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION = 3

################################# INPUT VARIABLES ####################################
# Only make changes here !! These dict will be dumped in a .json file for future reference
datasetGenerator_DescriptorDict = {

    'Dataset_General_Settings' : {
    
        'absolute_Path' : '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/repo/SMC_thesis/Creation_of_synthetic_Audio_datasets/SDT_FluidFlow_dataset', # Audio, .json and .csv files will be stored here
        'audio_Files_Extension' : '.wav', # if you change this, also change the object 'prepend writewave' in Max_8_OSC_receiver.maxpat
        'number_Of_AudioFiles_ToBeGenerated' : int(100), # audio dataset size, MUST be an integer
        'random_Seed' : 0, # for reproducibility
        'distribution_Of_Values_For_Each_Synthesis_Control_Parameter' : Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION.name,
        'includeInCSVFile_ParametersValues_ScaledForMaxPDRanges' : False, # either True or False
        'dateAndTime_WhenGenerationFinished_dd/mm/YY H:M:S' : datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        },

    'Audio_Files_Settings' : {
    
        'sample_Rate' : int(44100), # problems in Max with values < 44100
        'file_Duration_Secs' : float(3), # secs 
        'quantization_Bits' : int(16), # (not used)
        'file_Names_Prefix' : 'SDT_FluidFlow', # increasing numbers will be appended (1, 2, ..., up to 'number_Of_AudioFiles_ToBeGenerated')
        
        'volume' : {
            'normalizedRandomRange_Min' : 0.9, # float, min value for generating random volume, normalized between 0. and 1.
            'normalizedRandomRange_Max' : 0.65, # float, max value for generating random volume, normalized between 0. and 1.
            'chance_Generating_New_Volume' : 1000, # int, chances of generating new volume values at each file, cumulative to 100
            'chance_Retaining_Previous_File_Volume' : 0, # int, chances of not generating new volume values at each file, cumulative to 100
            'maxPDScaledRanges_Min' : 0., # min value expected in the Max/PD patch for volume control
            'maxPDScaledRanges_Max' : 158. # max value expected in the Max/PD patch for volume control
            }
        },

    'Synthesis_Control_Parameters_Settings' : {
    
        'settings' : {
            # CAREFUL WITH THIS, as if too little decimal precision points are used, the generated values might not be unique
            'decimalPrecisionPoints' : 3, # number of decimal points precisions for normalized 0. <-> 1. synthesis control parameters
            },

        'Synthesis_Control_Parameters' : {
    
            'avgRate' : {
                'normMinValue' : 0.05, # >= 0. and <= 1.
                'normMaxValue' : 0.75, # >= 0. and <= 1.
                'scaledMinValue' : 0., # min val range in Max/PD patch
                'scaledMaxValue' : 100., # max val range in Max/PD patch
                'chance_Generating_New_Value' : 100, # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.RANDOM_UNIFORM
                'chance_Retaining_Previous_File_Value' : 0, # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.RANDOM_UNIFORM
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION
                },
            'minRadius' : {
                'normMinValue' : 0.1, # >= 0. and <= 1.
                'normMaxValue' : 0.2, # >= 0. and <= 1.
                'scaledMinValue' : 0., # min val range in Max/PD patch
                'scaledMaxValue' : 100., # max val range in Max/PD patch
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION
                },
            'maxRadius' : {
                'normMinValue' : 0.25, # >= 0. and <= 1.
                'normMaxValue' : 0.4, # >= 0. and <= 1.
                'scaledMinValue' : 0., # min val range in Max/PD patch
                'scaledMaxValue' : 100., # max val range in Max/PD patch
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION
                },
            'expRadius' : {
                'normMinValue' : 0.3, # >= 0. and <= 1.
                'normMaxValue' : 0.6, # >= 0. and <= 1.
                'scaledMinValue' : 0., # min val range in Max/PD patch
                'scaledMaxValue' : 100., # max val range in Max/PD patch
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION
                },
            'minDepth' : {
                'normMinValue' : 0.2, # >= 0. and <= 1.
                'normMaxValue' : 0.3, # >= 0. and <= 1.
                'scaledMinValue' : 0., # min val range in Max/PD patch
                'scaledMaxValue' : 100., # max val range in Max/PD patch
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION
                },
            'maxDepth' : {
                'normMinValue' : 0.5, # >= 0. and <= 1.
                'normMaxValue' : 0.6, # >= 0. and <= 1.
                'scaledMinValue' : 0., # min val range in Max/PD patch
                'scaledMaxValue' : 100., # max val range in Max/PD patch
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                 # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION
                },
            'expDepth' : {
                'normMinValue' : 0.4, # >= 0. and <= 1.
                'normMaxValue' : 0.55, # >= 0. and <= 1.
                'scaledMinValue' : 0., # min val range in Max/PD patch 
                'scaledMaxValue' : 100., # max val range in Max/PD patch
                'chance_Generating_New_Value' : 50,
                'chance_Retaining_Previous_File_Value' : 50,
                # HAS TO BE INTEGER AND > 0
                'number_Of_Minimum_Unique_SynthContrParam_Values' : 1 # only for Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION
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
######## Distribution_Of_Values_For_Each_Synthesis_Control_Parameter == UNIFORM_LINEARLY_SPACED_VALUES ########
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
    ######## Distribution_Of_Values_For_Each_Synthesis_Control_Parameter == UNIFORM_LINEARLY_SPACED_VALUES ######## 
    listWithForcedUniformDistr_ForThisParam = numpy.linspace(datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMinValue'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMaxValue'], datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated'])
    for i in range(len(listWithForcedUniformDistr_ForThisParam)):
        listWithForcedUniformDistr_ForThisParam[i] = float(decimalPrecPoints.format(listWithForcedUniformDistr_ForThisParam[i]))
    listWithForcedUniformDistr_ForThisParam = sorted(listWithForcedUniformDistr_ForThisParam)
    # print(f'List for linear uniform distribution -no repetitions- for parameter {synthContParam}: {listWithForcedUniformDistr_ForThisParam}')
    synthContrParam_ForceRandDistr_ListOfLists.append(listWithForcedUniformDistr_ForThisParam)
######## end of Distribution_Of_Values_For_Each_Synthesis_Control_Parameter == UNIFORM_LINEARLY_SPACED_VALUES ########

######## Distribution_Of_Values_For_Each_Synthesis_Control_Parameter == UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION ########
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

if datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'] == Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION.name:
    promptedNumAudioFiles = datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']
    paramRelativeVariance = [datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][key]['number_Of_Minimum_Unique_SynthContrParam_Values'] for key in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys()]
    numUniqueValues_ForEachParameter, actualNumAudioFilesToGenerate_WithUNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTIONDistr = approx_factorize(promptedNumAudioFiles, paramRelativeVariance)

    print('Computed number of unique synth contr param values:')
    i = 0
    for key in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys():
        print(f'    Param {key}: {numUniqueValues_ForEachParameter[i]}')
        i += 1

    print(f'You asked to generate {promptedNumAudioFiles} files.')
    if promptedNumAudioFiles != actualNumAudioFilesToGenerate_WithUNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTIONDistr:
        print(f'{actualNumAudioFilesToGenerate_WithUNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTIONDistr} files would instead satisfy all the combinations for the {i} synth contr param(s) with the prompted variance values.')
    userInput = ''
    while userInput != 'y' and userInput != 'n':
        userInput = input(f'Go ahead and generate {actualNumAudioFilesToGenerate_WithUNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTIONDistr} Audio files  ? y = yes, n = abort program')
    if userInput == 'n':
        exit()

    # substitute the prompted number of audio files to be generated with the actual number of audio files to be generated,
    # so that it is printed in the .json file and the end of this script
    i = 0
    for key in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys():
            datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][key]['number_Of_Unique_SynthContrParam_Values'] = numUniqueValues_ForEachParameter[i]
            del datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][key]['number_Of_Minimum_Unique_SynthContrParam_Values']
            i += 1

    # for each synth contr param, generate the unique values
    synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_Unique_Values_ListOfLists = list()
    synthContParamIterator = 0
    for synthContParam in datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'].keys():
        listWithLinearUniformAllCombinationsDistr_ForThisParam = numpy.linspace(datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMinValue'], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][synthContParam]['normMaxValue'], numUniqueValues_ForEachParameter[synthContParamIterator])
        for i in range(len(listWithLinearUniformAllCombinationsDistr_ForThisParam)):
            listWithLinearUniformAllCombinationsDistr_ForThisParam[i] = float(decimalPrecPoints.format(listWithLinearUniformAllCombinationsDistr_ForThisParam[i]))
        listWithLinearUniformAllCombinationsDistr_ForThisParam = sorted(listWithLinearUniformAllCombinationsDistr_ForThisParam)
        print(f'List for linear uniform distribution -all combinations- for parameter {synthContParam}: {listWithLinearUniformAllCombinationsDistr_ForThisParam}')
        synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_Unique_Values_ListOfLists.append(listWithLinearUniformAllCombinationsDistr_ForThisParam)
        synthContParamIterator += 1

    # '*' unpacks the list and passes each element as a separate argument
    synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_ListOfLists = list(product(*synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_Unique_Values_ListOfLists)) 
    for i in range(len(synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_ListOfLists)): # convert tuples into lists
        synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_ListOfLists[i] = list(synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_ListOfLists[i])

    print(f'Created {len(synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_ListOfLists)} combinations of different synth contr param values')
    print(synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_ListOfLists)
######## end of Distribution_Of_Values_For_Each_Synthesis_Control_Parameter == UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION ########

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
volumeFieldName = str('volume')
csvFileFieldnames += volumeFieldName
if datasetGenerator_DescriptorDict['Dataset_General_Settings']['includeInCSVFile_ParametersValues_ScaledForMaxPDRanges']:  
    csvFileFieldnames += [volumeFieldName + csvFileFieldNameSuffix_ScaledParamValues]
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
if datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'] == Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_LINEARLY_SPACED_VALUES.name:
    number_Of_Files_To_Be_Generated = datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']
elif datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'] == Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.RANDOM_UNIFORM.name:
    number_Of_Files_To_Be_Generated = datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']
elif datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'] == Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION.name:
    number_Of_Files_To_Be_Generated = actualNumAudioFilesToGenerate_WithUNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTIONDistr

startTime = time.time()
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
        if datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'] == Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_LINEARLY_SPACED_VALUES.name:
            newValNorm = float(decimalPrecPoints.format(random.choice(synthContrParam_ForceRandDistr_ListOfLists[scp])))
            # newValNorm = float(decimalPrecPoints.format(list(synthContrParam_ForceRandDistr_ListOfLists[scp])[0])) # test: get the first element in the list to generate values in ascending order
            synthContrParam_ForceRandDistr_ListOfLists[scp].remove(newValNorm) # if no KeyError is raised, operation was performed successfully
            newVal_MaxPDMap = round(newValNorm * (synthContrParam_ranges[scp][1] - synthContrParam_ranges[scp][0]) + synthContrParam_ranges[scp][0], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints'])
        elif datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'] == Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.RANDOM_UNIFORM.name:
            if random.choices([True, False], weights=synthContrParam_chanceNewVal[scp], cum_weights=None, k=1)[0]: # chose to generate  new value  
                newValNorm = float(decimalPrecPoints.format(random.uniform(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1])))
                newVal_MaxPDMap = round(newValNorm * (synthContrParam_ranges[scp][1] - synthContrParam_ranges[scp][0]) + synthContrParam_ranges[scp][0], datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['settings']['decimalPrecisionPoints'])
                synthContrParam_lastValuesNorm[scp] = newValNorm
                synthContrParam_lastValues[scp] = newVal_MaxPDMap
            else: # chose not to generate  new value  
                newValNorm = synthContrParam_lastValuesNorm[scp]
                newVal_MaxPDMap = synthContrParam_lastValues[scp]
        elif datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'] == Distribution_Of_Values_For_Each_Synthesis_Control_Parameter.UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION.name:
            newValNorm = synthContrParam_UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION_ListOfLists[fileNumber][scp]
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

    # save volume value to dict
    thisAudioFile_Dict.update({volumeFieldName : newVolumeNorm})
    if datasetGenerator_DescriptorDict['Dataset_General_Settings']['includeInCSVFile_ParametersValues_ScaledForMaxPDRanges']:
        thisAudioFile_Dict.update({volumeFieldName : newVolume_MaxPDMap})

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
endTime = time.time()
generationTimeElapsed = round(endTime - startTime)
generationTimeElapsed = str(datetime.timedelta(seconds = generationTimeElapsed))

# print(synthContrParam_Dictlist)
if oscReceiver.count == number_Of_Files_To_Be_Generated:
    print(f'Finished creating synthetic dataset ({number_Of_Files_To_Be_Generated} files in {generationTimeElapsed} time), no errors encountered')
else:
    print('Finished creating synthetic dataset, some errors were encountered')

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
datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_Generated'] = int(number_Of_Files_To_Be_Generated)
del datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']
total_Dataset_Audio_Files_Duration_secs = datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_Generated'] * datasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Duration_Secs']
datasetGenerator_DescriptorDict['Dataset_General_Settings']['total_Dataset_Audio_Files_Duration_H:M:S'] = str(datetime.timedelta(seconds = total_Dataset_Audio_Files_Duration_secs))
datasetGenerator_DescriptorDict['Dataset_General_Settings']['time_Elapsed_During_Generation_H:M:S'] = generationTimeElapsed

jsonFileName = datasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + str(".json")
jsonFilePath = os.path.join(datasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'], jsonFileName)
with open(jsonFilePath, 'w') as jsonfile:
    json.dump(datasetGenerator_DescriptorDict, jsonfile, indent=4)
print(f'Finished writing {jsonFileName} .json file with synthesis control parameters')
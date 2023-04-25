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

################################# INPUT VARIABLES ####################################
# Only make changes here !! These dict will be dumped in a .json file for future reference
datasetGenerator_DescriptorDict = {
    'Dataset_General_Settings' : {
        'absolute_Path' : '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/repo/SMC_thesis/Creation_of_synthetic_Audio_datasets/SDT_FluidFlow_dataset', # Audio, .json and .csv files will be stored here
        'audio_Files_Extension' : '.wav', # if you change this, also change the object 'prepend writewave' in Max_8_OSC_receiver.maxpat
        'number_Of_AudioFiles_ToBeGenerated' : int(10), # audio dataset size, MUST be an integer
        'random_Seed' : 0, # for reproducibility
        'includeInCSVFile_ParametersValues_ScaledForMaxPDRanges' : True # either True or False
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
        'synthContrParam_decPrecPoints' : 2, # number of decimal points precisions for normalized 0. <-> 1. synthesis control parameters
        'avgRate' : {
            'minValue' : 10,
            'maxValue' : 75,
            'chance_Generating_New_Value' : 80,
            'chance_Retaining_Previous_File_Value' : 20
            },
        'minRadius' : {
            'minValue' : 10,
            'maxValue' : 20,
            'chance_Generating_New_Value' : 25,
            'chance_Retaining_Previous_File_Value' : 75
            },
        'maxRadius' : {
            'minValue' : 25,
            'maxValue' : 40,
            'chance_Generating_New_Value' : 25,
            'chance_Retaining_Previous_File_Value' : 75
            },
        'expRadius' : {
            'minValue' : 30,
            'maxValue' : 60,
            'chance_Generating_New_Value' : 25,
            'chance_Retaining_Previous_File_Value' : 75
            },
        'minDepth' : {
            'minValue' : 20,
            'maxValue' : 30,
            'chance_Generating_New_Value' : 25,
            'chance_Retaining_Previous_File_Value' : 75
            },
        'maxDepth' : {
            'minValue' : 50,
            'maxValue' : 65,
            'chance_Generating_New_Value' : 25,
            'chance_Retaining_Previous_File_Value' : 75
            },
        'expDepth' : {
            'minValue' : 40,
            'maxValue' : 55,
            'chance_Generating_New_Value' : 25,
            'chance_Retaining_Previous_File_Value' : 75
            }
        },

    'OSC_Communication_Settings'    : { 
        'oscComm_IPNumber' : '127.0.0.1',
        'oscComm_PyToMaxPD_PortNumber' : 8000,
        'oscComm_MaxPDToPy_PortNumber' : 8001 # can not be the same as oscComm_PyToMaxPD_PortNumber
        }
}
###################################################################################################

################################# INPUT VARIABLES #################################### Only make changes here !!
######## Dataset_General_Settings ########
random.seed(datasetGenerator_DescriptorDict['Dataset_General_Settings']['random_Seed']) # for reproducibility

######## Synthesis_Control_Parameters_Settings ########
synthContrParam_decPrecPoints = 2 # number of decimal points precisions for normalized 0. <-> 1. synthesis control parameters
# SYNTHESIS CONTROL PARAMETERS NAMES
synthContrParam_names = ['avgRate', 'minRadius', 'maxRadius', 'expRadius', 'minDepth', 'maxDepth', 'expDepth']
# All parameters in this script are normalized between 0. and 1., but Max/PD may expect different ranges.
# This list describe those Max/PD ranges, one list per parameter [min, max]
synthContrParam_ranges = [[0., 100.],  [0., 100.], [0., 100.], [0., 100.], [0., 100.], [0., 100.], [0., 100.]]
# The following lists must have the same length as synthContrParam_names,
# and 1 list of length 2 for each synthesis control parameter
# SYNTHESIS CONTROL PARAMETERS RANDOM VALUES MIN/MAX RANGES [MIN, MAX], ONE LIST PER PARAMETER
synthContrParam_minMax = [[0.05, 0.75], [0.1, 0.2], [0.25, 0.4], [0.3, 0.6], [0.2, 0.3], [0.5, 0.6], [0.4, 0.55]]
# WEIGHTS FOR CHANCES OF GENERATING NEW VALUES AT EACH FILE [TRUE, FALSE], ONE LIST PER PARAMETER
synthContrParam_chanceNewVal = [[80,20], [25, 75], [25, 75], [25, 75], [25, 75], [25, 75], [25, 75]]

######## OSC_Communication_Settings ########
oscComm_IPNumber = '127.0.01'
oscComm_PyToMaxPD_PortNumber = 8000
oscComm_MaxPDToPy_PortNumber = 8001 # can not be the same as oscComm_PyToMaxPD_PortNumber
###################################################################################################

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

decimalPrecPoints = str('{:.') + str(synthContrParam_decPrecPoints) + str('f}')
csvFileFieldnames = ['AudioFileName'] # .csv file header name for audio files names column
csvFileFieldnames += synthContrParam_names # add synthesis control parameters names to the .csv file header
csvFileFieldNameSuffix_ScaledParamValues = str('_Scaled')
if datasetGenerator_DescriptorDict['Dataset_General_Settings']['includeInCSVFile_ParametersValues_ScaledForMaxPDRanges']:
    for scpName in synthContrParam_names:
        csvFileFieldnames += [scpName + csvFileFieldNameSuffix_ScaledParamValues]
# initialize audio file volume last values with random values
newVolumeNorm = float(decimalPrecPoints.format(random.uniform(datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['normalizedRandomRange_Min'], datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['normalizedRandomRange_Max'])))
newVolume_MaxPDMap = round(newVolumeNorm * (datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Max'] - datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Min']) + datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Min'], synthContrParam_decPrecPoints)
print(f'Generated normalized random volume : {newVolumeNorm}')
print(f'Generated Max/PD mapped random volume : {newVolume_MaxPDMap}')
audioFilesVolume_lastValuesNorm = newVolumeNorm
audioFilesVolume_lastValues = newVolume_MaxPDMap
# initialize synthesis control parameters' last values with random values
synthContrParam_lastValues = list()
synthContrParam_lastValuesNorm = list()
for scp in range(len(synthContrParam_names)): 
    newValNorm = float(decimalPrecPoints.format(random.uniform(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1])))
    newVal_MaxPDMap = round(newValNorm * (synthContrParam_ranges[scp][1] - synthContrParam_ranges[scp][0]) + synthContrParam_ranges[scp][0], synthContrParam_decPrecPoints)
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
ip = oscComm_IPNumber
# Create OSC sender
sending_to_max_pd_port = oscComm_PyToMaxPD_PortNumber
oscSender = udp_client.SimpleUDPClient(ip, sending_to_max_pd_port)
print(f'Started OSC sender server with Host: {ip}, and port: {sending_to_max_pd_port}')
# Create OSC receiver
receiving_from_max_pd_port = oscComm_MaxPDToPy_PortNumber
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
# generate audio files and save synthesis control parameters
for fileNumber in range(datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']):
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
        if random.choices([True, False], weights=synthContrParam_chanceNewVal[scp], cum_weights=None, k=1)[0]: # chose to generate  new value  
            newValNorm = float(decimalPrecPoints.format(random.uniform(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1])))
            newVal_MaxPDMap = round(newValNorm * (synthContrParam_ranges[scp][1] - synthContrParam_ranges[scp][0]) + synthContrParam_ranges[scp][0], synthContrParam_decPrecPoints)
            synthContrParam_lastValuesNorm[scp] = newValNorm
            synthContrParam_lastValues[scp] = newVal_MaxPDMap
        else: # chose not to generate  new value  
            newValNorm = synthContrParam_lastValuesNorm[scp]
            newVal_MaxPDMap = synthContrParam_lastValues[scp]
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
        newVolume_MaxPDMap = round(newVolumeNorm * (datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Max'] - datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Min']) + datasetGenerator_DescriptorDict['Audio_Files_Settings']['volume']['maxPDScaledRanges_Min'], synthContrParam_decPrecPoints)
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
if oscReceiver.count == datasetGenerator_DescriptorDict['Dataset_General_Settings']['number_Of_AudioFiles_ToBeGenerated']:
    print('Finished creating synthetic dataset, no errors encountered')
else:
    print('Finished creating synthetic dataset, some errors were encountered')

# generate .csv file with audio file names and synthesis control parameters
csvFileName = datasetGenerator_DescriptorDict['Audio_Files_Settings']['file_Names_Prefix'] + str(".csv")
csvFilePath = os.path.join(datasetGenerator_DescriptorDict['Dataset_General_Settings']['absolute_Path'], csvFileName)
with open(csvFilePath, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csvFileFieldnames)
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
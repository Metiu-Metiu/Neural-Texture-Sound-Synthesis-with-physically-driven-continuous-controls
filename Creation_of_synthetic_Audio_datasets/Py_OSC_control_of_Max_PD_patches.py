from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client
import argparse
import random
import time
import csv
import os
import threading

########### INPUT VARIABLES ############## Only change these variables !!
random.seed(0) # for reproducibility
pathToStoreFilesInto = '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/repo/SMC_thesis/Creation_of_synthetic_Audio_datasets/SDT_FluidFlow_dataset' # Audio files and .csv file will be stored here
audioFIlesExtension = '.wav' # if you change this, also change the object 'prepend writewave' in Max_8_OSC_receiver.maxpat
numberOfFilesToBeGenerated = 10 # dataset size
sampleRate = int(44100) # problems with values < 44100
fileDurationSecs = float(3) # secs
quantization = int(16) # bits (not used)
fileNamePrefix = 'SDT_FluidFlow' # increasing numbers will be appended (1, 2, ..., up to numberOfFilesToBeGenerated)
# SYNTHESIS CONTROL PARAMETERS NAMES
synthContrParam_names = ['avgRate', 'minRadius', 'maxRadius', 'expRadius', 'minDepth', 'maxDepth', 'expDepth']
# The following lists must have the same length as synthContrParam_names,
# and 1 list of length 2 for each synthesis control parameter
# SYNTHESIS CONTROL PARAMETERS RANDOM VALUES MIN/MAX RANGES [MIN, MAX], ONE LIST PER PARAMETER
synthContrParam_minMax = [[10,75], [10, 20], [25,40], [30, 60], [20, 30], [50, 65], [40, 55]]
# WEIGHTS FOR CHANCES OF GENERATING NEW VALUES AT EACH FILE [TRUE, FALSE], ONE LIST PER PARAMETER
synthContrParam_chanceNewVal = [[80,20], [25, 75], [25, 75], [25, 75], [25, 75], [25, 75], [25, 75]]
csvFileFieldnames = ['AudioFileName'] # .csv file header name for audio files names column
oscComm_IPNumber = '127.0.01'
oscComm_PyToMaxPD_PortNumber = 8000
oscComm_MaxPDToPy_PortNumber = 8001 # can not be the same as oscComm_PyToMaxPD_PortNumber
############################################

############################################
class OscMessageReceiver(threading.Thread):
    def __init__(self, ip, receive_from_port):
        super(OscMessageReceiver, self).__init__()
        self.ip = ip
        self.receiving_from_port = receive_from_port

        # dispatcher is used to assign a callback to a received osc message
        self.dispatcher = Dispatcher()
        self.dispatcher.set_default_handler(self.default_handler)

        # python-osc method for establishing the UDP communication with pd
        self.server = BlockingOSCUDPServer((self.ip, self.receiving_from_port), self.dispatcher)

        self.oscMessageReceived_Flag = False
        self.isOSCMessageReceiverNeeded = True

    def run(self):
        print("OscMessageReceiver Started ---")
        count = 0
        while 1:
            self.server.handle_request()
            count += 1
            if self.isOSCMessageReceiverNeeded == False:
                break
            time.sleep(0.1)
        print('OscMessageReceiver Stopped ---')

    def default_handler(self, address, *args):
        self.oscMessageReceived_Flag = True
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

csvFileFieldnames += synthContrParam_names # add synthesis control parameters names to the .csv file header
synthContrParam_lastValues = list()
for scp in range(len(synthContrParam_names)): # initialize last values with random values
    synthContrParam_lastValues.append(random.randint(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1]))

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
oscReceiver.start()
print(f'Started OSC receiver server with Host: {ip}, and port: {receiving_from_max_pd_port}')
#########################################################################

'''
print(f'Prompted synthesis control parameters names and ranges:')
for scp in range(len(synthContrParam_names)):
    print(f'{synthContrParam_names[scp]} : {synthContrParam_minMax[scp][0]} to {synthContrParam_minMax[scp][1]}')
'''

oscSender.send_message('maxPDToPy_OSCPortNumber', receiving_from_max_pd_port)
oscSender.send_message('sampleRate', sampleRate)
oscSender.send_message('bufferLength_ms', int(fileDurationSecs * 1000))

# list of dictionaries, each dictionary represents a .csv line which will
# be saved to a file. Each dictionary/line represents an audio file name and
# synthesis control parameters used to generate the corresopnding audio file
synthContrParam_Dictlist = list()

# generate audio files and save synthesis control parameters
for fileNumber in range(numberOfFilesToBeGenerated):
    oscSender.send_message('clearBuffer', True)

    # generate audio file name and path, send message
    audioFileName = fileNamePrefix + '_' + str(fileNumber + 1) + audioFIlesExtension
    audioFilePath = os.path.join(pathToStoreFilesInto, audioFileName)
    oscSender.send_message('filePath', audioFilePath)
    print(f'File: {audioFileName}')

    # initialise dictionary for storing data with this audio file's name
    thisAudioFile_Dict = {csvFileFieldnames[0] : audioFileName}

    # for each synthesis control parameter, generate a new value or use the last one
    for scp in range(len(synthContrParam_names)):
        if random.choices([True, False], weights=synthContrParam_chanceNewVal[scp], cum_weights=None, k=1)[0]:
            # print(f'Chose to generate new value for {synthContrParam_names[scp]}')
            newVal = random.randint(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1])
            synthContrParam_lastValues[scp] = newVal
        else:
            # print(f'Chose not to generate new value for {synthContrParam_names[scp]}')
            newVal = synthContrParam_lastValues[scp]

        # save the values for this parameter to the dictionary
        thisAudioFile_Dict.update({synthContrParam_names[scp] : newVal})
        print(f'    {synthContrParam_names[scp]} : {newVal}')

        # send value for this parameter
        oscSender.send_message(synthContrParam_names[scp], newVal)
    
    # when values for all parameters have been sent,
    # save the info dictionary for this audio file to the list
    synthContrParam_Dictlist.append(thisAudioFile_Dict)

    # trigger start recording, then wait until a flag message is received
    # back from Max/PD to indicate that the recording has finished
    oscSender.send_message('startRecordAudioToBuffer', True)
    while(oscReceiver.oscMessageReceived_Flag == False): # wait until the server receives the OSC message
        time.sleep(0.1)
    # message received (flag set to True by OscMessageReceiver), reset flag
    oscReceiver.oscMessageReceived_Flag = False
    print(f'Finished recording file: {audioFileName}')
    
# print(synthContrParam_Dictlist)
oscReceiver.isOSCMessageReceiverNeeded = False
oscReceiver.join(1.0)
print('Finished creating synthetic dataset')

# generate .csv file with audio file names and synthesis control parameters
csvFileName = fileNamePrefix + str(".csv")
csvFilePath = os.path.join(pathToStoreFilesInto, csvFileName)
with open(csvFilePath, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csvFileFieldnames)
    writer.writeheader()
    for dict in synthContrParam_Dictlist:
        writer.writerow(dict)
print(f'Finished writing {csvFileName} .csv file with synthesis control parameters at path:')
print(f'    {csvFilePath}')

# close OSC sender
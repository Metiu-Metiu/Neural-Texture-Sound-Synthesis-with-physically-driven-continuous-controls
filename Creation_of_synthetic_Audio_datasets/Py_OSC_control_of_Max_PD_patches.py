import argparse
from pythonosc import udp_client
import random
import time
import csv
import os

########### GLOBAL VARIABLES ##############
random.seed(0)
pathToStoreFilesInto = '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/repo/SMC_thesis/Creation_of_synthetic_Audio_datasets/SDT_FluidFlow_dataset' # Audio files and .csv file will be stored here
audioFIlesExtension = '.wav'
numberOfFilesToBeGenerated = 10
sampleRate = int(44100)
fileDuration = float(3) # secs
quantization = int(16) # bits
fileNamePrefix = 'SDT_FluidFlow' # + 1, 2, 3, ..., up to numberOfFilesToBeGenerated
# SYNTHESIS CONTROL PARAMETERS NAMES
synthContrParam_names = ['avgRate', 'minRadius', 'maxRadius', 'expRadius', 'minDepth', 'maxDepth', 'expDepth']
# SYNTHESIS CONTROL PARAMETERS RANDOM VALUES MIN/MAX RANGES
synthContrParam_minMax = [[10,75],  [10, 20],     [25,40],     [30, 60],   [20, 30],   [50, 65],    [40, 55]]
# WEIGHTS FOR CHANCES OF GENERATING NEW VALUES AT EACH FILE [TRUE, FALSE]
synthContrParam_chanceNewVal = [[80,20],  [25, 75],     [25, 75],      [25, 75],    [25, 75],    [25, 75],     [25, 75], ]
csvFileFieldnames = ['AudioFileName']
csvFileFieldnames += synthContrParam_names
############################################

synthContrParam_lastValues = list()
for scp in range(len(synthContrParam_names)):
    synthContrParam_lastValues.append(random.randint(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1]))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Send OSC messages to a Max MSP 8 patch')
parser.add_argument('--host', type=str, default='localhost',
                    help='the IP address or hostname of the OSC server (default: localhost)')
parser.add_argument('--port', type=int, default=8000,
                    help='the port number of the OSC server (default: 8000)')
args = parser.parse_args()

# Create OSC client
client = udp_client.SimpleUDPClient('127.0.0.1', args.port)

#########################################################################
print(f'Started UDP client with Host: 127.0.0.1, and port: {args.port}')
print(f'Prompted synthesis control parameters:')
for scp in range(len(synthContrParam_names)):
    print(f'{synthContrParam_names[scp]} : {synthContrParam_minMax[scp][0]} to {synthContrParam_minMax[scp][1]}')
#########################################################################

client.send_message('sampleRate', sampleRate)
client.send_message('bufferLength_ms', int(fileDuration * 1000))

# list of dictionaries, each dictionary represents an audio file and .csv line
# with ground truth synthesis control parameters
synthContrParam_Dictlist = list()

for fileNumber in range(numberOfFilesToBeGenerated):
    client.send_message('clearBuffer', True)

    audioFileName = fileNamePrefix + '_' + str(fileNumber + 1) + audioFIlesExtension
    audioFilePath = os.path.join(pathToStoreFilesInto, audioFileName)
    print(f'File: {audioFileName}')
    # print(f'File path: {audioFilePath}')
    thisAudioFile_Dict = {csvFileFieldnames[0] : audioFileName}
    for scp in range(len(synthContrParam_names)):
        if random.choices([True, False], weights=synthContrParam_chanceNewVal[scp], cum_weights=None, k=1)[0]:
            # print(f'Chose to generate new value for {synthContrParam_names[scp]}')
            newVal = random.randint(synthContrParam_minMax[scp][0], synthContrParam_minMax[scp][1])
            synthContrParam_lastValues[scp] = newVal
        else:
            # print(f'Chose not to generate new value for {synthContrParam_names[scp]}')
            newVal = synthContrParam_lastValues[scp]

        thisAudioFile_Dict.update({synthContrParam_names[scp] : newVal})
        print(f'    {synthContrParam_names[scp]} : {newVal}')
        client.send_message(synthContrParam_names[scp], newVal)
        client.send_message('filePath', audioFilePath)
    
    synthContrParam_Dictlist.append(thisAudioFile_Dict)
    # print(synthContrParam_Dictlist)

    client.send_message('startRecordAudioToBuffer', True)
    time.sleep(3.5)
    # print(f'Finished recording file: {fileName}')
    
# print(synthContrParam_Dictlist)
print('Finished creating synthetic dataset')

csvFileName = fileNamePrefix + str(".csv")
csvFilePath = os.path.join(pathToStoreFilesInto, csvFileName)


with open(csvFilePath, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csvFileFieldnames)
    writer.writeheader()
    for dict in synthContrParam_Dictlist:
        writer.writerow(dict)

print(f'Finished writing {csvFileName} .csv file with synthesis control parameters at path:')
print(csvFilePath)
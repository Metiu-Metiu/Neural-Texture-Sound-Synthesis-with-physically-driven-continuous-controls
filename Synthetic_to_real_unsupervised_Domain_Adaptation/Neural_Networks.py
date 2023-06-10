import torch
from torch import nn
from enum import Enum
import os # to save .pth file as a checkpoint
import time
from Configuration_Dictionary import configDict

class NN_Type(Enum):
    NONE = 1
    ONE_D_CONV = 2
    TWO_D_CONV = 3

########################################
class Convolutional_DynamicNet(nn.Module):
    def __init__(self,
                 inputShape, # expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
                 numberOfFeaturesToExtract,
                 numberOfConvLayers = 1,
                 kernelSizeOfConvLayers = 3,
                 strideOfConvLayers = 1,
                 numberOfFeaturesToExtract_IncremMultiplier_FromLayer1 = 1,
                 kernelSizeOfPoolingLayers = 2,
                 strideOfPoolingLayers = 2,
                 numberOfFullyConnectedLayers = 1,
                 fullyConnectedLayers_InputSizeDecreaseFactor = 2): # divider
        if len(inputShape) == 3:
            print(f'{type(self).__name__} constructor: Instantating a 1D Convolutional Neural Network')
            self.NN_Type = NN_Type.ONE_D_CONV.name
        elif len(inputShape) == 4:
            print(f'{type(self).__name__} constructor: Instantating a 2D Convolutional Neural Network')
            self.NN_Type = NN_Type.TWO_D_CONV.name
        else:
            raise Exception(f'{type(self).__name__} constructor: Input shape is not supported')
            exit()

        self.inputShape = inputShape
        numberOfInputChannels = self.inputShape[1]
        numberOfFeaturesToExtract = numberOfInputChannels * numberOfFeaturesToExtract

        super(Convolutional_DynamicNet, self).__init__()
        self.conv_blocks = nn.ModuleList()
        # Create the convolutional layers dynamically
        for convLayer in range(numberOfConvLayers):
            if convLayer == 0:
                in_channels = numberOfInputChannels
                out_channels = numberOfFeaturesToExtract
            else:
                in_channels = num_out_channels_of_previous_layer
                out_channels = in_channels * numberOfFeaturesToExtract_IncremMultiplier_FromLayer1

            if self.NN_Type == NN_Type.ONE_D_CONV.name:
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size = kernelSizeOfConvLayers, stride = strideOfConvLayers, padding = 0, groups = 1),
                    nn.LeakyReLU(configDict['neuralNetwork_Settings']['activation_Function']['negative_slope']),
                    nn.Dropout1d(configDict['neuralNetwork_Settings']['dropout_Probability']),
                    nn.BatchNorm1d(out_channels), 
                    nn.AvgPool1d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
            elif self.NN_Type == NN_Type.TWO_D_CONV.name:
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = kernelSizeOfConvLayers, stride = strideOfConvLayers, padding = 0, groups = 1),
                    nn.LeakyReLU(configDict['neuralNetwork_Settings']['activation_Function']['negative_slope']),
                    nn.Dropout2d(configDict['neuralNetwork_Settings']['dropout_Probability']),
                    nn.BatchNorm2d(out_channels),
                    nn.AvgPool2d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))

            num_out_channels_of_previous_layer = out_channels
                
        self.flattenLayer = nn.Flatten()
        # Calculate the number of features after convolutional layers
        num_features = self.CalculateInputSize_OfFirstFullyConnectedLayer()

        self.fc_blocks = nn.ModuleList()
        # Create the fully connected layers dynamically
        # for fullyConnLayer in range(numberOfFullyConnectedLayers):
        #     if numberOfFullyConnectedLayers == 1:
        #         self.fc_blocks.append(nn.Sequential(
        #             nn.Linear(num_features, numberOfFeaturesToExtract),
        #         ))
        #     elif fullyConnLayer < numberOfFullyConnectedLayers - 1:
        #         num_output_features = int(num_features / fullyConnectedLayers_InputSizeDecreaseFactor)
        #         self.fc_blocks.append(nn.Sequential(
        #             nn.Linear(num_features, num_output_features),
        #         ))
        #         num_features = num_output_features
        #     elif fullyConnLayer == numberOfFullyConnectedLayers - 1:
        #         self.fc_blocks.append(nn.Sequential(
        #             nn.Linear(num_features, numberOfFeaturesToExtract),
        #         ))

        # fixed architecture for fc layers (2 layers num_features -> 100, 100 -> numberOfFeaturesToExtract)
        self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, 100),
                    # nn.LeakyReLU(configDict['neuralNetwork_Settings']['activation_Function']['negative_slope'])
                ))
        self.fc_blocks.append(nn.Sequential(
                    nn.Linear(100, numberOfFeaturesToExtract),
                    nn.LeakyReLU(configDict['neuralNetwork_Settings']['activation_Function']['negative_slope'])
                ))

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = self.flattenLayer(x)
        # x = x.view(x.size(0), -1)  # Flatten the tensor

        for fc_block in self.fc_blocks:
            x = fc_block(x)

        return x

    def CalculateInputSize_OfFirstFullyConnectedLayer(self):
        dummy_input = torch.zeros(self.inputShape)  # Assuming input size of (batch_size, num_channels, input_size)
        dummy_output = dummy_input

        for conv_block in self.conv_blocks:
            dummy_output = conv_block(dummy_output)

        dummy_output = self.flattenLayer(dummy_output)
        num_features = dummy_output.shape[1]
        
        # num_features = dummy_output.view(dummy_output.size(0), -1).shape[1]
        return num_features
########################################

########################################
def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    # size = len(dataloader.dataset)

    cumulative_loss = 0.0
    batch_number = 1
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = loss_fn(output, target)

        # backpropagate error and update weights. https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html 
        loss.backward() # Backpropagate the prediction loss. PyTorch deposits the gradients of the loss w.r.t. each parameter.
        optimizer.step() # adjust the parameters by the gradients collected in the backward pass
        optimizer.zero_grad() # reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.

        print(f'Batch number: {batch_number}')
        # print(f'    Target: {target}')
        # print(f'    Model output: {output}')
        # if batch_number == 1:
        #     print(f'        Sample 1: Model output: {output[0]}')
        #     print(f'        Sample 1:       Target: {target[0]}')
        #     print(f'        Sample 1:         Loss: {loss_fn(output[0], target[0])}')

        print(f'        Sample 1:       Target: {target}')
        print(f'        Sample 1: Model output: {output}')
        print(f'        Sample 1: Model input: {input}')
        print(f'        Sample 1:         Loss: {loss_fn(output, target)}')

        print(f'    Loss: {loss.item()}')
        # time.sleep(20)

        cumulative_loss += loss.item()
        batch_number += 1

    print(f"Train loss of last batch: {loss.item()}")
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Mean train loss of whole epoch: {mean_loss}")
########################################

########################################
def train(nn_Model, train_dataloader, validation_dataLoader, loss_Function, optimizer, device, number_Of_Epochs):
    hasCheckpointFile_AlreadyBeenSaved = False
    checkpoint = {}
    for epoch in range(number_Of_Epochs):
        print(f"Epoch {epoch+1}")
        train_single_epoch(nn_Model, train_dataloader, loss_Function, optimizer, device)
        if validation_dataLoader is not None:
            validationLoss = validate(validation_dataLoader, nn_Model, loss_Function)
            if epoch == 0:
                lastBestValidationLoss = validationLoss
            else:
                if validationLoss > lastBestValidationLoss: # validation loss is increasing
                    if hasCheckpointFile_AlreadyBeenSaved == False:
                        print("Saving checkpoint dictionary with model...")
                        torch.save(checkpoint, os.path.join(os.path.abspath(configDict['outputFilesSettings']['outputFolder_Path']), (str(configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name']) +  str(".pth"))))
                        hasCheckpointFile_AlreadyBeenSaved = True
                        print("Checkpoint dictionary with model saved")
                    if configDict['neuralNetwork_Settings']['early_Stopping'] == True:
                        if epoch + 1 >= configDict['neuralNetwork_Settings']['minimum_NumberOfEpochsToTrain_RegardlessOfEarlyStoppingBeingActive']:
                            print("Early stopping")
                            break
                else:
                    lastBestValidationLoss = validationLoss # validation loss is decreasing
                    checkpoint = {
                        'epoch_n' : epoch + 1,
                        'validation_loss' : lastBestValidationLoss,
                        'model_state_dict' : nn_Model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                    }
                    hasCheckpointFile_AlreadyBeenSaved = False

        print("---------------------------")
    print("Finished training")
########################################

########################################
def validate(data_loader, model, loss_fn):
    # THE VALIDATION DATA LOADER SHOULD BE CREATED WITH drop_last = True
    # SO THAT ALL BATCHES HAVE THE SAME SIZE AND THE VALIDATION LOSS CAN BE MORE EASILY BE CALCULATED
    cumulative_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(configDict['pyTorch_General_Settings']['device'])
            target = target.to(configDict['pyTorch_General_Settings']['device'])

            output = model(x)
            loss = loss_fn(output, target)
            cumulative_loss += loss.item()
    
    print(f"Validation loss of last batch: {loss.item()}")
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Validation loss of whole epoch (all batches have the same size): {mean_loss}")
        
    model.train()
    return mean_loss
########################################

########################################
def test(data_loader, model, loss_fn):
    # if loader.dataset.train:
    #     print(f'Validating model on the training set')
    # else:
    #     print(f'Validating model on the test set')

    cumulative_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(configDict['pyTorch_General_Settings']['device'])
            target = target.to(configDict['pyTorch_General_Settings']['device'])

            output = model(x)
            loss = loss_fn(output, target)
            cumulative_loss += loss.item()
            print(f'    Output of the network: {output}')
            print(f'    Target: {target}')
            print(f"Batch test loss: {loss.item()}")
    
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Mean test loss over all batches: {mean_loss}")
        
    # configDict['statistics']['mean_TestLoss_OverAllBatches'] = mean_loss
    
    model.train()
    return loss
########################################

########################################
def perform_inference_byExtractingSynthesisControlParameters(data_loader, model, syntheticDataset_LabelsNames):
    # returns a dictionary like {'audioFileName.wav' : {synthContrParam1 : value1, synthContrParam2 : value2, ...}}
    # where 'audioFileName.wav' are the keys and [labels] are the values

    labelled_AudioFilesDict = dict()
    model.eval()
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(configDict['pyTorch_General_Settings']['device'])            

            batch_output = model(x)

            for audioFileIt, output in enumerate(batch_output.numpy()):
                labelled_AudioFilesDict[target[audioFileIt]] = dict()
                for labelIt, label in enumerate(output):
                    labelled_AudioFilesDict[target[audioFileIt]][syntheticDataset_LabelsNames[labelIt]] = float(label) # if not coverted to float or string, it will be a numpy.float32 or numpy.int64, which is not JSON serializable
    model.train()
    return labelled_AudioFilesDict
########################################
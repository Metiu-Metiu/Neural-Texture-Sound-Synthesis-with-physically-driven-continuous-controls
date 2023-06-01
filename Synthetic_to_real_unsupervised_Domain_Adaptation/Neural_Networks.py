import torch
from torch import nn
from enum import Enum
import os # to save .pth file as a checkpoint

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
                    nn.Dropout1d(0.4),
                    nn.BatchNorm1d(out_channels), 
                    nn.ReLU(), # outputs would all be 0. with ReLU and no normalization layer
                    nn.AvgPool1d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
            elif self.NN_Type == NN_Type.TWO_D_CONV.name:
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = kernelSizeOfConvLayers, stride = strideOfConvLayers, padding = 0, groups = 1),
                    # nn.ReLU(), # outputs would all be 0. with ReLU
                    nn.AvgPool2d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
                    # nn.MaxPool2d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
                    # nn.Dropout2d ####################################
                    # nn.BatchNorm2d ####################################
            
            num_out_channels_of_previous_layer = out_channels
                
        # Calculate the number of features after convolutional layers
        num_features = self.CalculateInputSize_OfFirstFullyConnectedLayer()

        self.fc_blocks = nn.ModuleList()
        # Create the fully connected layers dynamically
        for fullyConnLayer in range(numberOfFullyConnectedLayers):
            if numberOfFullyConnectedLayers  == 1:
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, numberOfFeaturesToExtract),
                    # nn.ReLU() # outputs would all be 0. with ReLU
                ))
            elif fullyConnLayer < numberOfFullyConnectedLayers - 1:
                num_output_features = int(num_features / fullyConnectedLayers_InputSizeDecreaseFactor)
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, num_output_features),
                    # nn.ReLU() # outputs would all be 0. with ReLU
                ))
                num_features = num_output_features
            elif fullyConnLayer == numberOfFullyConnectedLayers - 1:
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, numberOfFeaturesToExtract),
                    # nn.ReLU() # outputs would all be 0. with ReLU
                ))

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor

        for fc_block in self.fc_blocks:
            x = fc_block(x)

        return x

    def CalculateInputSize_OfFirstFullyConnectedLayer(self):
        dummy_input = torch.zeros(self.inputShape)  # Assuming input size of (batch_size, num_channels, input_size)
        dummy_output = dummy_input

        for conv_block in self.conv_blocks:
            dummy_output = conv_block(dummy_output)
        
        num_features = dummy_output.view(dummy_output.size(0), -1).shape[1]
        return num_features
########################################

########################################
def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    cumulative_loss = 0.0
    batch_number = 1
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # print(f"Batch number {batch_number}")

        output = model(input)
        loss = loss_fn(output, target)
        # print(f'    Model output: {output}')
        # print(f'    Target: {target}')

        # backpropagate error and update weights
        optimizer.zero_grad()
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()

        # print(f"    Train loss at batch number {batch_number} : {loss.item()}")

        cumulative_loss += loss.item()
        batch_number += 1

    print(f"Train loss of last batch: {loss.item()}")
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Mean train loss of whole epoch: {mean_loss}")
########################################

########################################
def train(nn_Model, train_dataloader, validation_dataLoader, loss_Function, optimizer, device, number_Of_Epochs):
    lastBest_ModelStateDict = dict()
    lastBest_OptimizerStateDict = dict()

    for epoch in range(number_Of_Epochs):
        print(f"Epoch {epoch+1}")
        train_single_epoch(nn_Model, train_dataloader, loss_Function, optimizer, device)
        if validation_dataLoader is not None:
            validationLoss = validate(validation_dataLoader, nn_Model, loss_Function)
            if epoch == 0:
                lastBestValidationLoss = validationLoss
            else:
                if validationLoss > lastBestValidationLoss: # validation loss is increasing
                    print("Saving checkpoint dictionary with model...")
                    torch.save(checkpoint, os.path.join(os.path.abspath(configDict['outputFilesSettings']['outputFolder_Path']), (str(configDict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name']) +  str(".pth"))))
                    print("Checkpoint dictionary with model saved")
                    # if early stopping is enabled
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
            print(f"Batch test loss: {loss.item()}")
            # print(f'    Output of the network: {output}')
            # print(f'    Target: {target}')
    
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Mean test loss over all batches: {mean_loss}")
        
    configDict['statistics']['mean_TestLoss_OverAllBatches'] = mean_loss
    
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
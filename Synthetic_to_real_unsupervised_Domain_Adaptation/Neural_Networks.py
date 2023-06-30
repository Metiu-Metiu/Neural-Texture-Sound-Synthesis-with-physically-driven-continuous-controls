import torch
from torch import nn
from enum import Enum
import os # to save .pth file as a checkpoint
import json
import time

class NN_Type(Enum):
    NONE = 1
    ONE_D_CONV = 2
    TWO_D_CONV = 3

########################################
class Convolutional_DynamicNet(nn.Module):
    def __init__(self,
                 inputShape, # expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
                 numberOfFeaturesToExtract,
                 config_Dict,
                 createOnlyConvLayers = False): 
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

        numberOfFeaturesToExtract_IncremMultiplier_FromLayer1 = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFeaturesToExtract_IncremMultiplier_FromLayer1']
        numberOfConvLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfConvLayers']
        kernelSizeOfConvLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['kernelSizeOfConvLayers']
        strideOfConvLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['strideOfConvLayers']
        kernelSizeOfPoolingLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['kernelSizeOfPoolingLayers']
        strideOfPoolingLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['strideOfPoolingLayers']
        numberOfFullyConnectedLayers = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['numberOfFullyConnectedLayers']
        fullyConnectedLayers_InputSizeDecreaseFactor = config_Dict['neuralNetwork_Settings']['arguments_For_Convolutional_DynamicNet_Constructor']['fullyConnectedLayers_InputSizeDecreaseFactor']
        leakyReLU_NegativeSlope = config_Dict['neuralNetwork_Settings']['activation_Function']['negative_slope']

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
                    nn.BatchNorm1d(out_channels), 
                    # nn.ReLU(),
                    nn.LeakyReLU(leakyReLU_NegativeSlope),
                    nn.MaxPool1d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
            elif self.NN_Type == NN_Type.TWO_D_CONV.name:
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = kernelSizeOfConvLayers, stride = strideOfConvLayers, padding = 0, groups = 1),
                    nn.BatchNorm2d(out_channels),
                    # nn.ReLU(),
                    nn.LeakyReLU(leakyReLU_NegativeSlope),
                    nn.MaxPool2d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))

            num_out_channels_of_previous_layer = out_channels
                
        self.flattenLayer = nn.Flatten()
        # Calculate the number of features after convolutional layers
        num_features = self.CalculateInputSize_OfFirstFullyConnectedLayer()

        self.createOnlyConvLayers = createOnlyConvLayers
        if not self.createOnlyConvLayers:
            self.fc_blocks = nn.ModuleList()
            # Create the fully connected layers dynamically
            for fullyConnLayer in range(numberOfFullyConnectedLayers):
                if numberOfFullyConnectedLayers == 1:
                    self.fc_blocks.append(nn.Sequential(
                        nn.Linear(num_features, numberOfFeaturesToExtract),
                        nn.LeakyReLU(leakyReLU_NegativeSlope),
                        # nn.Dropout1d(config_Dict['neuralNetwork_Settings']['dropout_Probability']),
                    ))
                elif fullyConnLayer < numberOfFullyConnectedLayers - 1:
                    num_output_features = int(num_features / fullyConnectedLayers_InputSizeDecreaseFactor)
                    self.fc_blocks.append(nn.Sequential(
                        nn.Linear(num_features, num_output_features),
                        nn.LeakyReLU(leakyReLU_NegativeSlope),
                        # nn.Dropout1d(config_Dict['neuralNetwork_Settings']['dropout_Probability']),
                    ))
                    num_features = num_output_features
                elif fullyConnLayer == numberOfFullyConnectedLayers - 1:
                    self.fc_blocks.append(nn.Sequential(
                        nn.Linear(num_features, numberOfFeaturesToExtract),
                        nn.LeakyReLU(leakyReLU_NegativeSlope),
                    ))

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = self.flattenLayer(x)
        # x = x.view(x.size(0), -1)  # Flatten the tensor

        if not self.createOnlyConvLayers:
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
class SyntheticAndReal_Sounds_Classifier_FullyConnectedLayers(nn.Module):
    def __init__(self,
                 inputShape, # expects tuple or TORCH.TENSOR.SIZE representing number of input dimensions as (batch_size, channels, width) or (batch_size, channels, height, width), use torch.tensor.shape 
                 numberOfFullyConnectedLayers,
                 fullyConnectedLayers_InputSizeDecreaseFactor): 
        print(f'SyntheticAndReal_Sounds_Classifier_FullyConnectedLayers constructor : input shape = {inputShape}')
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

        super(SyntheticAndReal_Sounds_Classifier_FullyConnectedLayers, self).__init__()
        
        num_features = self.CalculateInputSize_OfFirstFullyConnectedLayer()
        print(f'SyntheticAndReal_Sounds_Classifier_FullyConnectedLayers num_features = {num_features}')

        self.fc_blocks = nn.ModuleList()
        # Create the fully connected layers dynamically
        for fullyConnLayer in range(numberOfFullyConnectedLayers):
            if numberOfFullyConnectedLayers == 1:
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, 1),
                    nn.Sigmoid(),
                    # nn.Dropout1d(config_Dict['neuralNetwork_Settings']['dropout_Probability']),
                ))
            elif fullyConnLayer < numberOfFullyConnectedLayers - 1:
                num_output_features = int(num_features / fullyConnectedLayers_InputSizeDecreaseFactor)
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, num_output_features),
                    nn.LeakyReLU(0.9),
                    # nn.Dropout1d(config_Dict['neuralNetwork_Settings']['dropout_Probability']),
                ))
                num_features = num_output_features
            elif fullyConnLayer == numberOfFullyConnectedLayers - 1:
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, 1),
                    nn.Sigmoid(),
                ))

    def forward(self, x):
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        return x

    def CalculateInputSize_OfFirstFullyConnectedLayer(self):
        dummy_input = torch.zeros(self.inputShape)  # Assuming input size of (batch_size, num_channels, input_size)
        dummy_output = dummy_input

        num_features = dummy_output.shape[2]
        
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
        if batch_number == 1:
            print(f'        Sample 1: Model output: {output[0]}')
            print(f'        Sample 1:       Target: {target[0]}')
            print(f'        Sample 1:         Loss: {loss_fn(output[0], target[0])}')

        print(f'    Loss: {loss.item()}')

        cumulative_loss += loss.item()
        batch_number += 1

    print(f"Train loss of last batch: {loss.item()}")
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Mean train loss of whole epoch: {mean_loss}")
########################################

########################################
def train(nn_Model, train_dataloader, validation_dataLoader, loss_Function, optimizer, device, number_Of_Epochs, config_Dict):
    hasCheckpointFile_AlreadyBeenSaved = False
    checkpoint = {}
    for epoch in range(number_Of_Epochs):
        print(f"Epoch {epoch+1}")
        train_single_epoch(nn_Model, train_dataloader, loss_Function, optimizer, device)
        if validation_dataLoader is not None:
            validationLoss = validate(validation_dataLoader, nn_Model, loss_Function, config_Dict)
            if epoch == 0:
                lastBestValidationLoss = validationLoss
            else:
                if validationLoss > lastBestValidationLoss: # validation loss is increasing
                    if hasCheckpointFile_AlreadyBeenSaved == False:
                        print("Saving checkpoint dictionary with model...")
                        torch.save(checkpoint, os.path.join(os.path.abspath(config_Dict['outputFilesSettings']['outputFolder_Path']), (str(config_Dict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name']) +  str(".pth"))))
                        checkpoint_JSonDict = {
                            'epoch_n' : checkpoint['epoch_n'],
                            'validation_loss' : checkpoint['validation_loss'],
                        }
                        with open(os.path.join(os.path.abspath(config_Dict['outputFilesSettings']['outputFolder_Path']), (str(config_Dict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name']) + str('_Checkpoint') + str(".json"))), 'w') as jsonfile:
                            json.dump(checkpoint_JSonDict, jsonfile, indent=4)
                        hasCheckpointFile_AlreadyBeenSaved = True
                        print("Checkpoint dictionary with model saved")
                    if config_Dict['neuralNetwork_Settings']['early_Stopping'] == True:
                        if epoch + 1 >= config_Dict['neuralNetwork_Settings']['minimum_NumberOfEpochsToTrain_RegardlessOfEarlyStoppingBeingActive']:
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
def train_single_epoch_FCLayers_withFrozenConvLayers(frozen_nn_Model, 
                       model,
                       data_loader,
                       loss_fn,
                       optimizer,
                       device):
    # size = len(dataloader.dataset)

    frozen_nn_Model.eval()
    for param in frozen_nn_Model.parameters():
        param.requires_grad = False

    cumulative_loss = 0.0
    batch_number = 1
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        input = frozen_nn_Model(input)

        output = model(input)
        loss = loss_fn(output, target)

        # backpropagate error and update weights. https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html 
        loss.backward() # Backpropagate the prediction loss. PyTorch deposits the gradients of the loss w.r.t. each parameter.
        optimizer.step() # adjust the parameters by the gradients collected in the backward pass
        optimizer.zero_grad() # reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.

        print(f'Batch number: {batch_number}')
        print(f'        Sample 1: Model output: {output[0]}')
        print(f'        Sample 1:       Target: {target[0]}')
        print(f'        Sample 1:         Loss: {loss_fn(output[0], target[0])}')

        print(f'    Loss: {loss.item()}')

        cumulative_loss += loss.item()
        batch_number += 1

    print(f"Train loss of last batch: {loss.item()}")
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Mean train loss of whole epoch: {mean_loss}")
########################################

########################################
def train_FCLayers_withFrozenConvLayers(frozen_nn_Model,
                          nn_Model,
                          train_dataloader,
                          validation_dataLoader,
                          loss_Function,
                          optimizer,
                          device,
                          number_Of_Epochs,
                          config_Dict):
    frozen_nn_Model.eval()
    for param in frozen_nn_Model.parameters():
        param.requires_grad = False
    stringSuffix = str('_FCLayers_SyntheticAndRealAudioClassifier')
    hasCheckpointFile_AlreadyBeenSaved = False
    checkpoint = {}
    for epoch in range(number_Of_Epochs):
        print(f"Epoch {epoch+1}")
        train_single_epoch_FCLayers_withFrozenConvLayers(frozen_nn_Model, nn_Model, train_dataloader, loss_Function, optimizer, device)
        if validation_dataLoader is not None:
            validationLoss = validate_FCLayers_withFrozenConvLayers(validation_dataLoader, frozen_nn_Model, nn_Model, loss_Function, config_Dict)
            if epoch == 0:
                lastBestValidationLoss = validationLoss
            else:
                if validationLoss > lastBestValidationLoss: # validation loss is increasing
                    if hasCheckpointFile_AlreadyBeenSaved == False:
                        print("Saving checkpoint dictionary with model...")
                        torch.save(checkpoint, os.path.join(os.path.abspath(config_Dict['outputFilesSettings']['outputFolder_Path']), (str(config_Dict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name']) + stringSuffix + str(".pth"))))
                        checkpoint_JSonDict = {
                            'epoch_n' : checkpoint['epoch_n'],
                            'validation_loss' : checkpoint['validation_loss'],
                        }
                        with open(os.path.join(os.path.abspath(config_Dict['outputFilesSettings']['outputFolder_Path']), (str(config_Dict['outputFilesSettings']['pyTorch_NN_StateDict_File_Name']) + stringSuffix + str('_Checkpoint') + str(".json"))), 'w') as jsonfile:
                            json.dump(checkpoint_JSonDict, jsonfile, indent=4)
                        hasCheckpointFile_AlreadyBeenSaved = True
                        print("Checkpoint dictionary with model saved")
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
def validate(data_loader, model, loss_fn, config_Dict):
    # THE VALIDATION DATA LOADER SHOULD BE CREATED WITH drop_last = True
    # SO THAT ALL BATCHES HAVE THE SAME SIZE AND THE VALIDATION LOSS CAN BE MORE EASILY BE CALCULATED
    cumulative_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(config_Dict['pyTorch_General_Settings']['device'])
            target = target.to(config_Dict['pyTorch_General_Settings']['device'])

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
def validate_FCLayers_withFrozenConvLayers(data_loader, frozen_nn_Model, model, loss_fn, config_Dict):
    # THE VALIDATION DATA LOADER SHOULD BE CREATED WITH drop_last = True
    # SO THAT ALL BATCHES HAVE THE SAME SIZE AND THE VALIDATION LOSS CAN BE MORE EASILY BE CALCULATED
    cumulative_loss = 0.0
    frozen_nn_Model.eval()
    model.eval()
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(config_Dict['pyTorch_General_Settings']['device'])
            target = target.to(config_Dict['pyTorch_General_Settings']['device'])

            x = frozen_nn_Model(x)

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
def test_FCLayers_withFrozenConvLayers(data_loader, frozen_model, model, loss_fn, config_Dict):
    # if loader.dataset.train:
    #     print(f'Validating model on the training set')
    # else:
    #     print(f'Validating model on the test set')

    cumulative_loss = 0.0
    frozen_model.eval()
    model.eval()
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(config_Dict['pyTorch_General_Settings']['device'])
            target = target.to(config_Dict['pyTorch_General_Settings']['device'])

            x = frozen_model(x)

            output = model(x)
            loss = loss_fn(output, target)
            cumulative_loss += loss.item()
            print(f'    Output of the network: {output}')
            print(f'    Target: {target}')
            print(f"Batch test loss: {loss.item()}")
    
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Mean test loss over all batches: {mean_loss}")
            
    model.train()
    return loss
########################################

########################################
def test(data_loader, model, loss_fn, config_Dict):
    # if loader.dataset.train:
    #     print(f'Validating model on the training set')
    # else:
    #     print(f'Validating model on the test set')

    cumulative_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(config_Dict['pyTorch_General_Settings']['device'])
            target = target.to(config_Dict['pyTorch_General_Settings']['device'])

            output = model(x)
            loss = loss_fn(output, target)
            cumulative_loss += loss.item()
            print(f'    Output of the network: {output}')
            print(f'    Target: {target}')
            print(f"Batch test loss: {loss.item()}")
    
    mean_loss = cumulative_loss / len(data_loader)
    print(f"Mean test loss over all batches: {mean_loss}")
            
    model.train()
    return loss
########################################

########################################
def perform_inference_byExtractingSynthesisControlParameters(data_loader, model, syntheticDataset_LabelsNames, config_Dict):
    # returns a dictionary like {'audioFileName.wav' : {synthContrParam1 : value1, synthContrParam2 : value2, ...}}
    # where 'audioFileName.wav' are the keys and [labels] are the values

    labelled_AudioFilesDict = dict()
    model.eval()
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(config_Dict['pyTorch_General_Settings']['device'])            

            batch_output = model(x)

            for audioFileIt, output in enumerate(batch_output.numpy()):
                labelled_AudioFilesDict[target[audioFileIt]] = dict()
                for labelIt, label in enumerate(output):
                    labelled_AudioFilesDict[target[audioFileIt]][syntheticDataset_LabelsNames[labelIt]] = float(label) # if not coverted to float or string, it will be a numpy.float32 or numpy.int64, which is not JSON serializable
    model.train()
    return labelled_AudioFilesDict
########################################
import torch
from torch import nn
from enum import Enum

class NN_Type(Enum):
    NONE = 1
    ONE_D_CONV = 2
    TWO_D_CONV = 3

# functional syntax
Color = Enum('Color', ['RED', 'GREEN', 'BLUE'])

######################################################################################################################################################
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
                    nn.ReLU(),
                    nn.AvgPool1d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
                    # nn.MaxPool1d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
            elif self.NN_Type == NN_Type.TWO_D_CONV.name:
                self.conv_blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = kernelSizeOfConvLayers, stride = strideOfConvLayers, padding = 0, groups = 1),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
                    # nn.MaxPool2d(kernel_size = kernelSizeOfPoolingLayers, stride = strideOfPoolingLayers)))
            
            num_out_channels_of_previous_layer = out_channels
                
        # Calculate the number of features after convolutional layers
        num_features = self.CalculateInputSize_OfFirstFullyConnectedLayer()

        self.fc_blocks = nn.ModuleList()
        # Create the fully connected layers dynamically
        for fullyConnLayer in range(numberOfFullyConnectedLayers):
            if numberOfFullyConnectedLayers  == 1:
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, numberOfFeaturesToExtract),
                    nn.ReLU()
                ))
            elif fullyConnLayer < numberOfFullyConnectedLayers - 1:
                num_output_features = int(num_features / fullyConnectedLayers_InputSizeDecreaseFactor)
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, num_output_features),
                    nn.ReLU()
                ))
                num_features = num_output_features
            elif fullyConnLayer == numberOfFullyConnectedLayers - 1:
                self.fc_blocks.append(nn.Sequential(
                    nn.Linear(num_features, numberOfFeaturesToExtract),
                    nn.ReLU()
                ))

        # self.output_layer = nn.Linear(num_features, 1)  

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor

        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # x = self.output_layer(x)
        return x

    def CalculateInputSize_OfFirstFullyConnectedLayer(self):
        dummy_input = torch.zeros(self.inputShape)  # Assuming input size of (batch_size, num_channels, input_size)
        dummy_output = dummy_input

        for conv_block in self.conv_blocks:
            dummy_output = conv_block(dummy_output)
        
        num_features = dummy_output.view(dummy_output.size(0), -1).shape[1]
        return num_features
######################################################################################################################################################

######################################################################################################################################################
class FF_NN(nn.Module):
    def __init__(self, input_shape):
        super(FF_NN, self).__init__()
        self.flatten = nn.Linear(input_shape, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.flatten(x)
        y = self.relu(y)
        return y
######################################################################################################################################################

######################################################################################################################################################
# Neural network
class SynthesisControlParameters_Extractor_Network(nn.Module):
    def __init__(self):
        super(SynthesisControlParameters_Extractor_Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (1,1)))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (1,1)))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (1,1)))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (1,1)))
        
        self.flatten = nn.Flatten()
        '''
        To determine the size of the inputs for the Linear layer in your network,
        you need to know the number of output features from the preceding Conv2d layers.
        In your case, the number of output features from the last Conv2d layer (self.conv4) is 56.
        Since you're using nn.MaxPool2d with a kernel size of 2 after each convolutional layer,
        the spatial dimensions (height and width) of the feature maps are reduced by a factor of 2 after each pooling operation.
        This means that the height and width of the feature maps at the end of self.conv4 are 1/16th of the original input size.
        Given that you have 56 output channels and the spatial dimensions are reduced by a factor of 16,
        the total number of input features for the Linear layer can be calculated as follows:
        '''
        linearLayerInput_size = 905160 # 40 * int(401 / 16) * int(61 / 16)
        self.linear = nn.Linear(linearLayerInput_size, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.conv3(out)
        # print(out.shape)
        out = self.conv4(out)
        # print(out.shape)
        out = self.flatten(out)
        out = self.relu(out)
        # print(out.shape)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        # print(type(out))
        out = torch.tensor(out, dtype=torch.float64)
        # out.requires_grad = True
        return out
######################################################################################################################################################

###########################
def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        #calculate loss
        output = model(input) 
        # print(f'Model output: {output}')
        # print(f'Model output type: {type(output)}')
        # print(f'Model output shape: {output.shape}')
        # print(f'Model target: {target}')
        # print(f'Model target type: {type(target)}')
        # print(f'Model target shape: {target.shape}')
        loss = loss_fn(output, target)

        # backpropagate error and update weights
        optimizer.zero_grad()
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")

    
def train(nn_Model, train_dataloader, loss_Function, optimizer, device, number_Of_Epochs):
    for epoch in range(number_Of_Epochs):
        print(f"Epoch {epoch+1}")
        train_single_epoch(nn_Model, train_dataloader, loss_Function, optimizer, device)
        print("---------------------------")
    print("Finished training")
########################################
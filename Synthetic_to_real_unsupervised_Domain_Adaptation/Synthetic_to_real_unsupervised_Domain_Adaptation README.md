# Synthetic_to_real_unsupervised_Domain_Adaptation
This folder represents one of the main parts of the Project, the one that deals with labelling the real audio files datasets-subsets with continuous Synthesis Control Parameters values, which will be used in the future for conditioning the real time Synthesis of the same type of sounds.
Specifically, this module articulates in two main parts:
- Designing and training a Convolutional-based Neural Network, representing a Synthesis Control Parameters extractor, which is able to extract the Synthesis Control Parameters values from the audio files of the synthetic dataset.
- Adapting the afore-mentioned pre-trained Neural Network to the distribution of the real audio files dataset, in order to be able to extract the Synthesis Control Parameters values from the real audio files of the real dataset. Specifically, this part articulates in two main parts:
  - Design and train a binary synthetic/real sounds classifier Neural Network, consisting on Fully-Connected layers only, which takes as input the (frozen, pre-trained and not learnable) the last Convolutional Layer (in fact, the Flatten layer) of the afore-mentioned Synthesis Control Parameters extractor network. This network is trained on the synthetic and real audio files datasets, in order to be able to distinguish between the two distributions.
  - Design and train a new Neural Network, consisting on Convolutional Layers only, out of the pre-trained Convolutional Layers of the Synthesis Control Parameters extractor. This network is trained on the real audio files dataset, and its output is connected to the previosly-created synthetic/real sounds classifier Neural Network. The aim of this network is to adapt the (target) domain distribution of the real dataset, to the (source) domain distribution of the synthetic dataset, in order to be able to adapt the Synthesis Control Parameters extractor to the real dataset, and to extract the Synthesis Control Parameters values from the real audio files of the real dataset.

## <strong>HOW TO USE</strong>

First, configure the Configuration_Dictionary.py as you prefer, then run the Python scripts one after the other in the following order (assuming you have 2 folders with synthetic and real sounds datasets, see other README.md files for more details):

- SynthesisControlParameters_OfSyntheticSounds_Extractor_NN.py
- RealSounds_Labeler.py (optional)
- SyntheticToRealUnsupervisedDomainAdaptation_SyntheticAndRealSoundsClassifier.py
- SyntheticToRealUnsupervisedDomainAdaptation_TargetDomainConvLayers_Adaptation.py
- SyntheticToRealUnsupervisedDomainAdaptation_TargetDomainConvLayers_RealSoundsLabeler.py

## Dataset_Wrapper.py

This class extends the torch.utils.data.dataset class and it is a custom Dataset wrapper for the synthetic and real sounds datasets.
In addition to the transforms specified in the Configuration_Dictionary.py, noise can be applied (with, again, options specified in the Configuration_Dictionary.py) to the audio files depending on an optional argument passed to the constructor of the class (applyNoise).
In this project, noise has been applied to Synthetic sounds.
Also, the supervised_Task argument passed to the constructor of the class specifies whether the dataset is used for a supervised task (in which case, will return the Synthesis Control Parameters values as labels) or for an unsupervised task (in which case, will return the audio files names as labels).

In case of MelSpectrogram transforms (used as example case in this Project), a DB-scale conversion is performed afterwards and the resulting tensor is normalized (this process yields much more robust networks).

## Neural_Networks.py

This file contains the definition of the Neural Networks classes used in this Project, as well as some utility functions to train, validate and test them.
The main class definition is Convolutional_DynamicNet, which is a Convolutional Neural Network with a dynamic number of parameters such as number of Convolutional Layers, their kernel size, number of Fully Connected layers, etc. specified in the Configuration_Dictionary.py.
The size of the Fully Connected layers' input is automatically calculated, so that the construction of the network architecture is completely automatic and dynamic.
For example, a thing that needs to be specified in the Configuration_Dictionary.py is the number of Synthesis Control Parameters to be extracted, which is the size of the output of the network.

The only a propri decision is that Convolutional Layers are followed by the following Layers types as they proved to be effective:
 - nn.BatchNorm1d
 - nn.LeakyReLU
 - nn.MaxPool1d
LeakyReLU is used because the task is a regression task, which needs to output positive numerical values.

## virtEnv_SynthToRealUnsup_DA

Virtual environment folder with necessary dependencies installed.

Some packages are strictly related to Software development (torch, torchvision, torchaudio), others are VSCode extensions (jupyter for editing .ipynb files in VSCode, pandas for using the Data Viewer in VSCode, torch_tb_profiler for using TensorBoard profiler and PyTorch profiler in VSCode) installed for the sake of a better programming environment.
Developing on MacOS Catalina 10.15.7 (19H15), Visual Studio Code (VSCode) 1.78.2 (Universal), Python 3.9.0.

Commands used:

- python3 -m venv virtEnv_SynthToRealUnsup_DA

- source virtEnv_SynthToRealUnsup_DA/bin/activate

- pip3 install torch torchvision torchaudio (https://pytorch.org/get-started/locally/)

- pip3 install jupyter (https://code.visualstudio.com/docs/datascience/pytorch-support#:~:text=Data%20Viewer%20support%20for%20Tensors%20and%20data%20slices&text=To%20access%20the%20Data%20Viewer,of%20the%20Tensor%20as%20well.)

- pip3 install pandas (required for VSCode's Data Viewer)

- pip3 install torch_tb_profiler (https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/)

- pip install matplotlib (https://matplotlib.org/)
  
- pip3 install librosa
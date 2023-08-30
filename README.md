# Sound and Music Computing Master thesis
## Neural Texture Sound Synthesis with physically-driven continuous controls using synthetic-to-real unsupervised Domain Adaptation

The Project is developed in the context of the Master's Thesis in Sound and Music Computing at the Music Technology Group of the Universitat Pompeu Fabra, Barcelona, Spain.
The Neural Network Architectures design, training, validation and testing are performed with PyTorch.

The Project encompasses the whole pipeline of
- labelling realistic sound files datasets with continuous, physically-driven Synthesis Control Parameters (e.g. numerical values controllable with slider or knobs, representing physical properties of the objects involved in the sound creation, like 'average rate of bubbles' for streaming water sounds).
  This module of the Project is represented by the first 3 sub-parts of the Project (see below).
- use the labelled realistic-sounds dataset to condition the Sound Synthesis process, in order to obtain a Neural Network that can perform real-time Texture Sound Synthesis with humanly meaningful and continuous controls.
  This module of the Project is represented by the last sub-part of the Project (see below).
  
This particular piece of Research is undertaken on Texture sounds, particularly on sounds of streaming water, or flowing water, but the code (mainly Python) is designed and structured in such a way so that it can be easily adapted to other kinds of sounds, to other kinds of Synthesis Control Parameters, to other datasets, to other Convolutional-based Neural Networks Architectures, etc..
Each of the 4 sub-parts/folders exposes an interface to the user, which mainly consists of:
- a JSON dictionary configuration file, which is used to specify the parameters of the intended task (e.g. the number of sound samples to create, the duration of each sound sample, the tags to take into account when extracting the sub-set of the canonical dataset, etc.), and it is designed to be as human-readable, portable, and human-friendly as possible. Only for the Creation_of_synthetic_Audio_datasets and Creation_of_real_Audio_datasets sub-repos, the JSON configuration dictionary is not in a separate file but is directly written in the Python script, for simplicity.
- one or more Python scripts, runnable from the command line. The user does not need to modify the Python scripts; it is just necessary to modify the JSON files and run the relative Python scripts

This Project is articulated in 4 main sub-repositories, each of which corresponds to a particular part and milestone of the Project, and is contained in a specific folder (listed in order of intended use) with dedicated READMEs, scripts, configuration files and Python environments (with relative requirements.txt files, etc.):
- ## Creation_of_synthetic_Audio_datasets
    Software dedicated to the creation of datasets of synthetic sound files, by controlling Max/PD patches with already-made sound engines.
    The software focuses on various high-level characteristics of the intended Dataset to be created (synthesis control parameters distribution, number of sound samples, duration, etc.).
    Together with the .wav Audio files created at the specified path, a .csv file is created, containing the file name and the values of the Synthesis Control Parameters used to create it, as well as a copy of the configuration dictionary used to create the dataset.

- ## Creation_of_real_Audio_datasets
    Software dedicated to the creation of real sound files datasets, by creating sub-sets of pre-existing Audio datasets.
    The software focuses on the policies for which tags/labels to consider and/or to exclude in the created subset (many available Datasets loader libraries do not offer these kind of functionality in depth), as well as creating segments of audio files chunks by specyfing a desired duration. Silent chunks, if present -they often ARE present !!- are not considered in the created sub-set, as they can be harmful to use when training Neural Networks.
    Together with the .wav Audio files created at the specified path, a .csv file is created, containing each file's tags and its name, as well as a copy of the configuration dictionary used to create the dataset.

- ## Synthetic_to_real_unsupervised_Domain_Adaptation
    Software dedicated to the design and training of a Neural Network which, given some unlabelled sound files (of an artificially-generated, synthetic dataset), performs some Synthesis Control Parameters extraction process, as a regression task. Then, two sub-parts of the same pre-trained Neural Network are re-trained in order to perform unsupervised Domain Adaptation from the synthetic dataset to the real dataset. In fact, synthetic-to-real Unsupervised Domain Adaptation consists in adapting a Neural Network from a given data distribution (source domain, the synthetic dataset in my case) to another data distribution (target domain, the real dataset in my case), so that the mappings learned on the former data distribution, can be ported on the latter distribution with minimal or no loss of performance.
    In this case, the pre-trained Neural Network is adapted to the distribution of the real dataset, by using the real/synthetic sound classifier network as a discriminator. The weights of the Convolutional layers are modified so that the real sounds have the same distribution as the synthetic sounds, on which the Fully Connected layers have already tested to be good synthesis control parameters extractors. This networks is used to label the real dataset with the synthesis control parameters' labels.
    The labels of the real sounds dataset will be used to condition the Sound Synthesis process (see next folder).

    The software focuses on the design and training of 3 Neural Networks, all of which are Convolutional-based (the Domain Adaptation part, also called Adversarial Discriminative Domain Adaptation, is inspired by the paper [Adversarial Discriminative Domain Adaptation](https://arxiv.org/pdf/1702.05464.pdf) by Tzeng et al.):
    - A synthesis control parameters extractor, which extracts a number of high-level synthesis control parameters from the synthetic dataset
    - A real/synthetic sound classifier, which classifies the distribution of a sound sample (specifically, its representation in the Flatten layer of the previously-created synthesis control parameters-extractor network mentioned above) as being either synthetic or real
    - A domain adaptation network, which basically starts from the pre-trained Convolutional layers of the synthesis control parameters-extractor network, and adapts those same layers to the distribution of the real dataset, by using the real/synthetic sound classifier network as a discriminator (the weights of the Convolutional layers are modified so that the real sounds have the same distribution as the synthetic sounds, on which the Fully Connected layers have already tested to be good synthesis control parameters extractors). This networks is used to label the real dataset with the Synthesis Control Parameters' labels.

- ## Conditioned Neural Audio Synthesis
    Software dedicated to preparing the previously continuously-labelled real dataset for conditional training of the MTCRNN (multi-tier conditional recurrent neural network) by Lonce Wyse (https://github.com/Metiu-Metiu/MTCRNN).
  First, a numerical analysis of the labels of the real dataset is performed; min/max, mean and std deviation values are calculated for each Synthesis Control parameters. Then, a subset of the labelled real dataset is created separately, taking care of only selecting sound samples whose variables values distribution adds up to a uniform distribution. Specifically, an histogram is created out of the distribution of one variable, and the same number of audio files is selected out of each histogram bin; if not enough samples are available in a particular bin, the bin is discarded from the subset; if more samples are present in a particular bin, only n samples are selected, where n is the selected number (to be as close as possible to the median number of samples for each bin). 
  Second, the real data set is resampled to a target sample rate (default = 16 kHz) and .params files are created for each audio file and relative numeric label. The format of the .params file is just a text file containing JSON-like dictionary like the following;
  
  {
     "meta": {
         "filename": "RegularPopRandomPitch_r02.00.wav"
     },
     "parameter_name": {
         "times": [
             0,
             10
         ],
         "values": [
             0.0,
             0.0
         ],
         "units": "norm",
         "nvals": 11,
         "minval": 0,
         "maxval": 1,
     },
  }
  
  Please refer to the (https://github.com/Metiu-Metiu/MTCRNN) README for further details.

  A 10 seconds Audio file generated by the MTCRNN model trained on waterflow shower sounds and conditioned on the 'expRadius' variable, can be found at https://github.com/Metiu-Metiu/Neural-Texture-Sound-synthesis---trained-Neural-Networks/tree/main/2D_CNN_SynthParamExtractor_June26_2023_Batch128_NoDropouts_10000Dataset_32kHz_3FCLayers_4ConvFilters_IncreasedNumberOfChannels_BatchNorm_DBScale/Generated%20Audio%20files. The 'expRadius' variable increases linearly from 0. to 1..

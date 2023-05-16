# Creation_of_synthetic_Audio_datasets

This folder is dedicated to software designed to create Synthetic Audio Datasets with annotated synthesis control parameters, which can be used as ground truth in supervised Machine Learning applications.

Specifically, <b>this project takes care of generating software that controls external 3rd parties Procedural Audio synthesis engines, without generating sounds itself; it rather focuses on managing probability distributions of synthesis control parameters values</b>. 

3rd parties synthesis control engines are only slightly changed in order to receive OSC messages from this project's python script.

After the HOW TO USE section, you can find a brief description of the scripts,  patches and folders contained in this folder.

## <strong>HOW TO USE</strong>

In order to generate a synthetic Audio dataset, you need to run the script Creation_of_synthetic_Audio_datasets.py, which will control a Max/PD patch via OSC messages. 

Max_8_OSC_receiver.maxpat is provided to receive OSC messages from the script Creation_of_synthetic_Audio_datasets.py and dispatch messages to whatever other Max patch (obtained from 3rd parties, not produced in this Project) containing the actual Audio synthesis engine. An example of external Audio synthesis engine is provided in the SDT_v2.2-078 (Sound Design Toolkit) folder (see the corresponding section).

You can set some global settings for the generated dataset (e.g. number of audio files to be generated, audio files duration, path to store the files into, files names, etc.), as well as the specific synth contr param variables (e.g. ranges and distribution), in the datasetGenerator_DescriptorDict dictionary (which will be dumped in a .json file for future reference).

### <b>Dataset generation</b>

Open Creation_of_synthetic_Audio_datasets.py, Max_8_OSC_receiver.maxpat and the Max patch containing the actual Audio synthesis engine (you can slightly modify it in order to receive the correct synth contr param with the correct names).

In Max_8_OSC_receiver.maxpat, set udpreceive argument (port n.) to whatever variable you set in Creation_of_synthetic_Audio_datasets.py -> oscComm_PyToMaxPD_PortNumber (default = 8000). 

To speed up the Dataset generation process, you can set the Max/PD patch to generate audio files in a faster-than-realtime fashion (e.g. in MacOS, Max 8, go to Options -> Audio Status -> Driver -> NonRealTime).

## <strong>Creation_of_synthetic_Audio_datasets.py</strong>

This script allows you to generate a synthetic Audio dataset, by controlling a Max/PD patch via OSC messages.

The Max/PD patch is a synthesiser, which -in the context of this Project- takes as input a set of synthesis control parameters (synth contr param), and outputs an audio file.

The synth contr param, which in a Procedural Audio context represent physically-driven variables (e.g. mass, stiffness in a membrane percussion sound), are controlled via OSC messages sent from this script to the Max/PD patch.
All synth contr param values for all Audio files -usable as ground truth for Machine Learning models- will also be stored in a separate .csv file.

All synth contr param values are normalized between 0. and 1. in this script (again, useful if used as ground truth in ML models),and then mapped to the expected ranges -settable in this scripts' dictionary- in the Max/PD patch.

You can set some global settings for the generated dataset (e.g. number of audio files to be generated, audio files duration, path to store the files into, files names, etc.), as well as the specific synth contr param variables (e.g. ranges and distribution), in the datasetGenerator_DescriptorDict dictionary (which will be dumped in a .json file for future reference).

Specifically, defining the Distribution_Of_Values_For_Each_Synthesis_Control_Parameter enum data structure,
you can control how the synth contr param values are distributed across the generated dataset.

You can only set one unique distribution type for the entire dataset, which is valid for all the marginal distributions
(the marginal distributions are the distributions of each of the individual variables, a.k.a. each of the individual synth contr param).

### Probability theory concepts: summary of <b>Joint, Marginal, and Conditional Probability Distributions</b> (https://en.wikipedia.org/wiki/Joint_probability_distribution )

Given two random variables that are defined on the same probability space, the joint probability distribution is the corresponding probability distribution
on all possible pairs of outputs. The joint distribution can just as well be considered for any given number of random variables.
The joint distribution encodes the marginal distributions, i.e. the distributions of each of the individual random variables.
It also encodes the conditional probability distributions, which deal with how the outputs of one random variable are distributed
when given information on the outputs of the other random variable(s).

### <strong>Distribution_Of_Values_For_Each_Synthesis_Control_Parameter enum</strong>

This enum represents the possible settings for the marginal distributions, i.e. the distributions of each of the individual variables (in this case, the synthesis control parameters). 

The marginal distributions determine the joint and conditional distributions as well, but only in UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION you can directly control the marginal and joint distributions
 Set the enum value for the marginal distributions in datasetGenerator_DescriptorDict['Dataset_General_Settings']['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'].

 Below you find a section with a brief description of each enum value.

##### <strong>RANDOM_UNIFORM</strong>

<b>RANDOM_UNIFORM involves non-deterministic (except for the seed number, made for reproducibility) stochastic processes yields a marginal uniform distribution, for each synth contr param, as the number of audio files to be generated approaches infinity.</b>

For each audio file, for each synth contr param, 2 stochastic processes are involved.

        - a binary choice is randomly taken -with random.choice()-, to decide whether to generate a new synth contr param value or to re-use the same one used in the previous file
        - if a new synth contr param value has to be generated, a value is generated randomly -with random.uniform()- within the given numerical ranges

##### <strong>UNIFORM_LINEARLY_SPACED_VALUES</strong>

<b>In UNIFORM_LINEARLY_SPACED_VALUES, marginal distributions are guaranteed to be uniform, but the joint distribution does not take into account all combinations of outputs between synth contr param variables.</b>

In order to generate synth contr param values, no stochastic process is involved:

        - n linearly spaced values are generated (min and max values included) for each synth contr param, with n = number of audio files to be generated,
        - so, for each synth contr param, there is 1 different value for each audio file and thus none of the values is repeated more than once across the entire dataset
  
The only stochastic process involved is the choice of which value to use for each audio file (in other words, a random choice is made for choosing the order of values with respect to the series of files).

For each synth contr param, a list of linearly spaced values is created (the given numerical ranges are included), producing 1 different synth contr param value for each audio file to be generated.

For each audio file, for each synth contr param, only 1 stochastic process is involved; a synth contr param is randomly chosen -with random.choice()- from the corresponding pool of possible values, and then that value is deleted from the pool of future possible values.

##### <strong>UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION</strong>

<b>In UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION, both marginal and joint distributions are guaranteed to be uniform.

The joint distribution is guaranteed to take into account all combinations of outputs between synth contr param variables.

Priority is given to your choice of deciding arbitrary ratios between the number of unique values for each synth contr param, useful when a synth contr param needs to have different variance than others.

Unique values are a set of non-repeated values for each synth contr param, which will be repeated in the joint distribution as many time as needed so that every possible combinatorial match between variables outputs is covered.

This is why the prompted number of audio files to be generated is not necessarily respected, since it is computed automatically and it is equal to the product of the number of unique values for each synth contr param (which is computed automatically as well, by respecting the ratios between the number of unique values for each synth contr param you prompted).</b>

In datasetGenerator_DescriptorDict['Synthesis_Control_Parameters_Settings']['Synthesis_Control_Parameters'][YOUR_PARAM_NAME]['number_Of_Minimum_Unique_SynthContrParam_Values'], you can set the minimum number of unique values for each synth contr param, which will be used to compute the ratios between the number of unique values for each synth contr param (they will be respectively multiplied by an incremental int number, and the product of the enlarged numbers is checked against the prompted number of audio files to be generated (x): this process terminates when the closest possible match to x -with the prompted ratios- is reached).

No stochastic process at all is involved in the generation of the synth contr param values.

        - the number of unique values for each synth contr param is computed automatically by respecting the prompted ratios between the number of unique values for each synth contr param
        # you are asked to confirm the resultant number of audio files that will be generated (equal to all the possible combinations of unique variables values)
        - if you decide to proceed, for each synth contr param, a list of n linearly spaced values is created (with given numerical ranges included), where n is the computed number of unique values for that synth contr param
        - then, all combinations of unique values for all synth contr param are computed, and for each audio file, a combination of unique values is chosen

## <strong>Max_8_OSC_receiver.maxpat</strong>

You can use this Max patch to receive OSC message from the script Creation_of_synthetic_Audio_datasets.py and dispatch messages to another Max patch containing the actual Audio synthesis engine.

## <strong>Py_OSC_control_of_Max_PD_patches folder</strong>

Py_OSC_control_of_Max_PD_patches is a folder with a python3 environment (created with venv) and the necessary dependencies installed (mostly PythonOSC, see requirements.txt), which are listed in the file 'requirements.txt'.

## <strong>SDT_v2.2-078 (Sound Design Toolkit) folder</strong>

SDT_v2.2-078 (Sound Design Toolkit) is a folder containing the Sound Design Toolkit (https://github.com/SkAT-VG/SDT). The following list shows the Max patches used (and slightly modified) and the corresponding type of sounds generated:

	- std.fluidflow~.maxhelp -> water streaming sounds
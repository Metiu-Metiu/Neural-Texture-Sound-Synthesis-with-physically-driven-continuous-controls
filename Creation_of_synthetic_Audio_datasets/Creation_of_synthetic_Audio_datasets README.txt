{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Bold;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red70\green137\blue204;\red24\green24\blue24;\red193\green193\blue193;
\red109\green109\blue109;\red194\green126\blue101;}
{\*\expandedcolortbl;;\cssrgb\c33725\c61176\c83922;\cssrgb\c12157\c12157\c12157;\cssrgb\c80000\c80000\c80000;
\cssrgb\c50196\c50196\c50196;\cssrgb\c80784\c56863\c47059;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl360\partightenfactor0

\f0\b\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # Creation_of_synthetic_Audio_datasets
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 This folder is dedicated to software designed to create Synthetic Audio Datasets with annotated synthesis control parameters, which can be used as ground truth in supervised Machine Learning applications.\cb1 \
\
\cb3 Specifically, \cf5 \strokec5 <\cf2 \strokec2 b\cf5 \strokec5 >\cf4 \strokec4 this project takes care of generating software that controls external 3rd parties Procedural Audio synthesis engines, without generating sounds itself; it rather focuses on managing probability distributions of synthesis control parameters values\cf5 \strokec5 </\cf2 \strokec2 b\cf5 \strokec5 >\cf4 \strokec4 . \cb1 \
\
\cb3 3rd parties synthesis control engines are only slightly changed in order to receive OSC messages from this project's python script.\cb1 \
\
\cb3 After the HOW TO USE section, you can find a brief description of the scripts,  patches and folders contained in this folder.\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 HOW TO USE\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 In order to generate a synthetic Audio dataset, you need to run the script Creation_of_synthetic_Audio_datasets.py, which will control a Max/PD patch via OSC messages. \cb1 \
\
\cb3 Max_8_OSC_receiver.maxpat is provided to receive OSC messages from the script Creation_of_synthetic_Audio_datasets.py and dispatch messages to whatever other Max patch (obtained from 3rd parties, not produced in this Project) containing the actual Audio synthesis engine. An example of external Audio synthesis engine is provided in the SDT_v2.2-078 (Sound Design Toolkit) folder (see the corresponding section).\cb1 \
\
\cb3 You can set some global settings for the generated dataset (e.g. number of audio files to be generated, audio files duration, path to store the files into, files names, etc.), as well as the specific synth contr param variables (e.g. ranges and distribution), in the datasetGenerator_DescriptorDict dictionary (which will be dumped in a .json file for future reference).\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ### \cf5 \strokec5 <\cf2 \strokec2 b\cf5 \strokec5 >\cf2 \strokec2 Dataset generation\cf5 \strokec5 </\cf2 \strokec2 b\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 Open Creation_of_synthetic_Audio_datasets.py, Max_8_OSC_receiver.maxpat and the Max patch containing the actual Audio synthesis engine (you can slightly modify it in order to receive the correct synth contr param with the correct names).\cb1 \
\
\cb3 In Max_8_OSC_receiver.maxpat, set udpreceive argument (port n.) to whatever variable you set in Creation_of_synthetic_Audio_datasets.py -> oscComm_PyToMaxPD_PortNumber (default = 8000). \cb1 \
\
\cb3 To speed up the Dataset generation process, you can set the Max/PD patch to generate audio files in a faster-than-realtime fashion (e.g. in MacOS, Max 8, go to Options -> Audio Status -> Driver -> NonRealTime).\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 Creation_of_synthetic_Audio_datasets.py\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 This script allows you to generate a synthetic Audio dataset, by controlling a Max/PD patch via OSC messages.\cb1 \
\
\cb3 The Max/PD patch is a synthesiser, which -in the context of this Project- takes as input a set of synthesis control parameters (synth contr param), and outputs an audio file.\cb1 \
\
\cb3 The synth contr param, which in a Procedural Audio context represent physically-driven variables (e.g. mass, stiffness in a membrane percussion sound), are controlled via OSC messages sent from this script to the Max/PD patch.\cb1 \
\cb3 All synth contr param values for all Audio files -usable as ground truth for Machine Learning models- will also be stored in a separate .csv file.\cb1 \
\
\cb3 All synth contr param values are normalized between 0. and 1. in this script (again, useful if used as ground truth in ML models),and then mapped to the expected ranges -settable in this scripts' dictionary- in the Max/PD patch.\cb1 \
\
\cb3 You can set some global settings for the generated dataset (e.g. number of audio files to be generated, audio files duration, path to store the files into, files names, etc.), as well as the specific synth contr param variables (e.g. ranges and distribution), in the datasetGenerator_DescriptorDict dictionary (which will be dumped in a .json file for future reference).\cb1 \
\
\cb3 Specifically, defining the Distribution_Of_Values_For_Each_Synthesis_Control_Parameter enum data structure,\cb1 \
\cb3 you can control how the synth contr param values are distributed across the generated dataset.\cb1 \
\
\cb3 You can only set one unique distribution type for the entire dataset, which is valid for all the marginal distributions\cb1 \
\cb3 (the marginal distributions are the distributions of each of the individual variables, a.k.a. each of the individual synth contr param).\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ### Probability theory concepts: summary of \cf5 \strokec5 <\cf2 \strokec2 b\cf5 \strokec5 >\cf2 \strokec2 Joint, Marginal, and Conditional Probability Distributions\cf5 \strokec5 </\cf2 \strokec2 b\cf5 \strokec5 >\cf2 \strokec2  (https://en.wikipedia.org/wiki/Joint_probability_distribution )
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 Given two random variables that are defined on the same probability space, the joint probability distribution is the corresponding probability distribution\cb1 \
\cb3 on all possible pairs of outputs. The joint distribution can just as well be considered for any given number of random variables.\cb1 \
\cb3 The joint distribution encodes the marginal distributions, i.e. the distributions of each of the individual random variables.\cb1 \
\cb3 It also encodes the conditional probability distributions, which deal with how the outputs of one random variable are distributed\cb1 \
\cb3 when given information on the outputs of the other random variable(s).\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ### \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 Distribution_Of_Values_For_Each_Synthesis_Control_Parameter enum\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 This enum represents the possible settings for the marginal distributions, i.e. the distributions of each of the individual variables (in this case, the synthesis control parameters). \cb1 \
\
\cb3 The marginal distributions determine the joint and conditional distributions as well, but only in UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION you can directly control the marginal and joint distributions\cb1 \
\cb3  Set the enum value for the marginal distributions in datasetGenerator_DescriptorDict[\cf6 \strokec6 'Dataset_General_Settings'\cf4 \strokec4 ]['distribution_Of_Values_For_Each_Synthesis_Control_Parameter'].\cb1 \
\
\cb3  Below you find a section with a brief description of each enum value.\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ##### \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 RANDOM_UNIFORM\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf5 \cb3 \strokec5 <\cf2 \strokec2 b\cf5 \strokec5 >\cf4 \strokec4 RANDOM_UNIFORM involves non-deterministic (except for the seed number, made for reproducibility) stochastic processes yields a marginal uniform distribution, for each synth contr param, as the number of audio files to be generated approaches infinity.\cf5 \strokec5 </\cf2 \strokec2 b\cf5 \strokec5 >\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 For each audio file, for each synth contr param, 2 stochastic processes are involved.\cb1 \
\
\cb3         - a binary choice is randomly taken -with random.choice()-, to decide whether to generate a new synth contr param value or to re-use the same one used in the previous file\cb1 \
\cb3         - if a new synth contr param value has to be generated, a value is generated randomly -with random.uniform()- within the given numerical ranges\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ##### \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 UNIFORM_LINEARLY_SPACED_VALUES\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf5 \cb3 \strokec5 <\cf2 \strokec2 b\cf5 \strokec5 >\cf4 \strokec4 In UNIFORM_LINEARLY_SPACED_VALUES, marginal distributions are guaranteed to be uniform, but the joint distribution does not take into account all combinations of outputs between synth contr param variables.\cf5 \strokec5 </\cf2 \strokec2 b\cf5 \strokec5 >\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 In order to generate synth contr param values, no stochastic process is involved:\cb1 \
\
\cb3         - n linearly spaced values are generated (min and max values included) for each synth contr param, with n = number of audio files to be generated,\cb1 \
\cb3         - so, for each synth contr param, there is 1 different value for each audio file and thus none of the values is repeated more than once across the entire dataset\cb1 \
\cb3   \cb1 \
\cb3 The only stochastic process involved is the choice of which value to use for each audio file (in other words, a random choice is made for choosing the order of values with respect to the series of files).\cb1 \
\
\cb3 For each synth contr param, a list of linearly spaced values is created (the given numerical ranges are included), producing 1 different synth contr param value for each audio file to be generated.\cb1 \
\
\cb3 For each audio file, for each synth contr param, only 1 stochastic process is involved; a synth contr param is randomly chosen -with random.choice()- from the corresponding pool of possible values, and then that value is deleted from the pool of future possible values.\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ##### \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf5 \cb3 \strokec5 <\cf2 \strokec2 b\cf5 \strokec5 >\cf4 \strokec4 In UNIFORM_CONTROLLABLE_VARIANCE_LINEARLY_SPACED_VALUES_UNIFORM_JOINT_DISTRIBUTION, both marginal and joint distributions are guaranteed to be uniform.\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 The joint distribution is guaranteed to take into account all combinations of outputs between synth contr param variables.\cb1 \
\
\cb3 Priority is given to your choice of deciding arbitrary ratios between the number of unique values for each synth contr param, useful when a synth contr param needs to have different variance than others.\cb1 \
\
\cb3 Unique values are a set of non-repeated values for each synth contr param, which will be repeated in the joint distribution as many time as needed so that every possible combinatorial match between variables outputs is covered.\cb1 \
\
\cb3 This is why the prompted number of audio files to be generated is not necessarily respected, since it is computed automatically and it is equal to the product of the number of unique values for each synth contr param (which is computed automatically as well, by respecting the ratios between the number of unique values for each synth contr param you prompted).\cf5 \strokec5 </\cf2 \strokec2 b\cf5 \strokec5 >\cf4 \cb1 \strokec4 \
\
\cb3 In datasetGenerator_DescriptorDict[\cf6 \strokec6 'Synthesis_Control_Parameters_Settings'\cf4 \strokec4 ]['Synthesis_Control_Parameters'][YOUR_PARAM_NAME]['number_Of_Minimum_Unique_SynthContrParam_Values'], you can set the minimum number of unique values for each synth contr param, which will be used to compute the ratios between the number of unique values for each synth contr param (they will be respectively multiplied by an incremental int number, and the product of the enlarged numbers is checked against the prompted number of audio files to be generated (x): this process terminates when the closest possible match to x -with the prompted ratios- is reached).\cb1 \
\
\cb3 No stochastic process at all is involved in the generation of the synth contr param values.\cb1 \
\
\cb3         - the number of unique values for each synth contr param is computed automatically by respecting the prompted ratios between the number of unique values for each synth contr param\cb1 \
\cb3         # you are asked to confirm the resultant number of audio files that will be generated (equal to all the possible combinations of unique variables values)\cb1 \
\cb3         - if you decide to proceed, for each synth contr param, a list of n linearly spaced values is created (with given numerical ranges included), where n is the computed number of unique values for that synth contr param\cb1 \
\cb3         - then, all combinations of unique values for all synth contr param are computed, and for each audio file, a combination of unique values is chosen\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 Max_8_OSC_receiver.maxpat\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 You can use this Max patch to receive OSC message from the script Creation_of_synthetic_Audio_datasets.py and dispatch messages to another Max patch containing the actual Audio synthesis engine.\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 Py_OSC_control_of_Max_PD_patches folder\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 Py_OSC_control_of_Max_PD_patches is a folder with a python3 environment (created with venv) and the necessary dependencies installed (mostly PythonOSC, see requirements.txt), which are listed in the file 'requirements.txt'.\cb1 \
\
\pard\pardeftab720\sl360\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## \cf5 \strokec5 <\cf2 \strokec2 strong\cf5 \strokec5 >\cf2 \strokec2 SDT_v2.2-078 (Sound Design Toolkit) folder\cf5 \strokec5 </\cf2 \strokec2 strong\cf5 \strokec5 >
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\sl360\partightenfactor0
\cf4 \cb3 SDT_v2.2-078 (Sound Design Toolkit) is a folder containing the Sound Design Toolkit (https://github.com/SkAT-VG/SDT). The following list shows the Max patches used (and slightly modified) and the corresponding type of sounds generated:\cb1 \
\
\cb3         - std.fluidflow~.maxhelp -> water streaming sounds\cb1 \
}
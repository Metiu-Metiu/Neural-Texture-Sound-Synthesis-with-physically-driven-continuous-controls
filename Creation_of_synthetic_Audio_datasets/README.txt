Py_OSC_control_of_Max_PD_patches is a folder with a python3 environment (created with venv) and the necessary dependencies installed (mostly PythonOSC, see requirements.txt).

SDT_v2.2-078 (Sound Design Toolkit) is a folder containing the Sound Design Toolkit (https://github.com/SkAT-VG/SDT). The following list shows the Max patches used (and slightly modified) and the corresponding type of sounds generated:
	std.fluidflow~ -> water streaming sounds

SDT_FluidFlow_dataset is the folder where the synthetic dataset files are created, together with the .csv files containing the synthesis control parameters used to generate them. Any pre-existing files with the same name will be truncated and overwritten.

######################### How to use #########################
To generate the water streaming sounds dataset:
	In Max_8_OSC_receiver.maxpat, set udpreceive argument (port n.) to whatever variable you set in Py_OSC_control_of_Max_PD_patches.py -> oscComm_PyToMaxPD_PortNumber (default = 8000). 


######################### Py_OSC_control_of_Max_PD_patches.py #########################
Known issues:
	Max 8 can not work with sample rate < 44100 (?)
	No quantisation (bit resolution) is specified

Possible errors in Py_OSC_control_of_Max_PD_patches.py and solutions:
OSError: [Errno 48] Address already in use
	ps -fA | grep python
	Kill (second number)
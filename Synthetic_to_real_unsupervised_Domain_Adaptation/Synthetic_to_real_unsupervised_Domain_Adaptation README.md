# Synthetic_to_real_unsupervised_Domain_Adaptation



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
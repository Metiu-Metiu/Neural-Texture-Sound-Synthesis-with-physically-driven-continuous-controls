import soundata

print(f'    {soundata.list_datasets()}')

fsd50k_dataset = soundata.initialize('fsd50k', data_home = '/Users/matthew/Desktop/UPF/Courses/Master thesis project (Frederic Font)/Lonce Wyse - Data-Driven Neural Sound Synthesis/Software/datasets/FSD50K')
# fsd50k_dataset.download()  # download the dataset
fsd50k_dataset.validate()  # validate that all the expected files are there
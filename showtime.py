
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
import os
import time
import nibabel as nib
from pathlib import PosixPath

from deeplearning_code_files.train import Trainer
from deeplearning_code_files.datautils import MyTrainDataset
from deeplearning_code_files.utils import mri_downsampling,voxel_to_mask
from deeplearning_code_files.model import eegsubnet, mrisubnet, fusion

import numpy as np
import pandas as pd
import statistics
import mne


device = torch.device('cuda:0')


ids =[1,2,3,4,5,6,7,8,9,10,11,12,13]
import random
training_id = [1,2,3,4,5,6,7,8,9,10]#random.sample(ids, 20)
unseen_id = [item for item in ids if item not in training_id]

print(f"training : {training_id}")
print(f"unseen : {unseen_id}")


training_data = MyTrainDataset(mri_id=training_id, 
                                outputtype='peak_ras', 
                                mri_dir=PosixPath('/mnt/d/mris_MJ'),
                                eeg_subject_dir=PosixPath('/mnt/d/REAL_FINAL_MJ/montage_standard_1020/amplitude_1/single dipole/snr_db10'),
                                eegtype='raw', 
                                DTYPE=torch.float32, 
                                mri_n_downsampling=1, 
                                eeg_per_mri=2000,
                                eeg_filter=0,
                                less_noise=False,
                                ch_names =mne.channels.make_standard_montage("biosemi32").ch_names)
    

unseen_data = MyTrainDataset(mri_id=unseen_id, 
                                outputtype='peak_ras', 
                                mri_dir=PosixPath('/mnt/d/mris_MJ'),
                                eeg_subject_dir=PosixPath('/mnt/d/REAL_FINAL_MJ/montage_standard_1020/amplitude_1/single dipole/snr_db10'),
                                eegtype='raw', 
                                DTYPE=torch.float32, 
                                mri_n_downsampling=1, 
                                eeg_per_mri=2000,
                                eeg_filter=0,
                                less_noise=False,
                                ch_names =mne.channels.make_standard_montage("biosemi32").ch_names)




# Define split ratios
train_rate = 0.8  # 80% for train
val_rate = 0.1    # 10% for validation
test_rate = 0.1   # 10% for test

# Ensure the split ratios sum up to 1.0
assert train_rate + val_rate + test_rate == 1.0, "Split ratios must sum up to 1.0"

# Calculate dataset sizes
train_size = int(train_rate * len(training_data))
val_size = int(val_rate * len(training_data))
test_size = len(training_data) - train_size - val_size  # Remaining for test

# Perform random split
train_dataset, val_dataset, test_dataset = random_split(training_data, [train_size, val_size, test_size])

# Define batch size
batch_size = 16

# Create DataLoaders for each split
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # Shuffle False for validation
seen_test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # Shuffle False for test

unseen_test_loader = DataLoader(dataset=unseen_data, batch_size=batch_size, shuffle=False, drop_last=True)  # Shuffle False for test





#When you want to see some data
def check_your_data(train_loader):
    for i, mri, eeg, dipoles in train_loader:
        i, checkmri,checkeeg,checksource = i, mri, eeg, dipoles
        break
    return i, checkmri, checkeeg, checksource

identity,mri,eeg,source = check_your_data(train_loader)


mid_dim = 100
mri_conv_dims = [1,1,1,1]
mri_mlp_dims=[2744,2744,2744,mid_dim,mid_dim]
eeg_conv_dims=[32,32,32,16,16]
eeg_mlp_dims=[3072,500,mid_dim,mid_dim]
fusion_dims = [2*mid_dim,mid_dim,3]

torch.cuda.empty_cache()



"""
model = eegsubnet(conv_dims=eeg_conv_dims, 
               mlp_dims=eeg_mlp_dims,
               conv_kernel_size=3, 
               dropout=0,
               ).to(torch.float32).to(device)

"""
#I just want to use pre-saved model.
#checkpoint_path = "checkpoint/cuda:0checkpoint.pt"
#checkpoint = torch.load(checkpoint_path)
#model.load_state_dict(checkpoint)


model = fusion(mri_conv_dims=mri_conv_dims,
    mri_mlp_dims=mri_mlp_dims,
    mri_conv_kernel_size=(3,3,3), 
    eeg_conv_dims=eeg_conv_dims, 
    eeg_mlp_dims=eeg_mlp_dims,
    eeg_conv_kernel_size=3,
    fusion_conv_dims=[1],
    fusion_dims=fusion_dims, 
    fusion_conv_kernel_size=3, 
    dropout=[0,0,0],
    batch_size=None, 
    output_as_3d=False
    ).to(torch.float32)



trainer1 = Trainer(model=model,
                   train_data=train_loader,
                   validation_data=val_loader,
                   optimizer= torch.optim.Adam(model.parameters()),
                   gpu_id=device,
                   save_energy=50,
                   loss_ft= nn.L1Loss(),
                   model_mode = 'mri+eeg',
                   autocast=False,
                   saving_dir_path='/home/pheeeeee/neuroimaging/results_amplitude1/mri+eeg/downsampling1/L1loss'
                   )


trainer1.train(1000)

#test on seen mri data
trainer1.test(seen_test_loader,loss_ft= nn.L1Loss(), seen=True)

#test on unseen mri data
trainer1.test(unseen_test_loader, loss_ft=nn.L1Loss())

trainer1.save_results(tested_on_seen_mri=True)

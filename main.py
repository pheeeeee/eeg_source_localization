
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
from deeplearning_code_files.datautils import MyTrainDataset, MyTrainDataset1
from deeplearning_code_files.utils import mri_downsampling,voxel_to_mask
from deeplearning_code_files.model import eegsubnet, sensorsubnet, mrisubnet, fusion

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
import mne
import random

import argparse



def check_and_create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' has been created.")
        #response = input(f"The directory '{dir_name}' does not exist. Do you want to create it? (yes/no): ").strip().lower()
        #if response == 'yes':
        #    os.makedirs(dir_name)
        #    print(f"Directory '{dir_name}' has been created.")
        #else:
        #    print("No directory was created.")
    else:
        print(f"The directory '{dir_name}' already exists.")


def main():
    parser = argparse.ArgumentParser(description="실험 세팅(device, snr, eegtype, amplitude, training_id, ch_names, eeg_time_window, task, loss_ft, early_stop, epoch)을 설정하시오.")
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--snr", type=int, required=True)
    parser.add_argument("--eegtype", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--loss_ft", type=str ,required=True)
    parser.add_argument("--amplitude", type=str, default='amplitude0')
    parser.add_argument("--training_id", type=list, default=[1,2,3,4,5,6,7,8,9,10])
    parser.add_argument("--ch_names", type=str, default='biosemi32' )
    parser.add_argument("--eeg_time_window_portion", type=float, default = 1.0)
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=500)

    args = parser.parse_args()

    device = torch.device(args.device)
    snr = args.snr
    assert snr in [1,5,10,30], "snr must be one of 1,5,10,30."
    snr = f'snr_db{snr}'
    eegtype = args.eegtype
    assert eegtype in ['raw', 'fourier sin' , 'fourier cos' , 'fourier concatenate']
    amplitude = args.amplitude
    assert amplitude in ['amplitude0', 'amplitude1']

    ids =[1,2,3,4,5,6,7,8,9,10,11,12,13]
    training_id = args.training_id
    unseen_id = [item for item in ids if item not in training_id]

    ch_names1 = args.ch_names
    ch_names = mne.channels.make_standard_montage(ch_names1).ch_names
    eeg_time_window_portion = args.eeg_time_window_portion
    center_len = int(100 * eeg_time_window_portion)
    start = (100 - center_len) // 2
    end = start + center_len
    eeg_time_window = 2*np.arange(100)[start:end]


    task = args.task
    assert task in ['eeg', 'mri+eeg'], "task must be either 'eeg' or 'mri+eeg'. "
    
    loss_ft = args.loss_ft
    assert loss_ft in ['L1', 'MSE'], 'loss_ft must be either "L1" or "MSE"'
    if loss_ft == "L1":
        LOSSFT = nn.L1Loss() 
    else:
        LOSSFT = nn.MSELoss()
    
    early_stop = args.early_stop #if 0, no early_stopping. if some positive integer, it becomes the patience number.
    epoch = args.epoch


    sensor = False
    sinpeak = 1
    n_dipole = 'singledipole'

    dataname = 'new_MJ_data'

    training_data = MyTrainDataset(mri_id=training_id, 
                                    outputtype='center_ras', 
                                    mri_dir=PosixPath('/home/user/data/pheeeeee/mris_MJ'),
                                    eeg_subject_dir=PosixPath(f'/home/user/data/pheeeeee/REAL_FINAL_MJ/montage_standard_1020/amplitude_{amplitude[-1]}/single dipole/{snr}'),
                                    n_dipole = n_dipole,
                                    amplitude = amplitude,
                                    snr = snr,
                                    eegtype=eegtype,
                                    DTYPE=torch.float32, 
                                    mri_n_downsampling=1, 
                                    sensor = sensor,
                                    eeg_per_mri=2000,
                                    eeg_filter=0,
                                    eeg_time_window=eeg_time_window,
                                    ch_names = ch_names,
                                    all_dipole=False,
                                    output_config = False)
        

    unseen_data = MyTrainDataset(mri_id=unseen_id, 
                                    outputtype='center_ras',#'center_ras', 
                                    mri_dir=PosixPath('/home/user/data/pheeeeee/mris_MJ'),
                                    eeg_subject_dir=PosixPath(f'/home/user/data/pheeeeee/REAL_FINAL_MJ/montage_standard_1020/amplitude_{amplitude[-1]}/single dipole/{snr}'),
                                    n_dipole = n_dipole,
                                    amplitude = amplitude,
                                    snr = snr,
                                    eegtype=eegtype, 
                                    DTYPE=torch.float32, 
                                    mri_n_downsampling=1,
                                    sensor = sensor,
                                    eeg_per_mri=2000,
                                    eeg_filter=0,
                                    eeg_time_window=eeg_time_window,
                                    ch_names = ch_names,
                                    all_dipole=False,
                                    output_config = False)

    # Define split ratios
    train_rate = 0.8  # 70% for train
    val_rate = 0.1    # 10% for validation
    #conformal_rate = 0.1 #10% for conformal learning
    test_rate = 0.1   # 10% for test

    # Calculate dataset sizes
    train_size = int(train_rate * len(training_data))
    val_size = int(val_rate * len(training_data))
    #conformal_size = int(conformal_rate * len(training_data))
    test_size = len(training_data) - train_size - val_size #- conformal_size # Remaining for test

    # Perform random split
    train_dataset, val_dataset, test_dataset = random_split(training_data, [train_size, val_size, test_size])

    # Define batch size
    batch_size = 16

    # Create DataLoaders for each split
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # Shuffle False for validation
    #conformal_loader= DataLoader(dataset=conformal_dataset, batch_size = 1)
    seen_test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # Shuffle False for test

    unseen_test_loader = DataLoader(dataset=unseen_data, batch_size=batch_size, shuffle=False, drop_last=True)  # Shuffle False for test

    if eegtype == 'fourier concatenate':
        task1 = task + '_fourier_concatenate'
    elif eegtype == 'fourier cos':
        task1 = task + '_fourier_cos'
    elif eegtype == 'fourier sin':
        task1 = task+ '_fourier_sin'
    else:
        task1 = task

    mid_dim = 100
    mri_conv_dims = [1,1,1,1]
    if dataname == 'new_openneuro_data':
        mri_mlp_dims=[1960,500,mid_dim,mid_dim]
    elif dataname == 'new_MJ_data':
        mri_mlp_dims=[2744,500,mid_dim,mid_dim]

    if ch_names1 == 'biosemi32':
        eeg_conv_dims=[32,32,32,32,32]
    elif ch_names1 == 'biosemi16':
        eeg_conv_dims = [16,16,16,16,16]
    elif ch_names1 == 'biosemi64':
        eeg_conv_dims = [64,64,64,64,64]

    if eegtype == 'fourier concatenate':
        if eeg_time_window_portion == 1:
            aaa = 6144
        elif eeg_time_window_portion == 0.8:
            aaa = 4864
        elif eeg_time_window_portion == 0.5:
            aaa = 2944
        elif eeg_time_window_portion == 0.2:
            aaa = 1024
        if ch_names1 == 'biosemi16':
            aaa = int(aaa/2)
        elif ch_names1 == 'biosemi64':
            aaa = 2*aaa
        eeg_mlp_dims=[aaa,5*mid_dim,mid_dim,mid_dim]
    elif eegtype == 'raw':
        if eeg_time_window_portion == 1:
            aaa = 2944
        elif eeg_time_window_portion == 0.8:
            aaa = 2304
        elif eeg_time_window_portion == 0.5:
            aaa = 1344
        elif eeg_time_window_portion == 0.2:
            aaa = 384
        if ch_names1 == 'biosemi16':
            aaa = int(aaa/2)
        elif ch_names1 == 'biosemi64':
            aaa = 2*aaa
        eeg_mlp_dims=[aaa,5*mid_dim,mid_dim,mid_dim]

    sensor_dims = [96, 100, 100]

    if task[:3] =='mri':
        fusion_dims = [2*mid_dim,mid_dim,3]
    elif task[:3] == 'eeg':
        eeg_mlp_dims = eeg_mlp_dims + [mid_dim,mid_dim,3]
    elif task[:3] == 'sen':
        fusion_dims = [3*mid_dim, mid_dim, mid_dim, 3]
    torch.cuda.empty_cache()

    if 'mri' in task:
        mriexist = True
    else:
        mriexist = False

    if 'sensor' in task:
        sensor = True
    else:
        sensor = False    

    if 'mri' in task:
        model = fusion(mri_conv_dims=mri_conv_dims,
        mri_mlp_dims=mri_mlp_dims,
        mri_conv_kernel_size=(3,3,3), 
        eeg_conv_dims=eeg_conv_dims, 
        eeg_mlp_dims=eeg_mlp_dims,
        eeg_conv_kernel_size=3,
        sensor_dims = sensor_dims,
        fusion_conv_dims=[1],
        fusion_dims=fusion_dims, 
        fusion_conv_kernel_size=3, 
        dropout=[0,0,0,0],
        batch_size=None, 
        output_as_3d=False,
        MRI=mriexist,                      ########################## Caution!!!!
        sensor=sensor,
        ).to(torch.float32)

    elif task[:3] =='eeg':
        model = eegsubnet(conv_dims=eeg_conv_dims, 
        mlp_dims=eeg_mlp_dims, 
        conv_kernel_size=3, 
        dropout=0).to(torch.float32)

    torch.cuda.empty_cache()
    saving_dir_path=f'/home/user/pheeeeee/neuroimaging/output_configured/{dataname}/results_{amplitude}/ch_names_{ch_names1}/{snr}/{task1}/{str(LOSSFT)[:-2]}' ##Caution!!!mri+eeg file requires downsampling number.
    check_and_create_directory(saving_dir_path)

    trainer1 = Trainer(model=model,
                    train_data=train_loader,
                    validation_data=val_loader,
                    optimizer= torch.optim.Adam(model.parameters()),
                    gpu_id=device,
                    save_energy=50,
                    loss_ft= LOSSFT ,
                    model_mode = task,
                    autocast=False,
                    early_stop=early_stop,
                    saving_dir_path=saving_dir_path
                    )

    trainer1.train(epoch)

    #test on seen mri data
    trainer1.test(seen_test_loader,loss_ft= LOSSFT, seen=True)

    #test on unseen mri data
    trainer1.seentestloss
    trainer1.test(unseen_test_loader, loss_ft=LOSSFT)

    trainer1.save_results(tested_on_seen_mri=True)



    # Plotting the loss
    plt.figure(figsize=(8, 5))
    plt.plot(trainer1.train_loss_traj, label='Train Loss', color='blue', marker='o')
    plt.plot(trainer1.validation_loss_traj, label='Validation Loss', color='red', marker='x')
    plt.axvline(x=trainer1.best_epoch, color='green', linestyle='--', label=f'Best Epoch ({trainer1.best_epoch})')
    plt.title(f'(results_{amplitude},snr{snr},{task1},{str(LOSSFT)[:-2]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{saving_dir_path}/training_plot.png', dpi=300)  # You can also use .pdf or .svg for vector formats
    #plt.show()


    print(f"task : {task1}, ch :{ch_names1} ,snr : {snr}, loss fun : {loss_ft}, early_stopping : {early_stop}")
    #test with the best model on seen mri data
    trainer1.test(seen_test_loader,loss_ft= nn.L1Loss() , seen=True)
    #test on unseen mri data
    trainer1.test(unseen_test_loader, loss_ft= nn.L1Loss() ) 

    #trainer1.seentestloss
    trainer1.save_results(tested_on_seen_mri=True)




    #### Conformal Learning
    model = trainer1.model
    seen_conformal_dataset = MyTrainDataset(mri_id=training_id, 
                                    outputtype='center_ras', 
                                    mri_dir=PosixPath('/home/user/data/pheeeeee/mris_MJ'),
                                    eeg_subject_dir=PosixPath(f'/home/user/data/pheeeeee/REAL_FINAL_MJ/montage_standard_1020/amplitude_{amplitude[-1]}/single dipole/{snr}'),
                                    n_dipole = n_dipole,
                                    amplitude = amplitude,
                                    snr = snr,
                                    eegtype=eegtype,
                                    DTYPE=torch.float32, 
                                    mri_n_downsampling=1, 
                                    sensor = sensor,
                                    eeg_per_mri=200,
                                    eeg_filter=0,
                                    eeg_time_window=eeg_time_window,
                                    ch_names = ch_names,
                                    all_dipole=False,
                                    output_config = False,
                                    conformal = True)

    unseen_conformal_dataset = MyTrainDataset(mri_id=unseen_id, 
                                    outputtype='center_ras', 
                                    mri_dir=PosixPath('/home/user/data/pheeeeee/mris_MJ'),
                                    eeg_subject_dir=PosixPath(f'/home/user/data/pheeeeee/REAL_FINAL_MJ/montage_standard_1020/amplitude_{amplitude[-1]}/single dipole/{snr}'),
                                    n_dipole = n_dipole,
                                    amplitude = amplitude,
                                    snr = snr,
                                    eegtype=eegtype, 
                                    DTYPE=torch.float32, 
                                    mri_n_downsampling=1,
                                    sensor = sensor,
                                    eeg_per_mri=200,
                                    eeg_filter=0,
                                    eeg_time_window=eeg_time_window,
                                    ch_names = ch_names,
                                    all_dipole=False,
                                    output_config = False,
                                    conformal = True)

    seen_conformal_loader= DataLoader(dataset=seen_conformal_dataset, batch_size = 1)
    unseen_conformal_loader= DataLoader(dataset=unseen_conformal_dataset, batch_size = 1)

    model.eval()
    seenconformalloss = []
    unseenconformalloss = []
    for identity, mri, eeg, targets in seen_conformal_loader:
        identity = identity.to(device)
        mri = mri.to(device)
        eeg = eeg.to(device)
        targets = targets.to(device)
        
        if task == 'eeg':
            output = model(eeg)
        elif task == 'mri+eeg':      
            output = model([mri, eeg])
        elif task == 'sensor+mri+eeg':
            output = model([identity, mri, eeg]) 
        
        loss = LOSSFT(output, targets)
        seenconformalloss.append(loss.item())
    
    for identity, mri, eeg, targets in unseen_conformal_loader:
        identity = identity.to(device)
        mri = mri.to(device)
        eeg = eeg.to(device)
        targets = targets.to(device)
        
        if task == 'eeg':
            output = model(eeg)
        elif task == 'mri+eeg':      
            output = model([mri, eeg])
        elif task == 'sensor+mri+eeg':
            output = model([identity, mri, eeg]) 
        
        loss = LOSSFT(output, targets)
        unseenconformalloss.append(loss.item())

    alpha = 0.05
    seenconformalloss = np.array(seenconformalloss)
    sconformal_n = len(seen_conformal_loader)
    q_level = np.ceil((sconformal_n+1)*(1-alpha))/sconformal_n    
    sqhat = np.quantile(seenconformalloss, q_level, method='higher')

    unseenconformalloss = np.array(unseenconformalloss)
    unsconformal_n = len(unseen_conformal_loader)
    q_level = np.ceil((unsconformal_n+1)*(1-alpha))/unsconformal_n    
    unsqhat = np.quantile(unseenconformalloss, q_level, method='higher')

    print(f'Radius of 95% CI for seen MRI : {sqhat}')
    print(f'Radius of 95% CI for unseen MRI : {unsqhat}')



if __name__ == "__main__":
    main()
    

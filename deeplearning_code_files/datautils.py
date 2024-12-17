
import torch
from torch.utils.data import Dataset
import os
from pathlib import PosixPath
from deeplearning_code_files.utils import mri_downsampling,voxel_to_mask,ras_to_voxel
import nibabel as nib
from scipy.fft import fft
from scipy.signal import welch
import numpy as np
import mne
class MyTrainDataset(Dataset):
    def __init__(self, mri_id:list, 
                 outputtype:str, 
                 eegtype='raw', 
                 DTYPE=torch.float32, 
                 mri_n_downsampling:int=0, 
                 eeg_per_mri:int=2000,
                 eeg_filter:int=0,
                 less_noise=False,
                 ch_names = None,
                 ):
        self.mri_id = mri_id
        self.eeg_per_mri = eeg_per_mri
        self.size = len(mri_id) * eeg_per_mri
        self.DTYPE = DTYPE
        
        assert eegtype in ['raw','psd','fourier cos', 'fourier sin', 'fourier concatenate'], "eegtype must be one of 'raw','psd','fourier cos', 'fourier sin'"
        self.eegtype = eegtype
        
        self.ch_names = ch_names
        if ch_names is not None:
            info = PosixPath("/data/pheeeeee/FINAL_EEG_DATA//sub-01/info.fif")
            info = mne.io.read_info(info)
            eeg_channel_index =[]
            for ch in ch_names:
                eeg_channel_index.append(info.ch_names.index(ch))
            eeg_channel_index.sort()
            eeg_channel_names = [info.ch_names[_] for _ in eeg_channel_index]
            self.eeg_channel_names = eeg_channel_names
            self.eeg_channel_index = eeg_channel_index
        
        #load mri data and eeg directory
        #assert inputlocation in ['ras', 'peak_ras','voxel_coordinate', 'mask', 'voxel_coordinate_eeg', 'vertex_number_src_eeg', 'fsaverage_stc_eeg','ico4_fsaverage_stc_eeg'], \
        #    "inputlocation should be one of 'ras', 'voxel_coordinate_eeg', 'vertex_number_src', 'fsaverage_stc' ,'ico4_fsaverage_stc"
        #self.inputlocation = inputlocation
        
        assert outputtype in ['stc','64voxelmask','peak_ras','center_ras','mean_ras','mask_on_mri','scale'], "outputtype should be 'stc','mask','peak_ras','ras','mask_on_mri'"
        self.outputtype = outputtype
        
        mris = []
        eeg_dir = []
        output_dir = []
        for identity in mri_id:
            mri_data = nib.load(f'/data/pheeeeee/mris/sub-{identity:02d}/sample/mri/T1.mgz') #torch.load(f'/mnt/d/real_dataset/sub-{identity:02d}/mri.pt')
            mri_data = mri_data.get_fdata()
            mri_data = torch.tensor(mri_data, dtype=DTYPE)
            for _ in range(mri_n_downsampling):
                mri_data = mri_downsampling(mri_data,type='aver',down=True)
            mri_data = mri_data.clone().to(self.DTYPE)
            mris.append(mri_data)
            
            if eeg_filter == 0:
                eegpath = PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/eeg')
            else:
                eegpath = PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/eeg{eeg_filter}filtered')
            eeg_dir.append(eegpath)
            
            if outputtype in ['mask', 'mask_on_mri']:
                outputpath = PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/voxel_coordinate')
            elif outputtype == 'scale':
                outputpath = PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/subb-{identity:02d}/ras')
            else:
                outputpath = PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/{outputtype}')
            output_dir.append(outputpath)
            
        self.mris = mris
        self.eeg_dir = eeg_dir
        self.output_dir = output_dir
        self.mri_shape = mri_data.shape
        
        self.less_noise = less_noise
        
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        mri_ind, eeg_ind = divmod(index, self.eeg_per_mri)
        if self.less_noise is True:
            eeg_ind = eeg_ind + 2000
        identity = self.mri_id[mri_ind]
        mri = self.mris[mri_ind]
        eegpath = os.path.join(self.eeg_dir[mri_ind], f'{eeg_ind}.npy')
        eeg = np.load(eegpath)
        if self.ch_names is not None:
            eeg = eeg[self.eeg_channel_index,:]
            
        if self.outputtype == '64voxelmask':
            outputpath = os.path.join(self.output_dir[mri_ind], f'{eeg_ind}.npz')
            output = load_npz(outputpath).toarray()
        else:
            outputpath = os.path.join(self.output_dir[mri_ind], f'{eeg_ind}.npy' )
            output = np.load(outputpath)
        
        if self.eegtype == 'psd':
            _, eeg = welch(np.array(eeg), fs=100, nperseg=100)
        elif self.eegtype == 'fourier cos':
            eeg = np.real(fft(np.array(eeg)))
        elif self.eegtype == 'fourier sin':
            eeg = np.imag(fft(np.array(eeg)))
        elif self.eegtype ==  'fourier concatenate':
            fft_eeg = np.fft.fft(eeg)
            eeg = np.concatenate((np.real(fft_eeg), np.imag(fft_eeg)), axis=-1)
        eeg = torch.tensor(eeg,dtype=self.DTYPE)
        
        if self.outputtype == 'mask':
            output = voxel_to_mask(output, shape=(256, 256, 256))
        #elif self.outputtype == 'mask_on_mri' -> this can be done afterwards.
        elif self.outputtype == 'mean_ras':
            output = output.mean(dim=0)
        elif self.outputtype == 'scale':
            center = np.load(PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/center_ras/{eeg_ind}.npy'))
            distances = np.linalg.norm(output - center, axis=1)
            output = np.max(distances)
        output = torch.tensor(output, dtype=self.DTYPE)
        output = output.squeeze()        
        return identity, mri, eeg, output
    

class MyTrainDataset_sourcereconstructor(Dataset):
    def __init__(self, mri_id:list, 
                 outputtype:str, 
                 eegtype='raw', 
                 DTYPE=torch.float32, 
                 mri_n_downsampling:int=0, 
                 eeg_per_mri:int=2000 
                 ):
        from utils import ras_to_voxel
        self.mri_id = mri_id
        self.eeg_per_mri = eeg_per_mri
        self.size = len(mri_id) * eeg_per_mri
        self.DTYPE = DTYPE
        
        assert eegtype in ['raw','psd','fourier cos', 'fourier sin', 'fourier concatenate'], "eegtype must be one of 'raw','psd','fourier cos', 'fourier sin'"
        self.eegtype = eegtype

        #load mri data and eeg directory
        #assert inputlocation in ['ras', 'peak_ras','voxel_coordinate', 'mask', 'voxel_coordinate_eeg', 'vertex_number_src_eeg', 'fsaverage_stc_eeg','ico4_fsaverage_stc_eeg'], \
        #    "inputlocation should be one of 'ras', 'voxel_coordinate_eeg', 'vertex_number_src', 'fsaverage_stc' ,'ico4_fsaverage_stc"
        #self.inputlocation = inputlocation
        
        assert outputtype in ['ras','mask','mask_on_mri'], "outputtype should be 'ras','mask','mask_on_mri'"
        self.outputtype = outputtype
        
        mris = []
        eeg_dir = []
        output_dir = []
        affines = []
        for identity in mri_id:   
            mri_data = nib.load(f'/mnt/d/openneuro_mris/sub-{identity:02d}/sample/mri/T1.mgz') #torch.load(f'/mnt/d/real_dataset/sub-{identity:02d}/mri.pt')
            affine = mri_data.affine #voxel to ras
            affines.append(affine)
            mri_data = mri_data.get_fdata()
            mri_data = torch.tensor(mri_data, dtype=DTYPE)
            for _ in range(mri_n_downsampling):
                mri_data = mri_downsampling(mri_data,type='aver',down=True)
            mri_data = mri_data.clone().to(self.DTYPE)
            mris.append(mri_data)
            
            eegpath = PosixPath(f'/mnt/d/REAL_FINAL_DATA/montage_standard_1020/amplitude_1/single dipole/sub-{identity:02d}/eeg')
            eeg_dir.append(eegpath)
            
            if outputtype in ['mask', 'mask_on_mri']:
                outputpath = PosixPath(f'/mnt/d/REAL_FINAL_DATA/montage_standard_1020/amplitude_1/single dipole/sub-{identity:02d}/voxel_coordinate')
            elif outputtype == 'scale':
                outputpath = PosixPath(f'/mnt/d/REAL_FINAL_DATA/montage_standard_1020/amplitude_1/single dipole/sub-{identity:02d}/ras')
            else:
                outputpath = PosixPath(f'/mnt/d/REAL_FINAL_DATA/montage_standard_1020/amplitude_1/single dipole/sub-{identity:02d}/{outputtype}')
            output_dir.append(outputpath)
            
        self.mris = mris
        self.affines = affines
        self.eeg_dir = eeg_dir
        self.output_dir = output_dir
        self.mri_shape = mri_data.shape
        
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        mri_ind, eeg_ind = divmod(index, self.eeg_per_mri)
        identity = self.mri_id[mri_ind]
        mri = self.mris[mri_ind]
        affine = self.affines[mri_ind]
        eegpath = os.path.join(self.eeg_dir[mri_ind], f'{eeg_ind}.npy')
        eeg = np.load(eegpath)
        
        outputpath = os.path.join(self.output_dir[mri_ind], f'{eeg_ind}.npy' )
        output = np.load(outputpath)
        
        if self.eegtype == 'psd':
            _, eeg = welch(np.array(eeg), fs=100, nperseg=100)
        elif self.eegtype == 'fourier cos':
            eeg = np.real(fft(np.array(eeg)))
        elif self.eegtype == 'fourier sin':
            eeg = np.imag(fft(np.array(eeg)))
        elif self.eegtype ==  'fourier concatenate':
            fft_eeg = np.fft.fft(eeg)
            eeg = np.concatenate((np.real(fft_eeg), np.imag(fft_eeg)), axis=-1)
        eeg = torch.tensor(eeg,dtype=self.DTYPE)

        if self.outputtype == 'ras':
            output = ras_to_voxel(output , affine)
            output = voxel_to_mask(output, shape=(256, 256, 256))
        #elif self.outputtype == 'mask_on_mri' -> this can be done afterwards.
        output = torch.tensor(output, dtype=self.DTYPE)
        
        return identity, mri, eeg, output
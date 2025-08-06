
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
    def __init__(self, 
                 mri_id:list, 
                 outputtype:str, 
                 mri_dir:str,
                 eeg_subject_dir:str,
                 n_dipole:str,#= 'singledipole',
                 amplitude:str,#= 'amplitude1',
                 snr:str,# = 'snr15',
                 eegtype='raw',   #'fourier concatenate', 'fourier cos', 'fourier sin'
                 DTYPE=torch.float32, 
                 mri_n_downsampling:int=0, 
                 eeg_per_mri:int=2000,
                 eeg_filter:int=0,
                 sensor=False,
                 eeg_time_window=None, # Should be numpy array or list of wanted time window.
                 ch_names = None,
                 all_dipole = False,
                 output_config = True,
                 conformal = False
                 ):
        self.mri_id = mri_id
        self.eeg_per_mri = eeg_per_mri
        self.size = len(mri_id) * eeg_per_mri
        self.DTYPE = DTYPE

        self.mri_dir = mri_dir,
        self.eeg_subject_dir = eeg_subject_dir
        self.eeg_time_window = eeg_time_window
        
        self.sensor = sensor
        self.all_dipole = all_dipole
        self.output_config = output_config
        self.conformal = conformal
        
        assert eegtype in ['raw','psd','fourier cos', 'fourier sin', 'fourier concatenate'], "eegtype must be one of 'raw','psd','fourier cos', 'fourier sin'"
        self.eegtype = eegtype
        self.ch_names = ch_names
        if ch_names is not None:
            info = PosixPath('/home/user/data/pheeeeee/info.fif')#os.path.join(eeg_subject_dir, 'sub-01/info.fif') #PosixPath("/data/pheeeeee/FINAL_EEG_DATA//sub-01/info.fif")
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
        
        assert outputtype in ['stc','64voxelmask','configured_center_ras','barycenter_dipole','peak_ras','center_ras','mean_ras','mask_on_mri','scale', 'center_dipole'], "outputtype should be 'stc','64voxelmask','peak_ras','center_ras','mean_ras','mask_on_mri','scale', 'center_dipole'"
        self.outputtype = outputtype
        
        mris = []
        eeg_dir = []
        output_dir = []
        alldipole_dir = []
        for identity in mri_id:
            #mri_path = os.path.join(mri_dir, f'sub-{identity:02d}/mri{identity:02d}.nii')
            mri_path = os.path.join(mri_dir, f'sub-{identity:02d}/sample/mri/T1.mgz')
            mri_data = nib.load(mri_path) #torch.load(f'/mnt/d/real_dataset/sub-{identity:02d}/mri.pt')
            mri_data = mri_data.get_fdata()
            mri_data = torch.tensor(mri_data, dtype=DTYPE)
            for _ in range(mri_n_downsampling):
                mri_data = mri_downsampling(mri_data,type='aver',down=True)
            mri_data = mri_data.clone().to(self.DTYPE)
            mris.append(mri_data)
            
                        
            if eeg_filter == 0:
                #eegpath = os.path.join(eeg_subject_dir, f'sub-{identity:02d}/{n_dipole}/{amplitude}/eeg/{snr}')
                eegpath = os.path.join(eeg_subject_dir, f'sub-{identity:02d}/eeg')
            else:
                raise FileNotFoundError("Check your eeg subject directory path.")
                #eegpath = os.path.join(eeg_subject_dir, f'sub-{identity:02d}/{n_dipole}/{amplitude}/{snr}/eeg_{eeg_filter}') #PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/eeg{eeg_filter}filtered')
            eeg_dir.append(eegpath)
            
            if outputtype in ['mask', 'mask_on_mri']:
                outputpath = os.path.join(eeg_subject_dir, f'sub-{identity:02d}/{n_dipole}/{amplitude}/{snr}/voxel_coordinate') #PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/voxel_coordinate')
            elif outputtype == 'scale':
                outputpath = os.path.join(eeg_subject_dir, f'sub-{identity:02d}/{n_dipole}/{amplitude}/{snr}/ras') #PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/subb-{identity:02d}/ras')
            else:
                #outputpath = os.path.join(eeg_subject_dir, f'sub-{identity:02d}/{n_dipole}/{amplitude}/{outputtype}') #PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/{outputtype}')
                outputpath = os.path.join(eeg_subject_dir, f'sub-{identity:02d}/{outputtype}')
            output_dir.append(outputpath)
            
            if all_dipole is True:
                alldipole_dir.append(os.path.join(eeg_subject_dir, f'sub-{identity:02d}/ras'))
                self.alldipole_dir = alldipole_dir
                
            
        if sensor is True:
            sensordata = []
            for identity in mri_id:
                default_sensor = [info['dig'][j+3]['r'] for j in eeg_channel_index]
                trans = mne.read_trans(os.path.join(mri_dir, f'sub-{identity:02d}/trans.fif'))  #trans is saved where "sample" is.
                for ith, dip in enumerate(default_sensor):
                    default_sensor[ith] = mne.transforms.apply_trans(trans['trans'], dip)*1000
                sensordata.append(default_sensor)
            self.sensordata = np.array(sensordata)
        
        if output_config:
            assert outputtype != 'configured_center_ras', "Output is already configured."
            trans_set = []
            for identity in mri_id:
                trans = mne.read_trans(os.path.join(mri_dir, f'sub-{identity:02d}/trans.fif'))
                trans_set.append(trans['trans'])
            self.trans_set = trans_set
                
        self.mris = mris
        self.eeg_dir = eeg_dir
        self.output_dir = output_dir
        self.mri_shape = mri_data.shape
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        mri_ind, eeg_ind = divmod(index, self.eeg_per_mri)
        if self.conformal is True:
            eeg_ind = 2000+eeg_ind
        identity = self.mri_id[mri_ind]
        mri = self.mris[mri_ind]
        eegpath = os.path.join(self.eeg_dir[mri_ind], f'{eeg_ind}.npy')
        eeg = np.load(eegpath)
        
        if self.sensor is True:
            sensor = self.sensordata[mri_ind]
            sensor = torch.tensor(sensor, dtype=self.DTYPE)
            
        
        if self.eeg_time_window is not None:
            eeg = eeg[:,self.eeg_time_window]
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
            center = np.load(os.path.join(self.eeg_subject_dir, f'center_ras/{eeg_ind}.npy')) #PosixPath(f'/data/pheeeeee/FINAL_EEG_DATA/sub-{identity:02d}/center_ras/{eeg_ind}.npy'))
            distances = np.linalg.norm(output - center, axis=1)
            output = np.max(distances)
        output = torch.tensor(output, dtype=self.DTYPE)
        output = output.squeeze()
        
        if self.output_config:
            trans = self.trans_set[mri_ind]
            output = mne.transforms.apply_trans(trans, output)
            output = torch.tensor(output, dtype=self.DTYPE)
        
        if self.sensor is True:
            identity = sensor
        
        if self.all_dipole is True:
            alldipole = os.path.join(self.alldipole_dir[mri_ind], f'{eeg_ind}.npy')
            return identity, mri, eeg, output, alldipole
        else:
            return identity, mri, eeg, output
    
    

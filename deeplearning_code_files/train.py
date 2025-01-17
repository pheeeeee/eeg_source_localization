
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from deeplearning_code_files.datautils import MyTrainDataset
from deeplearning_code_files.utils import *
import glob
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import random
import time
import json

class Trainer:
    def __init__(
            self,
            model : torch.nn.Module,
            train_data : DataLoader,
            validation_data : DataLoader,
            optimizer : torch.optim.Optimizer,
            gpu_id : id,
            save_energy: int,
            loss_ft,
            model_mode='mri+eeg',
            pinn_loss=False,
            autocast=False
            ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.save_energy = save_energy
        self.loss_ft = loss_ft
        self.train_loss_traj = []
        self.validation_loss_traj = []
        self.autocast = autocast
        
        assert model_mode in ['mri+eeg', 'eeg'], 'model mode must be either "mri+eeg" or "eeg"'
        self.model_mode = model_mode
        
        self.pinnloss = pinn_loss
        
    def _run_batch(self, identity, mri, eeg, targets):
        if self.phase == 'training':
            self.optimizer.zero_grad()
            if self.autocast is True:
                with autocast(device_type='cuda'):                    
                    if self.model_mode == 'eeg':
                        output = self.model(eeg)
                    else:      
                        output = self.model([mri, eeg])
                    if self.pinnloss == True:
                        brainmasks = torch.zeros_like(output)
                        for _,iii in enumerate(list(identity)):
                            brain_mask_path = PosixPath(f'/mnt/d/openneuro_mris/sub-{iii:02d}') 
                            brainmask = nib.load(os.path.join(brain_mask_path,'sample', 'mri','brainmask.mgz'))
                            brainmask = brainmask.get_fdata()
                            brainmask = torch.tensor(brainmask, dtype=output.dtype, device=output.device)
                            brainmasks[_] = brainmask
                        output = output * brainmasks
                        targets = targets * brainmasks
                        del brainmasks
                        loss = self.loss_ft(output, targets)
                    else:
                        loss = self.loss_ft(output, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.running_train_loss += loss.item()
            else:
                if self.model_mode == 'eeg':
                    output = self.model(eeg)
                else:      
                    output = self.model([mri, eeg])
                if self.pinnloss == True:
                    brainmasks = torch.zeros_like(output)
                    for _,iii in enumerate(list(identity)):
                        brain_mask_path = PosixPath(f'/mnt/d/openneuro_mris/sub-{iii:02d}') 
                        brainmask = nib.load(os.path.join(brain_mask_path,'sample', 'mri','brainmask.mgz'))
                        brainmask = brainmask.get_fdata()
                        brainmask = torch.tensor(brainmask, dtype=output.dtype, device=output.device)
                        brainmasks[_] = brainmask
                    output = output * brainmasks
                    targets = targets * brainmasks
                    del brainmasks
                    loss = self.loss_ft(output, targets)
                else:
                    loss = self.loss_ft(output, targets)
                loss.backward()
                self.optimizer.step()
                self.running_train_loss += loss.item()
                
        if self.phase == 'validation':
            if self.model_mode == 'eeg':
                output = self.model(eeg)
            else:      
                output = self.model([mri, eeg])            
            loss = self.loss_ft(output, targets)
            self.running_validation_loss += loss.item()

    def _run_epoch(self, epoch):
        #training
        self.running_train_loss = 0
        self.model.train()
        self.phase = 'training'
        for identity, mri, eeg, targets in self.train_data:
            mri = mri.to(self.gpu_id)
            eeg = eeg.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(identity, mri,eeg,targets)
        self.train_loss_traj.append(self.running_train_loss/len(self.train_data))
        
        #validation
        self.running_validation_loss = 0
        self.model.eval()
        self.phase = 'validation'
        with torch.no_grad():
            for identity, mri, eeg, targets in self.validation_data:
                mri = mri.to(self.gpu_id)
                eeg = eeg.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                self._run_batch(identity,mri,eeg,targets)
            self.validation_loss_traj.append(self.running_validation_loss/len(self.validation_data))
            
    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        torch.save(ckp, f"checkpoint/{self.gpu_id}checkpoint.pt")
        savingresults = {'epoch':epoch,
                         'training_loss_traj':self.train_loss_traj,
                         'validation_loss_traj':self.validation_loss_traj,
                         'loss function':str(self.loss_ft)
                         }
        with open('checkpoint/results_checkpoint.json', "w") as file:
            json.dump(savingresults, file)
        
        self.epoch = epoch
        if epoch % 10 == 1:
            print(f"Epoch {epoch} | Training checkpoint saved at checkpoit.pt")
        
    def train(self, max_epochs: int):
        if self.autocast is True:
            self.scaler = GradScaler()

        for epoch in tqdm(range(max_epochs)):
            self._run_epoch(epoch)
            if epoch % self.save_energy == 0:
                self._save_checkpoint(epoch)
            print('EPOCH : %6d/%6d | Train Loss : %8.7f  | Validation : %8.7f ' %(epoch, max_epochs, self.train_loss_traj[epoch], self.validation_loss_traj[epoch]))
            torch.cuda.empty_cache()
            
    
    def test(self,
         test_data:DataLoader,
         loss_ft=None,
         seen=False
         ):
        import time
        import statistics
        self.phase = 'test'
        if loss_ft is None:
            loss_ft = self.loss_ft
        self.tested_on_seen_mri = seen
        if seen == True:
            self.seentestloss = []
        self.testloss = []
        self.inference_time = []
        
        self.model.eval()
        with torch.no_grad():
            for identity, mri, eeg, targets in test_data:
                mri = mri.to(self.gpu_id)
                eeg = eeg.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                if self.model_mode == 'mri+eeg':       
                    start_time = time.perf_counter()
                    output = self.model([mri, eeg])
                    end_time = time.perf_counter()
                elif self.model_mode == 'eeg':
                    start_time = time.perf_counter()
                    output = self.model(eeg)
                    end_time = time.perf_counter()
                elif self.model_mode == 'eeg fourier':
                    start_time = time.perf_counter()
                    output = self.model(eeg[0], eeg[1])
                    end_time = time.perf_counter()
                elif self.model_mode == 'mri+eeg+sensor':
                    start_time = time.perf_counter()
                    output = self.model([mri, eeg, sensor])
                    end_time = time.perf_counter()
                if self.pinnloss == True:
                    brainmasks = torch.zeros_like(output)
                    for _,iii in enumerate(list(identity)):
                        brain_mask_path = PosixPath(f'/mnt/d/openneuro_mris/sub-{iii:02d}') 
                        brainmask = nib.load(os.path.join(brain_mask_path,'sample', 'mri','brainmask.mgz'))
                        brainmask = brainmask.get_fdata()
                        brainmask = torch.tensor(brainmask, dtype=output.dtype, device=output.device)
                        brainmasks[_] = brainmask
                    output = output * brainmasks
                    targets = targets * brainmasks
                    del brainmasks
                    loss = self.loss_ft(output, targets)
                else:
                    loss = self.loss_ft(output, targets)
                if seen == True:
                    self.seentestloss.append(loss.item())
                self.testloss.append(loss.item())
                self.inference_time.append(end_time - start_time)
            if seen == True:
                self.seentestloss = np.array(self.seentestloss)
            else:
                self.testloss = np.array(self.testloss)
            if seen == True:
                ave_testloss = self.seentestloss.mean()
            else:
                ave_testloss = self.testloss.mean()
            print(f'mean testloss :{ave_testloss} (std: {statistics.stdev(self.testloss)})' )
            self.inference_time = np.array(self.inference_time)
            self.ave_inference_times= self.inference_time.mean()
            print(f'mean inference time : {self.ave_inference_times} and std : {statistics.stdev(self.inference_time)}')


    def save_results(self, directory_path, tested_on_seen_mri):
        import json
        import glob
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        experiment_data = {
                "hyperparameters": {
                    "epochs": self.epoch,
                    "loss function" :str(self.loss_ft),
                    "autocast" : self.autocast
                },
                "results": {
                    "training loss" : self.train_loss_traj,
                    "average training loss" : np.array(self.train_loss_traj).mean(),
                    "validation loss" : self.validation_loss_traj,
                    "average validataion loss" : np.array(self.validation_loss_traj).mean(),
                    "test loss" : self.testloss,
                    "average test loss" : self.testloss.mean(),
                }
            }
        if tested_on_seen_mri == True:
            assert len(self.seentestloss) > 1, "not tested on seen mri data"
            experiment_data['results']['test loss for seen mri data'] = self.seentestloss
            experiment_data['results']['average test loss for seen mri data'] = self.seentestloss.mean()
            
        json_file_path = os.path.join(directory_path, "experiment_results.json")
        with open(json_file_path, "w") as json_file:
            json.dump(experiment_data, json_file, indent=4)
        
        model_checkpoint_path = os.path.join(directory_path, "model_checkpoint.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': experiment_data["hyperparameters"],
            'results': experiment_data["results"]
        }, model_checkpoint_path)

        print("results saved.")
    

#Helper Functions: It loads training set, model and optimizer
def load_train_objs(model:torch.nn.Module, mri_ids:list, outputtype:str, mri_n_downsampling:int=0, eeg_per_mri:int=2000):
    train_set = MyTrainDataset(mri_id=mri_ids, outputtype = outputtype, mri_n_downsampling = mri_n_downsampling, eeg_per_mri = eeg_per_mri) # Load Dataset
    model = fusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset:Dataset, batch_size: int , val_split: float = 0.2):
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=True,  # Shuffle the training set for better generalization
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=False,  # No need to shuffle the validation set
        drop_last=False  # Retain all validation data
    )
    
    return train_loader, val_loader

def main(model, mri_ids, outputtype, device, total_epochs, save_energy, mri_n_downsampling=0, eeg_per_mri=2000):
    dataset, model, optimizer = load_train_objs(model, mri_ids, outputtype, mri_n_downsampling, eeg_per_mri)
    train_data = prepare_dataloader(dataset, batch_size=32)
    trainer = Trainer(model, train_data, optimizer, device, save_energy)
    traner.train(total_epochs)
    
    
if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    device = 'cuda'
    main()

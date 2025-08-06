

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
#from torch.amp import autocast, GradScaler

from deeplearning_code_files.datautils import MyTrainDataset
from deeplearning_code_files.utils import *
import glob
import os
import math

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
            autocast=False,
            early_stop=True,
            saving_dir_path='/home/pheeeeee/neuroimaging/checkpoint'
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
        self.early_stop = early_stop
        self.saving_dir_path = saving_dir_path
        self.best_loss = float('inf')
        self.best_epoch=0
        if early_stop:
            if type(early_stop) is int:
                self.patience = early_stop
            else:
                self.patience = 30 #Number of epochs to wait before stopping.
            self.best_loss = float('inf')
            self.counter = 0
        
        if os.path.exists(saving_dir_path):
            print("Saving Directory already exists.") 
        else:
            choice = input(f"Do you want to create '{saving_dir_path}'? (yes/no): ").strip()
            if choice == 'yes':
                os.makedirs(saving_dir_path)
                print(f'Directory {saving_dir_path} created successfully.')
            else:
                print("Directory not created")
        
        assert model_mode in ['sensor+mri+eeg','mri+eeg', 'eeg'], 'model mode must be one of "sensor+mri+eeg", "mri+eeg" or "eeg"'
        self.model_mode = model_mode
        
        self.pinnloss = pinn_loss
        
    def _run_batch(self, identity, mri, eeg, targets):
        if self.phase == 'training':
            self.optimizer.zero_grad()
            if self.autocast is True:
                with autocast(device_type='cuda'):                    
                    if self.model_mode == 'eeg':
                        output = self.model(eeg)
                    elif self.model_mode == 'mri+eeg':      
                        output = self.model([mri, eeg])
                    elif self.model_mode == 'sensor+mri+eeg':
                        output = self.model([identity, mri, eeg]) #When sensor is True, identity is the sensor location.
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
                elif self.model_mode == 'mri+eeg':      
                    output = self.model([mri, eeg])
                elif self.model_mode == 'sensor+mri+eeg':
                    output = self.model([identity, mri, eeg]) #When sensor is True, identity is the sensor location.
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
            elif self.model_mode == 'mri+eeg':      
                output = self.model([mri, eeg])
            elif self.model_mode == 'sensor+mri+eeg':
                output = self.model([identity, mri, eeg]) #When sensor is True, identity is the sensor location.
            loss = self.loss_ft(output, targets)
            self.running_validation_loss += loss.item()

    def _run_epoch(self, epoch):
        #training
        self.running_train_loss = 0
        self.model.train()
        self.phase = 'training'
        for identity, mri, eeg, targets in self.train_data:
            identity = identity.to(self.gpu_id)
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
                identity = identity.to(self.gpu_id)
                mri = mri.to(self.gpu_id)
                eeg = eeg.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                self._run_batch(identity,mri,eeg,targets)
            self.validation_loss_traj.append(self.running_validation_loss/len(self.validation_data))
            
            if self.running_validation_loss/len(self.validation_data) < self.best_loss:
                self.best_loss = self.running_validation_loss/len(self.validation_data)
                self.best_epoch = len(self.validation_loss_traj)
                self.best_model = self.model
                if self.early_stop:
                    self.counter = 0
            else:
                if self.early_stop:
                    self.counter += 1
                    if self.counter >= self.patience:
                        torch.save(self.model.state_dict(), os.path.join(self.saving_dir_path, f"best_model_sofar_checkpoint_{self.gpu_id}.pt"))
                        return True   
            #print(f"\n Best validation loss so far : (epoch : {self.best_epoch}, validation loss : {self.best_loss})")


            
    def _save_checkpoint(self, epoch):
        directory_path = self.saving_dir_path
        ckp = self.model.state_dict()
        torch.save(ckp, os.path.join(directory_path, f"model_checkpoint_{self.gpu_id}.pt"))
        savingresults = {'epoch':epoch,
                         'training_loss_traj':self.train_loss_traj,
                         'validation_loss_traj':self.validation_loss_traj,
                         'loss function':str(self.loss_ft)
                         }
        with open(os.path.join(directory_path, 'results_checkpoint.json'), "w") as file:
            json.dump(savingresults, file)
        
        self.epoch = epoch
        if epoch % 10 == 1:
            print(f"Epoch {epoch} | Training checkpoint saved at checkpoit.pt")
        
    def train(self, max_epochs: int):
        if self.autocast is True:
            self.scaler = GradScaler()
        for epoch in tqdm(range(max_epochs)):
            should_stop = self._run_epoch(epoch)
            if epoch % self.save_energy == 0:
                self._save_checkpoint(epoch)
            print('EPOCH : %6d/%6d | Train Loss : %8.7f  | Validation : %8.7f ' %(epoch, max_epochs, self.train_loss_traj[epoch], self.validation_loss_traj[epoch]))
            if should_stop:
                print("Early Stopping Triggered")
                break
            
    
    def test(self,
         test_data:DataLoader,
         loss_ft=None,
         seen=False,
         model = None
         ):
        import time
        import statistics
        self.phase = 'test'
        if loss_ft is None:
            loss_ft = self.loss_ft
        self.tested_on_seen_mri = seen
        if seen == True:
            self.seentestloss = []
        else:
            self.testloss = []
        self.inference_time = []
        
        if model is None:
            model = self.model
        model.eval()
        with torch.no_grad():
            for identity, mri, eeg, targets in test_data:
                identity = identity.to(self.gpu_id)
                mri = mri.to(self.gpu_id)
                eeg = eeg.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                if self.model_mode == 'mri+eeg':       
                    start_time = time.perf_counter()
                    output = model([mri, eeg])
                    end_time = time.perf_counter()
                elif self.model_mode == 'eeg':
                    start_time = time.perf_counter()
                    output = model(eeg)
                    end_time = time.perf_counter()
                elif self.model_mode == 'sensor+mri+eeg':
                    start_time = time.perf_counter()
                    output = model([identity, mri, eeg])  #if sensor is True, identity is the sensor location.
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
                    loss = loss_ft(output, targets)
                else:
                    loss = loss_ft(output, targets)
                if seen == True:
                    self.seentestloss.append(loss.item())
                else:
                    self.testloss.append(loss.item())
                self.inference_time.append(end_time - start_time)
            if seen == True:
                self.seentestloss = np.array(self.seentestloss)
                testloss = self.seentestloss
            else:
                self.testloss = np.array(self.testloss)
                testloss = self.testloss
            
            if "MSE" in str(self.loss_ft):
                testloss = np.sqrt(testloss)
            
            ave_testloss = testloss.mean()
            print(f'Evaluated with metric {str(loss_ft)}')
            print(f'mean testloss :{ave_testloss} (std: {statistics.stdev(testloss)})' )
            self.inference_time = np.array(self.inference_time)
            self.ave_inference_times= self.inference_time.mean()
            print(f'mean inference time : {self.ave_inference_times} and std : {statistics.stdev(self.inference_time)}')


    def save_results(self, tested_on_seen_mri, directory_path=None):
        import json
        if directory_path is None:
            directory_path = self.saving_dir_path
        experiment_data = {
                "hyperparameters": {
                    "epochs": self.epoch,
                    "loss function" :str(self.loss_ft),
                    "autocast" : self.autocast
                },
                "results": {
                    "training loss" : list(self.train_loss_traj),
                    "average training loss" : np.array(self.train_loss_traj).mean(),
                    "validation loss" : list(self.validation_loss_traj),
                    "average validataion loss" : np.array(self.validation_loss_traj).mean(),
                    "unseen test loss" : list(self.testloss),
                    "average unseen test loss" : self.testloss.mean(),
                }
            }
        if tested_on_seen_mri == True:
            assert len(self.seentestloss) > 1, "not tested on seen mri data"
            experiment_data['results']['test loss for seen mri data'] = list(self.seentestloss)
            experiment_data['results']['average test loss for seen mri data'] = self.seentestloss.mean()
            
        json_file_path = os.path.join(directory_path, "experiment_results.json")
        with open(json_file_path, "w") as json_file:
            json.dump(experiment_data, json_file, indent=4)
        
        model_checkpoint_path = os.path.join(directory_path, "model_checkpoint_test.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': experiment_data["hyperparameters"],
            'results': experiment_data["results"]
        }, model_checkpoint_path)
        
        np.save(os.path.join(directory_path, 'trainingloss.npy'), self.train_loss_traj)
        np.save(os.path.join(directory_path, 'validationloss.npy'),self.validation_loss_traj)
        np.save(os.path.join(directory_path, 'seentestloss.npy'),self.seentestloss)
        np.save(os.path.join(directory_path, 'unseentestloss.npy'),self.testloss)

        print("results saved.")

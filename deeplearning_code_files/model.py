


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal
import math


class mrisubnet(nn.Module):
    def __init__(self, conv_dims, mlp_dims, conv_kernel_size, dropout=0.2):
        super(mrisubnet,self).__init__()
        if len(mlp_dims) == 1:
            mlp_dims = [conv_dims[-1]] + mlp_dims
            
        self.num_conv_blocks = len(conv_dims)-1
        self.num_mlp_layers = len(mlp_dims) - 1
        self.layers = nn.ModuleList()

        for i in range(self.num_conv_blocks):
            self.layers.append(self._conv_layer_set(conv_dims[i], conv_dims[i+1], kernel_size=conv_kernel_size))
        
        for i in range(self.num_mlp_layers):
            self.layers.append(nn.Linear(mlp_dims[i],mlp_dims[i+1]))
        
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        
    def _conv_layer_set(self, in_channels, out_channels, kernel_size):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=1,
                padding=0,
                ),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            )
        return conv_layer

    def forward(self, x):
        x = x.unsqueeze(1) #Conv3d require 5dim (batch, channels, depth, height, width)
        for i in range(self.num_conv_blocks):
            layer = self.layers[i]
            x = self.act(layer(x))
        x = x.view(x.size(0),-1)
        for layer in self.layers[self.num_conv_blocks:-1]:
            x = self.act(layer(x))
            x = self.drop(x)
        x = self.layers[-1](x)
        return x
    
    
        
class eegsubnet(nn.Module):
    def __init__(self, conv_dims, mlp_dims, conv_kernel_size, dropout=0.2, max_pooling=False):
        """fourier =True means both real and imag parts of fourier transformed are input."""
        super(eegsubnet,self).__init__()
        if len(mlp_dims) == 1:
            mlp_dims = [conv_dims[-1]] + mlp_dims
            
        self.num_conv_blocks = len(conv_dims)-1
        self.num_mlp_layers = len(mlp_dims) - 1
        self.layers = nn.ModuleList()
        for i in range(self.num_conv_blocks):
            self.layers.append(self._conv_layer_set(conv_dims[i], conv_dims[i+1], kernel_size=conv_kernel_size, maxpooling=max_pooling))
        
        for i in range(self.num_mlp_layers):
            self.layers.append(nn.Linear(mlp_dims[i],mlp_dims[i+1]))
        
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        
    def _conv_layer_set(self, in_channels, out_channels, kernel_size=3, maxpooling=True):
        if maxpooling:
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=1,
                    padding=0,
                    ),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                )
        else:
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=1,
                    padding=0,
                    ))
        return conv_layer
    
    def forward(self, x):
        for i in range(self.num_conv_blocks):
            layer = self.layers[i]
            x = self.act(layer(x))
        #print(f'eeg input shape before flattening:{x.shape}')
        x = x.view(x.size(0),-1)
        #print(f'eeg input shape after flattening:{x.shape}')
        for layer in self.layers[self.num_conv_blocks:-1]:
            x = self.act(layer(x))
            x = self.drop(x)
        x = self.layers[-1](x)
        return x
    

class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_x=256, max_y=256, max_z=256):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.x_pos = self._positional_encoding_1d(max_x, d_model // 3)
        self.y_pos = self._positional_encoding_1d(max_y, d_model // 3)
        self.z_pos = self._positional_encoding_1d(max_z, d_model // 3)

    def _positional_encoding_1d(self, length, d_model):
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x:int, y:int, z:int):
        """ 
        x, y, z : sensor location (int)
        """
        x_emb = self.x_pos[x]
        y_emb = self.y_pos[y]
        z_emb = self.z_pos[z]
        pos_emb = torch.cat((x_emb, y_emb, z_emb), dim=-1)
        return pos_emb
        
    
    
    
    
    
class sensorsubnet(nn.Module):
    def __init__(self, mlp_dims, dropout):
        super(sensorsubnet, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(mlp_dims)-1):
            self.layers.append(nn.Linear(mlp_dims[_], mlp_dims[_+1]))
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.GELU()
    def forward(self, x):
        x = x.view(x.size(0),-1)
        for layer in self.layers:
            x = self.act(layer(x))
            x = self.drop(x)
        return x
        

class fusion(nn.Module):
    def __init__(self, mri_conv_dims=[1], mri_mlp_dims=[1], mri_conv_kernel_size=(3,3,3), 
                 eeg_conv_dims=[1], eeg_mlp_dims=[1], eeg_conv_kernel_size=3, 
                 fusion_conv_dims=[2], fusion_dims=[1], fusion_conv_kernel_size=3, dropout=[0.2,0.2,0.2,0.2],
                 sensor_dims =[1],
                 batch_size=None, 
                 use_sigmoid=False, output_as_3d=False, ouput_shape_as_3d=None,
                 MRI=True, 
                 sensor=False):
        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        self.output_as_3d = output_as_3d
        self.ouput_shape_as_3d = ouput_shape_as_3d
        self.MRI = MRI
        self.sensor = sensor
        
        # define the pre-fusion subnetworks
        self.mri_subnet = mrisubnet(mri_conv_dims, mri_mlp_dims, mri_conv_kernel_size, dropout=dropout[1])
        self.eeg_subnet = eegsubnet(eeg_conv_dims, eeg_mlp_dims, eeg_conv_kernel_size, dropout=dropout[2])

        if sensor is True:
            self.sensor_subnet = sensorsubnet(sensor_dims, dropout=dropout[0])
        
        #Post_Fusion Learning
        if len(fusion_dims) == 1:
            fusion_dims = [fusion_conv_dims[-1]] + fusion_dims

        self.num_fusion_conv_blocks = len(fusion_conv_dims)-1
        self.fusion_num_mlp_layers = len(fusion_dims) - 1
        self.fusion_layers = nn.ModuleList()

        for i in range(self.num_fusion_conv_blocks):
            self.fusion_layers.append(self._conv_layer_set(fusion_conv_dims[i], fusion_conv_dims[i+1], kernel_size=fusion_conv_kernel_size))
        
        for i in range(self.fusion_num_mlp_layers):
            self.fusion_layers.append(nn.Linear(fusion_dims[i],fusion_dims[i+1]))
        
        self.drop = nn.Dropout(dropout[3])
        self.act = nn.GELU()

    def _conv_layer_set(self, in_channels, out_channels, kernel_size=3):
        conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=1,
                padding=0,
                ),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        return conv_layer

    
    def forward(self, inputx):
        #print(f'inputx type and length:{type(inputx), len(inputx)}')
        #print(f'mri shape:{inputx[0].shape}')
        #print(f'eeg shape:{inputx[1].shape}')
        if self.MRI is True:
            if self.sensor is True:
                sensor_h = self.sensor_subnet(inputx[0])
                mri_h = self.mri_subnet(inputx[1])
                eeg_h = self.eeg_subnet(inputx[2])
                output = torch.stack([sensor_h, mri_h, eeg_h], dim=1)
            else:
                mri_h = self.mri_subnet(inputx[0])
                eeg_h = self.eeg_subnet(inputx[1])
                #FUSION2. STACK
                output = torch.stack([mri_h, eeg_h], dim=1)
        else:
            output = self.eeg_subnet(inputx)
            #print(f'output of eeg_subnet : {output.shape}')
        output = output.view(output.size(0),-1)
        for i in range(self.num_fusion_conv_blocks):
            layer = self.fusion_layers[i]
            output = self.act(layer(output))
        output = output.view(output.size(0),-1)
        for layer in self.fusion_layers[self.num_fusion_conv_blocks:-1]:
            output = self.act(layer(output))
            output = self.drop(output)
        output = self.fusion_layers[-1](output)
        
        if self.use_sigmoid:
            output = F.sigmoid(output)
        if self.output_as_3d:
            output = output.view(-1, self.ouput_shape_as_3d[0],self.ouput_shape_as_3d[1],self.ouput_shape_as_3d[2])
        return output
    
    

        
        
        
        

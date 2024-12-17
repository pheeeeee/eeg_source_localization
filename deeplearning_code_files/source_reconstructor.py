


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal
from utils import ras_to_voxel
import nibabel as nib


# goal : EEG + estimate peak dipole => source reconstruction mask (258,258,258)


#use     max_norm = 1.0  # 그레이디언트 노름의 최대값
#   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
"""
# 학습 루프
for epoch in range(num_epochs):
    model.train()
    for batch_data in train_loader:
        affine, x, eeg, target = batch_data
        
        # Forward pass
        output = model(affine, x, eeg)
        
        # Loss 계산
        loss = criterion(output, target)
        
        # Backward 및 최적화
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()"""

class source_reconstructor(nn.Module):
    def __init__(self, pretrained_model):
        super(source_reconstructor, self).__init__()
        if type(pretrained_model) is not int:
            self.pretrained_model = pretrained_model
            d = 3
        else:
            d = pretrained_model
            mri_conv_dims = [1,1,1,1]
            mri_mlp_dims=[2744,1000,200,100]
            eeg_conv_dims=[94,32,32,16]
            eeg_mlp_dims=[3104,1000,200,100]
            fusion_dims = [200,100,d]
            pretrained_model = fusion(mri_conv_dims=mri_conv_dims, 
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
            self.pretrained_model = pretrained_model
            
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,512)
        self.fc4 = nn.Linear(512,2048)
        self.fc5 = nn.Linear(2048,4096) #making K
        
        self.eeg1 = self._conv_layer_set(94,64)
        self.eeg2 = self._conv_layer_set(64,32)
        self.eeg3 = self._conv_layer_set(32,32)
        #self.eeg4 = self._conv_layer_set(16,8) #making Q
        self.eegfc1 = nn.Linear(736, 2000)
        self.eegfc2 = nn.Linear(2000,4096)
        
        self.fusion1 = nn.Linear(4096*2,4096) #making V
        
        self.attention1 = ScaledDotProductAttention(64)
        self.fc7 = nn.Linear(4096, 20000)
        self.fc8 = nn.Linear(20000, 32768)
        
        self.up1 = nn.ConvTranspose3d(1, 1, kernel_size=4, stride=2, padding=1) #32->64
        self.up2 = nn.ConvTranspose3d(1, 1, kernel_size=4, stride=2, padding=1) #64->128
        self.up3 = nn.ConvTranspose3d(1, 1, kernel_size=4, stride=2, padding=1) #128->256

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
        mri = inputx[0]
        eeg = inputx[1]
        x = self.pretrained_model([mri,eeg])
        #x = ras_to_voxel(x, affine)
        x = nn.GELU()(self.fc1(x))
        x = nn.GELU()(self.fc2(x))
        x = nn.GELU()(self.fc3(x))
        x = nn.GELU()(self.fc4(x))
        x = nn.GELU()(self.fc5(x))
        
        eeg = nn.GELU()(self.eeg1(eeg))
        eeg = nn.GELU()(self.eeg2(eeg))
        eeg = nn.GELU()(self.eeg3(eeg))
        #eeg = self.eeg4(eeg)
        eeg = eeg.view(eeg.size(0),-1)
        eeg = nn.GELU()(self.eegfc1(eeg))
        eeg = nn.GELU()(self.eegfc2(eeg))
                
        output = torch.stack([x, eeg], dim=1)
        output = output.view(output.size(0),-1)
        output = nn.GELU()(self.fusion1(output))
        
        output = self.attention1(eeg.unsqueeze(1),x.unsqueeze(1),output.unsqueeze(1))
        output = nn.GELU()(self.fc7(output.squeeze(1)))
        output = nn.GELU()(self.fc8(output))
        
        output = output.view(-1, 1, 32, 32, 32)
        output = nn.GELU()(self.up1(output))
        output = nn.GELU()(self.up2(output))
        output = self.up3(output)
        output = output.view(-1, 256, 256, 256)
        #output = torch.sigmoid(output)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    def forward(self, Q, K, V):
        # Q: (batch_size, num_heads, seq_len, d_k)
        # K: (batch_size, num_heads, seq_len, d_k)
        # V: (batch_size, num_heads, seq_len, d_v)
        
        # Calculate the attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Multiply the weights by the values
        output = torch.matmul(attn_weights, V)
        
        return output
    
    
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
        x = x.squeeze(0)
        x = x.unsqueeze(1)
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
        x = x.view(x.size(0),-1)
        for layer in self.layers[self.num_conv_blocks:-1]:
            x = self.act(layer(x))
            x = self.drop(x)
        x = self.layers[-1](x)
        return x
    

class fusion(nn.Module):
    def __init__(self, mri_conv_dims=[1], mri_mlp_dims=[1], mri_conv_kernel_size=(3,3,3), 
                 eeg_conv_dims=[1], eeg_mlp_dims=[1], eeg_conv_kernel_size=3, 
                 fusion_conv_dims=[2], fusion_dims=[1], fusion_conv_kernel_size=3, dropout=[0.2,0.2,0.2],
                 batch_size=None, 
                 use_sigmoid=False, output_as_3d=False, ouput_shape_as_3d=None):
        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        self.output_as_3d = output_as_3d
        self.ouput_shape_as_3d = ouput_shape_as_3d
        
        # define the pre-fusion subnetworks
        self.mri_subnet = mrisubnet(mri_conv_dims, mri_mlp_dims, mri_conv_kernel_size, dropout=dropout[0])
        self.eeg_subnet = eegsubnet(eeg_conv_dims, eeg_mlp_dims, eeg_conv_kernel_size, dropout=dropout[1])

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
        
        self.drop = nn.Dropout(dropout[2])
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
        mri_h = self.mri_subnet(inputx[0])
        eeg_h = self.eeg_subnet(inputx[1])
        
        #FUSION2. STACK
        output = torch.stack([mri_h, eeg_h], dim=1)
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
    
    
    
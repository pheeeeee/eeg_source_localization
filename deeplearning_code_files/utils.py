

import numpy as np
import math
import torch
import copy
import mne
from torch.nn.utils.rnn import pad_sequence
import nibabel as nib
import torch.nn as nn
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
import os
from pathlib import PosixPath

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def ras_to_voxel(vertices, affine):
    # Convert RAS coordinates to homogeneous coordinates
    vertices_homogeneous = np.c_[vertices, np.ones(vertices.shape[0])]
    # Apply the inverse affine transformation
    voxels_homogeneous = np.linalg.inv(affine).dot(vertices_homogeneous.T).T
    # Extract the voxel coordinates (and round to the nearest integer if needed)
    voxel_coords = voxels_homogeneous[:, :3]
    voxel_coords = np.floor(voxel_coords).astype(int)
    return voxel_coords


#Convert Voxel Coordinate to a Mask
def voxel_to_mask(coordinate, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[coordinate[:, 0], coordinate[:, 1], coordinate[:, 2]] = 1
    return mask

def voxel_to_mask_shape(coordinate, shape):
    mask = torch.zeros(shape)
    mask[coordinate[:, 0], coordinate[:, 1], coordinate[:, 2]]=1
    return mask

# Place Mask on the MRI
def mask_on_mri_channel(mask, mri_data):
    if type(mri_data) != torch.Tensor:
        mri_data = torch.tensor(mri_data)
    if type(mask) != torch.Tensor:
        mask = torch.tensor(mask)
    if len(mri_data.shape) == 3:
        assert mask.shape == mri_data.shape, "The Shape of mask should be equal to the shape of MRI."
        mri_data = torch.stack([mri_data, mask], axis=0)
    elif len(mri_data.shape) == 4:
        mask = mask.unsqueeze(0)
        mri_data = torch.cat((mri_data, mask), axis=0)
    return mri_data

def mask_on_mri_color(mask, mri_data, color=[1,0,0]):
    mri_data = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
    mri_data = np.stack([mri_data] * 3, axis=-1)
    color = np.array(color) 
    colored_mask = np.zeros_like(mri_data)
    colored_mask[mask > 0] = color
    # Overlay the colored mask on the MRI with full opacity (no transparency)
    mri_data[mask > 0] = colored_mask[mask > 0]
    return mri_data



def process_subject(identity,downsampling,output_as_image):
    subjects_dir = PosixPath(f'/mnt/d/real_dataset/sub-{identity:02d}')
    datasets = {}
    mri_data = torch.load(f'/mnt/d/real_dataset/sub-{identity:02d}/mri.pt')
    mri_data = mri_data[:360,:480,:480]
    for _ in range(downsampling):
        mri_data = mri_downsampling(mri_data,type='aver',down=True)
    eegpath = os.path.join(subjects_dir, 'ras_eeg1.pt')
    eeg_set = torch.load(eegpath)
    if output_as_image:
        affine = np.load(f'/mnt/d/real_dataset/sub-{identity:02d}/voxel_to_RAS_affine_matrix.npy')
    for iiii, (eeg, dipole_coord) in enumerate(eeg_set):
        dipole_coord = dipole_coord.numpy()
        if output_as_image:
            dipole_coord = ras_to_voxel(dipole_coord, affine)
            dipole_coord = np.unique(dipole_coord, axis=0)
            dipole_coord = torch.tensor(dipole_coord)
        eeg_set[iiii] = (eeg, dipole_coord)
    torch.save(eeg_set, f'/mnt/d/real_dataset/sub-{identity:02d}/eeg1_dipole_voxel.pt')
    
    datasets['mri'] = mri_data
    datasets['eeg_set'] = eeg_set
    return identity, datasets



def load_data_ras(indices, downsampling, output_as_image):
    datasets = {}
    
    # Parallelizing the subject processing using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda identity: process_subject(identity, downsampling, output_as_image), indices)
    
    for identity, data in results:
        datasets[identity] = data
    return datasets




    
def load_src(identity):
    src = mne.read_source_spaces(f'/mnt/d/real_dataset/sub-{identity:02d}/src.fif')
    return src

def load_sensor(identity, output_as_image=False, downsampling=0):
    #Sensor location in mni space
    info = mne.io.read_info(f'/mnt/d/real_dataset/sub-{identity:02d}/info.fif')
    trans = mne.read_trans(f'/mnt/d/real_dataset/sub-{identity:02d}/head_MRI{identity}_trans.fif')
    affine = np.load(f'/mnt/d/real_dataset/sub-{identity:02d}/voxel_to_RAS_affine_matrix.npy')
    
    montage = info.get_montage()
    sensor_positions = montage.get_positions()['ch_pos']
    sensor_positions = np.array(list(sensor_positions.values()))
    sensor_positions = np.hstack([sensor_positions, np.ones((sensor_positions.shape[0], 1))])
    sensors_mri = sensor_positions.dot(trans['trans'].T)
    sensors_mri = sensors_mri[:, :3]*1000
    
    if output_as_image:
        sensor_voxel = ras_to_voxel(sensors_mri, affine)
        sensor_mask = voxel_to_mask_shape(sensor_voxel, [361,480,481])
        sensor_mask = torch.tensor(sensor_mask)
        for _ in range(downsampling):
            sensor_mask = mri_downsampling(sensor_mask,type='aver',down=True)
        return sensor_mask
    else:
        return sensors_mri

def load_white_matter(identity, downsampling=0):
    #White Matter (Support space)
    subjects_dir = PosixPath(f'/mnt/d/real_dataset/sub-{identity:02d}')
    lh_white_surface_dir = os.path.join(subjects_dir, 'sample', 'surf', 'lh.white' )
    rh_white_surface_dir = os.path.join(subjects_dir, 'sample', 'surf', 'rh.white' )
    lh_vertices, lh_faces = mne.read_surface(lh_white_surface_dir)
    rh_vertices, rh_faces = mne.read_surface(rh_white_surface_dir)
    vertices = np.unique(np.vstack((lh_vertices, rh_vertices)),axis=0)
    affine = np.load(f'/mnt/d/real_dataset/sub-{identity:02d}/voxel_to_RAS_affine_matrix.npy')
    
    lh_voxel_coords = ras_to_voxel(lh_vertices, affine)
    rh_voxel_coords = ras_to_voxel(rh_vertices, affine)
    
    #Because this is for desinating all the "candidate", I want both floor/ceiling of vertice to be masked.
    lh_voxel_coords = np.floor(lh_voxel_coords).astype(int)
    rh_voxel_coords = np.floor(rh_voxel_coords).astype(int)
    voxel_coords = np.vstack((lh_voxel_coords, rh_voxel_coords))
    voxel_coords = np.unique(voxel_coords, axis=0)
    voxel_coords = np.vstack((voxel_coords, voxel_coords + np.array([1,0,0]), voxel_coords + np.array([0,1,0]) ,  voxel_coords + np.array([0,0,1]), voxel_coords + np.array([0,1,1]), voxel_coords + np.array([1,0,1]), voxel_coords + np.array([1,1,0]), voxel_coords + np.array([1,1,1])))
    voxel_coords = np.unique(voxel_coords, axis=0)
    
    whitematter_mask = np.zeros([360,480,480], dtype=np.uint8)
    whitematter_mask[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1
    whitematter_mask = torch.tensor(whitematter_mask).float()
    for _ in range(downsampling):
        whitematter_mask = mri_downsampling(whitematter_mask,type='aver',down=True)
    if len(whitematter_mask.shape) == 4:
        whitematter_mask = whitematter_mask.squeeze(0)
    return whitematter_mask


# Function to load MRI images
def load_mri_images(file_paths):
    images = [nib.load(fp).get_fdata() for fp in file_paths]
    return images

# Compute the average image
def compute_average_image(images):
    average_image = np.mean(images, axis=0)
    return average_image



def register_to_average_image(image_path, average_image_path):
    fixed_image = sitk.ReadImage(average_image_path)
    moving_image = sitk.ReadImage(image_path)
    
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    
    # Set initial transformation
    initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
    registration_method.SetInitialTransform(initial_transform)
    
    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Apply the transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(final_transform)
    resampler.SetSize(fixed_image.GetSize())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    
    aligned_image = resampler.Execute(moving_image)
    return aligned_image




# Define a fluorescent green colormap
def fluorescent_green_colormap():
    colors = [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)]  # Pure fluorescent green
    return mcolors.LinearSegmentedColormap.from_list('fluorescent_green', colors)

# Define a fluorescent red colormap
def fluorescent_red_colormap():
    colors = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]  # Light red to pure red
    return mcolors.LinearSegmentedColormap.from_list('fluorescent_red', colors)

def fluorescent_blue_colormap():
    colors = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]  # Light blue to pure blue
    return mcolors.LinearSegmentedColormap.from_list('fluorescent_blue', colors)

def white_colormap():
    colors = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
    return mcolors.LinearSegmentedColormap.from_list('fluorescent_blue', colors)
    

def get_color_rgba(color_name):
    # Mapping color names to RGBA values
    color_map = {
        'red': (1, 0, 0, 1),
        'green': (0, 1, 0, 1),
        'blue': (0, 0, 1, 1),
        'yellow': (1, 1, 0, 1),
        'cyan': (0, 1, 1, 1),
        'magenta': (1, 0, 1, 1),
        'white': (1, 1, 1, 1),
        'black': (0, 0, 0, 1),
        'fluorescent_green': None  # Special case handled separately
    }
    return color_map.get(color_name, (1, 1, 1, 1))





def masking_on_mri(identity, *coordinates):
    print("Rhe coordinates must be in voxel coordinate frame.")
    assert all(isinstance(item, dict) for item in coordinates), \
        "coordinates input must be a list of dictionaries of  {'coordinates' : coordinates, 'color': color, 'radius': radius}"
    subjects_dir = PosixPath(f'/mnt/d/openneuro_hearing_loss/sub-{identity:02d}/ses-01/anat')
    mripath = os.path.join(subjects_dir, f'sub-{identity:02d}_ses-01_T1w.nii.gz')
    img = nib.load(mripath)
    data = img.get_fdata()
    mri_shape = (192, 256, 256)
    
    #Initialize overlay with grayscale MRI image
    overlay = np.zeros(data.shape + (4,), dtype=np.float32)
    overlay[..., 0] = data / np.max(data)
    overlay[..., 1] = data / np.max(data)
    overlay[..., 2] = data / np.max(data)
    overlay[..., 3] = 1.0
    
    for masking_object in coordinates:
        mask = np.zeros(data.shape, dtype=np.uint8)
        voxel_coords = masking_object['coordinates']
        radius = masking_object['radius']
        for coor in voxel_coords:
            x, y, z = np.indices(mri_shape)
            dist_squared = (x - coor[0])**2 + (y - coor[1])**2 + (z - coor[2])**2
            mask[dist_squared <= radius**2] = 1
                
        if masking_object['color'] == 'green':
            cmap = fluorescent_green_colormap()
        elif masking_object['color'] == 'red':
            cmap = fluorescent_red_colormap()
        elif masking_object['color'] == 'blue':
            cmap = fluorescent_blue_colormap()
        elif masking_object['color'] == 'white':
            cmap = white_colormap()
        
        colored_mask = cmap(mask / np.max(mask))[:, :, :, :3]
        overlay[..., :3] = np.where(mask[..., np.newaxis] == 1, colored_mask, overlay[..., :3])
        overlay[..., 3] = np.maximum(overlay[..., 3], mask)  # Combine alpha channels
    return overlay





def coord_to_vertex(coordinate, identity):
    trans = mriinfo[identity]['voxel-to-RAS affine']
    ras_coords = mne.transforms.apply_trans(trans, coordinate)
    src = mriinfo[identity]['source space']
    src_vertices = np.concatenate([s['rr'] for s in src])
    distances = np.linalg.norm(src_vertices - ras_coords, axis=1)
    nearest_vertex_idx = np.argmin(distances)
    if nearest_vertex_idx < len(src[0]['vertno']):
        hemisphere = 0
    else:
        hemisphere = 1
    vertex_id = src[hemisphere]['vertno'][nearest_vertex_idx % len(src[hemisphere]['vertno'])]
    return vertex_id
    









def coordinates_imaging(coord_dict, title, ax, lim='custom'):
    """
    Plots multiple sets of 3D coordinates on a single plot.

    Parameters:
    - coord_dict: A dictionary where keys are labels for the coordinates and 
                  values are tuples of (coordinates, color).
                  coordinates should be a numpy array of shape (N, 3).
                  color should be a string representing the color for the points.
    - title: Title of the plot.
    - ax: The matplotlib 3D axis to plot on.
    """
    global_min = [np.inf, np.inf, np.inf]
    global_max = [-np.inf, -np.inf, -np.inf]
    
    for label, (coords, color) in coord_dict.items():
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                   s=0.9, c=color, label=label, alpha=0.8)
        
        global_min = [min(global_min[0], coords[:, 0].min()),min(global_min[1], coords[:, 1].min()),min(global_min[2], coords[:, 2].min())]
        global_max = [max(global_max[0], coords[:, 0].max()),max(global_max[1], coords[:, 1].max()),max(global_max[2], coords[:, 2].max())]
    
    if lim != 'custom':
        global_min = [0,0,0]
        global_max = [192,256,256]

    ax.set_xlim(global_min[0], global_max[0])
    ax.set_ylim(global_min[1], global_max[1])
    ax.set_zlim(global_min[2], global_max[2])
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()





def loaddata(identity, eegpath, source_as_output=True, barycenter_as_output=True, scale_as_output=True, limit=None):
    '''
    input : identity, eegpath
        identity is the number for 'sub-0{identity}'
        eegpath is something like f'/mnt/d/dataset_for_DL/singlesource/sub-{identity:02d}/uniform/output_as_center_coordinate/eeg1.pt'
            if source == True, it returns the source coordinates.
            if 'barycenter == True, it also returns the barycenter of coordinates.
            if scale == True, it also returns the scale (radius) of dipoles.
    output : (mri, eeg_set)    where eeg_set is a list of tuples (eeg, source, barycenter, scale)
    '''
    outputs=[]
    subjects_dir = PosixPath(f'/mnt/d/openneuro_hearing_loss/sub-{identity:02d}/ses-01/anat')
    ras_mni_t = mne.transforms.read_ras_mni_t("sample", subjects_dir)  # this is RAS-to-MNI trasformation
    mripath = os.path.join(subjects_dir, f'sub-{identity:02d}_ses-01_T1w.nii.gz')
    mri = nib.load(mripath)
    affine = mri.affine #this is voxel-to-world(RAS) mapping
    mni_affine = np.dot(ras_mni_t['trans'], mri.affine) #Computing voxel-to-MNI transformation
    mri_in_mni = nib.Nifti1Image(mri.dataobj, mni_affine) #now mri is in MNI coords!
    mri_in_mni = mri_in_mni.get_fdata()
    mri_in_mni = torch.tensor(mri_in_mni, dtype=torch.float64)
    
    if limit is not None:
        eeg_set = torch.load(eegpath)['dataset'][:limit]
    else:
        eeg_set = torch.load(eegpath)['dataset'][:limit]

    for iii, (eeg, source) in enumerate(eeg_set):
        source = source.numpy() #source here is in RAS coordinate according to MNE freesurfer
        #source = nib.affines.apply_affine(ras_mni_t['trans'], source) #transform source from RAS to MNI space
        inverse_affine = np.linalg.inv(affine)
        source = nib.affines.apply_affine(inverse_affine, source) #source is in voxel coordinate system now.
        source = torch.tensor(source, dtype=torch.float64)
        barycenter = torch.mean(source, dim=0)
        distances = torch.norm(source - barycenter, dim=1)
        scale = distances.max().unsqueeze(0)
        source = torch.round(source)
        data = [eeg]
        if source_as_output is True:
            data.append(source)
        if barycenter_as_output is True:
            data.append(barycenter)
        if scale_as_output is True:
            data.append(scale)
        data = tuple(data)
        eeg_set[iii] = data
    outputs.append((mri_in_mni, eeg_set))
    return outputs

def sphere_volume(radius):
    """Calculate the volume of a sphere given its radius."""
    return (4/3) * math.pi * radius**3

def intersection_volume(R1, R2, d):
    """Calculate the volume of intersection between two spheres."""
    if d >= R1 + R2:
        # Spheres do not intersect
        return 0
    elif d <= abs(R1 - R2):
        # One sphere is completely inside the other
        return sphere_volume(min(R1, R2))
    else:
        # General case of intersection
        part1 = (R1 + R2 - d)**2
        part2 = d**2 + 2*d*(R1 + R2) - 3*(R1 - R2)**2
        return (math.pi * part1 * part2) / (12 * d)

def dice_score(center1, radius1, center2, radius2):
    """Calculate the Dice score between two spheres."""
    # Calculate the distance between the centers of the two spheres
    d = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(center1, center2)))
    
    # Calculate the volume of each sphere
    V1 = sphere_volume(radius1)
    V2 = sphere_volume(radius2)
    
    # Calculate the volume of the intersection
    V_intersection = intersection_volume(radius1, radius2, d)
    
    # Calculate the Dice score
    dice = (2 * V_intersection) / (V1 + V2)
    
    return dice



def create_skull_mask(volume_shape, outer_coords, inner_coords):
    """
    Create a skull mask from given outer and inner coordinates.

    Parameters:
    - volume_shape (tuple): Shape of the 3D MRI volume (e.g., (depth, height, width)).
    - outer_coords (list of tuples): List of coordinates representing the outer boundary of the skull.
    - inner_coords (list of tuples): List of coordinates representing the inner boundary of the skull.

    Returns:
    - mask (ndarray): A 3D binary mask with the same shape as the input volume.
    """
    # Initialize an empty mask with the same shape as the volume
    mask = np.zeros(volume_shape, dtype=np.uint8)

    # Mark the outer boundary
    for coord in outer_coords:
        z, y, x = coord
        z = int(z)
        y = int(y)
        x = int(x)
        if (x <= volume_shape[0]) & (y <= volume_shape[1]) & (z <= volume_shape[2]):
            mask[z, y, x] = 1

    # Fill the region inside the outer boundary
    filled_mask = binary_fill_holes(mask)

    # Mark the inner boundary and create holes inside
    inner_mask = np.zeros(volume_shape, dtype=np.uint8)
    for coord in inner_coords:
        z, y, x = coord
        z = int(z)
        y = int(y)
        x = int(x)
        if (x <= volume_shape[0]) & (y <= volume_shape[1]) & (z <= volume_shape[2]):
            inner_mask[z, y, x] = 1

    # Fill the region inside the inner boundary
    inner_filled_mask = binary_fill_holes(inner_mask)

    # Subtract the inner filled region from the outer filled region to get the skull mask
    skull_mask = filled_mask & ~inner_filled_mask

    return skull_mask



def mri_segmentation(mri,intensity,inner_skull_coords,outer_skull_coords,outer_skin_coords,method='only-three-layer', coordinate_system='RAS'):
    """
    Parameters
    ----------
    mri : nibabel.nifti1.Nifti1Image
    intensity : tuple
        (scalp intensity, skull intensity, brain intensity, else intensity)  : if else intensity is None, the rest is kept same as original MRI.
    inner_skull_coords : numpy.ndarray (Coordinate of inner skin boundary) Coordinate system Must match to the mri.affine
    outer_skull_coords : numpy.ndarray (Coordinate of outer skull boundary) Coordinate system Must match to the mri.affine
    outer_skin_coords : numpy.ndarray (Coordinate of outer skin boundary) Coordinate system Must match to the mri.affine
    method : string, optional
        'boundary'|'only-three-layer'|'keep-structure'. The default is 'only-three-layer'.
        if 'boundary', the boundary coordinates are highlighted.
        if 'only-three-layer', the mri will be in 4 homogeneous components, (brain,skull,skin,else)
        if 'keep-structure', the mri will be same as before but the values are multiplied by the intensity according to the components.
    coordinate_system : string, optional
        'RAS'. 
        The coordinate system of the coordinate on the surface boundary and the mri.affine must be from voxel to 'coordinate system'. They should match!!! Just keeep them "RAS"
    Returns
    -------
    mri : nibabel.nifti1.Nifti1Image
    """
    data = mri.get_fdata()
    shape = data.shape
    affine = mri.affine
    ras_to_voxel_affine = np.linalg.inv(affine)
    inner_skull_coords_voxel = nib.affines.apply_affine(ras_to_voxel_affine, inner_skull_coords)
    outer_skull_coords_voxel = nib.affines.apply_affine(ras_to_voxel_affine, outer_skull_coords)
    outer_skin_coords_voxel = nib.affines.apply_affine(ras_to_voxel_affine, outer_skin_coords)
    inner_skull_coords_voxel = np.round(inner_skull_coords_voxel)
    outer_skull_coords_voxel = np.round(outer_skull_coords_voxel)
    outer_skin_coords_voxel = np.round(outer_skin_coords_voxel)
    
    if method == 'three-layer':
        data = np.zeros_like(data)
        
        mask_inner_skin = np.zeros(shape, dtype=bool)
        for coord in inner_skull_coords_voxel:
            mask_inner_skin[int(coord[0])][int(coord[1])][int(coord[2])] = True
        mask_inner_skin = binary_fill_holes(mask_inner_skin).astype(int)
        data[mask_inner_skin] = intensity[0]
        
        mask_skull = np.zeros(shape, dtype=bool)
        for coord in outer_skull_coords_voxel:
            mask_skull[int(coord[0])][int(coord[1])][int(coord[2])] = True
        mask_skull = binary_fill_holes(mask_skull).astype(int)
        data[mask_skull] = intensity[1]
        
        mask_brain = np.zeros(shape, dtype=bool)
        for coord in inner_skull_coords_voxel:
            mask_brain[int(coord[0])][int(coord[1])][int(coord[2])] = True
        mask_brain = binary_fill_holes(mask_brain).astype(int)
        data[mask_brain] = intensity[2]
        
        if intensity[3] is not None:
            combined_mask = (mask_inner_skin | mask_skull | mask_brain) == 0
            data[combined_mask] = intensity[3]
    
    elif method == 'keep-structure':        
        mask_inner_skin = np.zeros(shape, dtype=bool)
        for coord in inner_skull_coords_voxel:
            mask_inner_skin[int(coord[0])][int(coord[1])][int(coord[2])] = True
        mask_inner_skin = binary_fill_holes(mask_inner_skin).astype(int)
        data[mask_inner_skin] = data[mask_inner_skin] * intensity[0]
        
        mask_skull = np.zeros(shape, dtype=bool)
        for coord in outer_skull_coords_voxel:
            mask_skull[int(coord[0])][int(coord[1])][int(coord[2])] = True
        mask_skull = binary_fill_holes(mask_skull).astype(int)
        data[mask_skull] = data[mask_skull] * intensity[1]
        
        mask_brain = np.zeros(shape, dtype=bool)
        for coord in inner_skull_coords_voxel:
            mask_brain[int(coord[0])][int(coord[1])][int(coord[2])] = True
        mask_brain = binary_fill_holes(mask_brain).astype(int)
        data[mask_brain] = data[mask_brain] * intensity[2]
        
        if intensity[3] is not None:
            combined_mask = (mask_inner_skin | mask_skull | mask_brain) == 0
            data[combined_mask] = data[combined_mask] * intensity[3]
            
    elif method == 'boundary':        
        mask_inner_skin = np.zeros(shape, dtype=bool)
        for coord in inner_skull_coords_voxel:
            mask_inner_skin[int(coord[0])][int(coord[1])][int(coord[2])] = True
        data[mask_inner_skin] = data[mask_inner_skin] * intensity[0]
        
        mask_skull = np.zeros(shape, dtype=bool)
        for coord in outer_skull_coords_voxel:
            mask_skull[int(coord[0])][int(coord[1])][int(coord[2])] = True
        data[mask_skull] = data[mask_skull] * intensity[1]
        
        mask_brain = np.zeros(shape, dtype=bool)
        for coord in inner_skull_coords_voxel:
            mask_brain[int(coord[0])][int(coord[1])][int(coord[2])] = True
        data[mask_brain] = data[mask_brain] * intensity[2]
        
        if intensity[3] is not None:
            combined_mask = (mask_inner_skin | mask_skull | mask_brain) == 0
            data[combined_mask] = data[combined_mask] * intensity[3]
        
    header = mri.header
    header['descrip'] = 'This is segmented mri NIfTI image.'
    
    new_mri = nib.Nifti1Image(data, affine, header)
    return new_mri


def skull(mri, inner_skull_coords, outer_skull_coords):
    data = mri.get_fdata()
    shape = data.shape    
    affine = mri.affine
    ras_to_voxel_affine = np.linalg.inv(affine)
    inner_skull_coords_voxel = np.round(nib.affines.apply_affine(ras_to_voxel_affine, inner_skull_coords))
    outer_skull_coords_voxel = np.round(nib.affines.apply_affine(ras_to_voxel_affine, outer_skull_coords))
    mask = np.zeros(shape)
    

    
def boundary_mask(mri,inner_skull_coords,outer_skull_coords,outer_skin_coords):
    data = mri.get_fdata()
    shape = data.shape
    affine = mri.affine
    ras_to_voxel_affine = np.linalg.inv(affine)
    inner_skull_coords_voxel = nib.affines.apply_affine(ras_to_voxel_affine, inner_skull_coords)
    outer_skull_coords_voxel = nib.affines.apply_affine(ras_to_voxel_affine, outer_skull_coords)
    outer_skin_coords_voxel = nib.affines.apply_affine(ras_to_voxel_affine, outer_skin_coords)
    inner_skull_coords_voxel = np.round(inner_skull_coords_voxel)
    outer_skull_coords_voxel = np.round(outer_skull_coords_voxel)
    outer_skin_coords_voxel = np.round(outer_skin_coords_voxel)
    
    mask_inner_skin = np.zeros(shape, dtype=bool)
    for coord in inner_skull_coords_voxel:
        mask_inner_skin[int(coord[0])][int(coord[1])][int(coord[2])] = True
    
    mask_skull = np.zeros(shape, dtype=bool)
    for coord in outer_skull_coords_voxel:
        mask_skull[int(coord[0])][int(coord[1])][int(coord[2])] = True
    
    mask_brain = np.zeros(shape, dtype=bool)
    for coord in inner_skull_coords_voxel:
        mask_brain[int(coord[0])][int(coord[1])][int(coord[2])] = True   
    
    scalp = mask_inner_skin & ~mask_skull
    skull = mask_skull & ~mask_brain
    brain = mask_brain
    
    return scalp, skull, brain


def bounding_box(mri):
    if type(mri) is torch.Tensor:
        non_zero_indices = torch.nonzero(mri)
        min_indices = non_zero_indices.min(dim=0).values
        max_indices = non_zero_indices.max(dim=0).values
        mri = mri[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]
    elif type(mri) is np.ndarray:
        non_zero_indices = np.argwhere(mri)
        min_indices = non_zero_indices.min(axis=0)
        max_indices = non_zero_indices.max(axis=0)
        mri = mri[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]
    return mri, min_indices
        

# Source in voxel coordinate space into binary data in MRI
def source_in_voxel_to_binary_data_in_mri(source, mri=None):
    if mri is None:
        source_in_mri = torch.zeros(192,256,256)
    else:
        if len(mri.shape)==4:
            mri = mri[0]    
        source_in_mri = torch.zeros_like(mri)
    for ss in source:
        source_in_mri[tuple(ss.to(int))] = 1
    return source_in_mri

def eeg_downsampling(eeg, kernel_size=3, stride=2, padding=1, dilation=1,type='aver', down=True):
    '''
    Parameters
    ----------
    mri : torch.tensor
    kernel_size : int or tuple
    stride : TYPE, optional The default is None.
    padding : TYPE, optional The default is 0.
    dilation : TYPE, optionalThe default is 0.
    type : string, either 'aver' or 'max' or 'min' 
    The default is 'aver'.
    Returns
    -------
    downsampled_mri
    '''
    if down is True:
        if type == 'aver':
            mm = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        elif type == 'max':
            mm = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=1)
        eeg = mm(eeg)
    else:
        print("No EEG Downsampling")
    return eeg



# Define a function to apply dropout to all linear layers
def apply_dropout(model, p=0.1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            model.add_module(f"{name}_dropout", nn.Dropout(p))



def mri_downsampling(mri, kernel_size=3, stride=2, padding=1, dilation=1,type='aver', down=True):
    '''
    Parameters
    ----------
    mri : torch.tensor
    kernel_size : int or tuple
    stride : TYPE, optional The default is None.
    padding : TYPE, optional The default is 0.
    dilation : TYPE, optionalThe default is 0.
    type : string, either 'aver' or 'max' or 'min' 
    The default is 'aver'.
    Returns
    -------
    downsampled_mri
    '''
    if type == 'aver':
        mm = nn.AvgPool3d(kernel_size, stride=stride, padding=padding)
    elif type == 'max':
        mm = nn.MaxPool3d(kernel_size, stride=stride, padding=padding, dilation=dilation)
    if mri.dim() == 3:
        mri = mri.unsqueeze(0)
        mri = mm(mri)
        mri = mri.squeeze(0)
    else:
        mri = mm(mri)
    return mri





def compute_barycenter(image):
    """Compute the barycenter of a 3D image."""
    data = image.get_fdata()
    coords = np.indices(data.shape)
    mass = data.sum()
    barycenter = np.array([np.sum(coords[i] * data) for i in range(3)]) / mass
    return barycenter

def translate_image(image, translation):
    """Translate the image by a given translation vector."""
    sitk_image = sitk.GetImageFromArray(image.get_fdata())
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(translation)
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(sitk_image)
    translated_image = resampler.Execute(sitk_image)
    return sitk.GetArrayFromImage(translated_image)


def load_nii(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine


# Determine common zero margins
def find_non_zero_bounds(data):
    non_zero_indices = np.argwhere(data != 0)
    min_bounds = np.min(non_zero_indices, axis=0)
    max_bounds = np.max(non_zero_indices, axis=0) + 1  # +1 because max index is inclusive
    return min_bounds, max_bounds


def custom_collate_fn(batch):
    # Unpack the batch
    mri, eeg, target = zip(*batch)

    # Convert tuples to lists (optional, for manipulation)
    mri = list(mri)
    eeg = list(eeg)
    target = list(target)

    mri_stacked = torch.stack(mri)
    eeg_stacked = torch.stack(eeg)
    target_stacked= torch.stack(target)

    return mri_stacked, eeg_stacked, target_stacked





def diverse_source_collate_fn(batch):
    # Unpack the batch
    mri, eeg, source = zip(*batch)

    # Convert tuples to lists (optional, for manipulation)
    mri = list(mri)
    eeg = list(eeg)
    source = list(source)

    mri_stacked = torch.stack(mri)
    eeg_stacked = torch.stack(eeg)
    padded_source = pad_sequence(source, batch_first=True, padding_value=0)
    

    """    max_size = 5 #max(c.shape[1] for c in source)
    # Initialize a padded tensor
    padded_source = torch.zeros((len(source), max_size, 3))  # Assuming the third dimension is always 3
    # Pad each 'c' tensor
    for i, c in enumerate(source):
        print(c, c.shape)
        end_dim = c.shape[1]
        padded_source[i, :end_dim, :] = c"""

    return mri_stacked, eeg_stacked, padded_source


def source_to_sourceEstimate(data, fwd, sfreq=1, subject='fsaverage', 
    simulationInfo=None, tmin=0):
    ''' Takes source data and creates mne.SourceEstimate object
    https://mne.tools/stable/generated/mne.SourceEstimate.html

    Parameters:
    -----------
    data : numpy.ndarray, shape (number of dipoles x number of timepoints)
    pth_fwd : path to the forward model files sfreq : sample frequency, needed
        if data is time-resolved (i.e. if last dim of data > 1)

    Return:
    -------
    src : mne.SourceEstimate, instance of SourceEstimate.

    '''
    
    data = np.squeeze(np.array(data))
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    
    source_model = fwd['src']
    number_of_dipoles = int(fwd['src'][0]['nuse']+fwd['src'][1]['nuse'])
    
    if data.shape[0] != number_of_dipoles:
        data = np.transpose(data)

    vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
    stc = mne.SourceEstimate(data, vertices, tmin=tmin, tstep=1/sfreq, 
        subject=subject)
    
    if simulationInfo is not None:
        setattr(stc, 'simulationInfo', simulationInfo)


    return stc


def gaussian(x,y,h=1):
    
    # I assume the space is not curved. That is, the covariance matrix is identity times scalar h.
        
    return np.exp(-np.power(x - y, 2.) / (2 * np.power(h, 2.)))#/h/((2*math.pi)**(1/2))

def robust_minmax_scaler(eeg):
    lower, upper = [torch.quantile(eeg, 0.25), torch.quantile(eeg, 0.75)]
    return (eeg-lower) / (upper-lower)

def scale_source(source):
    ''' 
        Scales the sources prior to training the neural network.

        Parameters
        ----------
        source : numpy.ndarray
            A 3D matrix of the source data (samples, dipoles, time_points)
        
        Return
        ------
        source : numpy.ndarray
            Scaled sources
        '''
    source_out = copy(source)
        # for sample in range(source.shape[0]):
        #     for time in range(source.shape[2]):
        #         # source_out[sample, :, time] /= source_out[sample, :, time].std()
        #         source_out[sample, :, time] /= np.max(np.abs(source_out[sample, :, time]))
    for sample, _ in enumerate(source):
        # source_out[sample, :, time] /= source_out[sample, :, time].std()
        source_out[sample] /= np.max(np.abs(source_out[sample]))

    return source_out
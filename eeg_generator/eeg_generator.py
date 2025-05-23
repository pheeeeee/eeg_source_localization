
import numpy as np
from pathlib import PosixPath, Path
import random
import mne
import nibabel as nib
import os
from deeplearning_code_files.utils import ras_to_voxel


class eeg_generator():
    def __init__(self,
                 subjects_dir, #path of directory with sample in it. ex)Path(f'D:\\mris_MJ\\sub-{iii:02d}')
                 montage:str,
                sfreq:int,
                spacing:str,
                conductivity,
                src=None,
                bem = None,
                info=None
                  ):
        self.subjects_dir = subjects_dir
        self.montage = montage
        self.sfreq = sfreq
        self.spacing = spacing
        self.subject = 'sample'
        self.srctype = 'surface'
        subject ='sample'
        self.montage = mne.channels.make_standard_montage(montage)
        if info is  None:
            info = mne.create_info(ch_names=self.montage.ch_names, sfreq = sfreq, ch_types='eeg')
            info.set_montage(montage)
        self.info = info
        if src is  None:
            if bem is None:
                model = mne.make_bem_model(subject=subject, ico=4, 
                                        conductivity=conductivity, subjects_dir=subjects_dir)
                bem = mne.make_bem_solution(model)
                self.bem = bem
            else:
                self.bem = bem
            src = mne.setup_source_space(subject, subjects_dir= subjects_dir, spacing = spacing)
        self.src = src
        coreg  = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials="estimated")
        coreg = coreg.fit_fiducials(verbose=True)
        coreg = coreg.fit_icp(n_iterations=100, nasion_weight=1.0, verbose=True)
        coreg = coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
        coreg = coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
        fwd = mne.make_forward_solution(info,trans=coreg.trans,src=src,bem=bem,meg=False,eeg=True,mindist=5,verbose=True)
        self.coreg = coreg
        self.fwd = fwd

    def pick_candidate_from_surface_src(self):
        src = self.src
        total_candidate = list(src[0]['rr']) + list(src[1]['rr'])
        total_candidate = np.unique(np.array(total_candidate),axis=0)
        total_candidate=total_candidate*1000
        self.candidate = total_candidate
        print("Dipole candidates picked.")

    def pick_candidate_from_mri(self, 
                                region:str #ex) 'white', 'pial'
                                ):
        subjects_dir = self.subjects_dir
        mri_dir = os.path.join(self.subjects_dir, 'sample/mri/T1.mgz')
        mri = nib.load(mri_dir)
        mri_data = mri.get_fdata()
        affine = mri.affine
        lh_white_surface_dir = os.path.join(subjects_dir, 'sample', 'surf', f'lh.{region}' )
        rh_white_surface_dir = os.path.join(subjects_dir, 'sample', 'surf', f'rh.{region}' )
        lh_vertices, lh_faces = mne.read_surface(lh_white_surface_dir)
        rh_vertices, rh_faces = mne.read_surface(rh_white_surface_dir)
        vertices = np.unique(np.vstack((lh_vertices, rh_vertices)),axis=0)
        
        # lh_pial_surface_dir = os.path.join(subjects_dir, 'sample', 'surf', 'lh.pial' )
        # rh_pial_surface_dir = os.path.join(subjects_dir, 'sample', 'surf', 'rh.pial' )
        # lh_pial_vertices, lh_pial_faces = mne.read_surface(lh_pial_surface_dir)
        # rh_pial_vertices, rh_pial_faces = mne.read_surface(rh_pial_surface_dir)
        # pial_vertices = np.unique(np.vstack((lh_pial_vertices, rh_pial_vertices)),axis=0)
        
        lh_voxel_coords = ras_to_voxel(lh_vertices, affine)
        rh_voxel_coords = ras_to_voxel(rh_vertices, affine)
        
        #Because this is for desinating all the "candidate", I want both floor/ceiling of vertice to be masked.
        lh_voxel_coords = np.floor(lh_voxel_coords).astype(int)
        rh_voxel_coords = np.floor(rh_voxel_coords).astype(int)
        voxel_coords = np.vstack((lh_voxel_coords, rh_voxel_coords))
        voxel_coords = np.unique(voxel_coords, axis=0)
        voxel_coords = np.vstack((voxel_coords, voxel_coords + np.array([1,0,0]), voxel_coords + np.array([0,1,0]) ,  voxel_coords + np.array([0,0,1]), voxel_coords + np.array([0,1,1]), voxel_coords + np.array([1,0,1]), voxel_coords + np.array([1,1,0]), voxel_coords + np.array([1,1,1])))
        voxel_coords = np.unique(voxel_coords, axis=0)

    def select_dipoles_in_ras(self, 
                      n_sources:int, 
                      extents=(21/2,58/2),  #mm
                      ):
        total_candidate = self.candidate
        selected_center_dipole = np.random.choice(np.arange(len(total_candidate)), size=n_sources, replace=False)
        selected_center_dipole = total_candidate[selected_center_dipole]
        radius = np.random.uniform(extents[0], extents[1], size=n_sources)
        self.radius = radius
        #Get all activating dipoles
        dipoles = []
        for __, dipole in enumerate(selected_center_dipole):
            for vertice in total_candidate:
                if np.linalg.norm(vertice - dipole) < radius[__]:
                    dipoles.append(vertice)
        dipoles = np.unique(np.array(dipoles), axis=0) #RAS space
        return dipoles

    def dipoles_in_src(self,                       
                    n_sources:int, 
                    extents=(21/2,58/2),  #mm
                    ):
        src = self.src
        total_candidate = self.candidate
        selected_center_dipole = np.random.choice(np.arange(len(total_candidate)), size=n_sources, replace=False)
        selected_center_dipole = total_candidate[selected_center_dipole]
        print('Center Dipoles Selected')
        radius = np.random.uniform(extents[0], extents[1], size=n_sources)
        self.radius = radius
        if self.srctype == 'surface':
            vertex_number_src = [[],[]]
            dipoles_src = []
            dipoles_vertex_coor_dis_paired_src = [[],[]]
            for hemisphere in [0,1]:
                for __, center_dipole in enumerate(selected_center_dipole):
                    for vertex_num in src[hemisphere]['vertno']:
                        vertex_coor = src[hemisphere]['rr'][vertex_num]*1000
                        d = np.linalg.norm(vertex_coor - (center_dipole)) #+ np.random.normal(0, 2, 3)))
                        if d < radius[__]:
                            vertex_number_src[hemisphere].append(vertex_num)
                            dipoles_src.append(vertex_coor)
                            dipoles_vertex_coor_dis_paired_src[hemisphere].append((vertex_num, vertex_coor, d))
                vertex_number_src[hemisphere] = np.unique(np.array(vertex_number_src[hemisphere]),axis=0)
            dipoles_src = np.unique(np.array(dipoles_src), axis=0)
            return dipoles_vertex_coor_dis_paired_src
        
    def create_labels(self,
                      n_sources:int,
                      extents=(21/2,58/2),  #mm
                      dipoles_vertex_coor_dis_paired_src=None
                      ):
        if self.srctype == 'surface':
            labels = [[],[]]
            if dipoles_vertex_coor_dis_paired_src is None:
                dipoles_vertex_coor_dis_paired_src = self.dipoles_in_src(n_sources=n_sources,extents=extents)
            for hhh, hemisphere in enumerate(['lh','rh']):
                for num, coord, d in dipoles_vertex_coor_dis_paired_src[hhh]:
                    labels[hhh].append((mne.Label(vertices=np.array([num]), hemi=hemisphere), d ))
            leftlabels = labels[0]
            rightlabels = labels[1]
            labels = leftlabels + rightlabels
            
            if len(leftlabels) > 0:
                left_combined_label = leftlabels[0][0]
                for _____ in range(len(leftlabels)):
                    left_combined_label = left_combined_label.__add__(leftlabels[_____][0])
                #left_combined_label = left_combined_label.morph(subject_from='sample', subject_to='fsaverage', 
                #                                                smooth = 20, subjects_dir=subjects_dir)
                #left_combined_label.save(os.path.join(dir_path, f'label/lh{kkkkk}.label'))
                
            if len(rightlabels) > 0:
                right_combined_label = rightlabels[0][0]
                for _____ in range(len(rightlabels)):
                    right_combined_label = right_combined_label.__add__(rightlabels[_____][0])
            return labels
    
    def create_src_simulator(self):
        tstep = 1.0 / self.info['sfreq']
        source_simulator = mne.simulation.SourceSimulator(self.src, tstep=tstep)
        self.source_simulator = source_simulator
        return source_simulator
    
    def create_raw(self,
                   labels,
                   amplitude_kernel,
                   time_course,
                   events=None,
                   ):
        if events is None:
            events = np.zeros((1, 3), int)
            events[:, 0] = self.sfreq * np.arange(1)
            events[:, 2] = 1
        
        if getattr(self, 'source_simulator', None) is None:
            source_simulator = self.create_src_simulator()
        
        if amplitude_kernel==0:
            for label, d in labels:
                source_simulator.add_data(label,time_course,events)  
        else:
            for label, d in labels:
                source_simulator.add_data(label, np.exp(-(d/10**2))*time_course, events)
        self.amplitude_kernel = amplitude_kernel
        raw = mne.simulation.simulate_raw(self.info, source_simulator, forward=self.fwd)
        raw.set_eeg_reference(projection=True)
        return raw

    def generate_eeg(self,
                 raw,
                 snr_db,
                 filter=None):
        raw1 = raw.copy()
        snr = 10**(snr_db/10)
        signal_power = np.mean(np.sum(raw.get_data()**2, axis=1))
        desired_noise_power = signal_power/snr
        cov = mne.make_ad_hoc_cov(raw.info)
        current_noise_power = np.sum(cov['data'])
        cov['data'] *= desired_noise_power/current_noise_power
        raw1 = mne.simulation.add_noise(raw1, cov=cov)

        #Add blink noise to raw data
        mne.simulation.add_eog(raw1) #Add ecg(심전도) noise to raw data
        raw1 = raw1.apply_proj()
        
        if filter is None:
            eeg = raw1.get_data(picks='eeg', tmin=0)
        else:
            raw_filter = raw1.copy()
            raw_filter = raw_filter.filter(filter[0], filter[1], filter_length='auto')
            eeg = raw_filter.get_data(picks='eeg',tmin=0)
        return eeg

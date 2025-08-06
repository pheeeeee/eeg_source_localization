
from eeg_generator.eeg_generator import eeg_generator
import nibabel as nib
from pathlib import PosixPath
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import random


n_sources = 1
extents = (21/2,58/2)
snr_db = 15
amplitude_kernel=0


# Temporal Property
sampling_rate = 500  # Hz 
duration = 1  # seconds
time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
frequency = 5  # Hz
sin_wave = np.sin(2*np.pi * frequency * time)
peak_duration = 0.1  
peak_start = int((duration - peak_duration) / 2 * sampling_rate)  
peak_end = peak_start + int(peak_duration * sampling_rate) 
output = np.zeros_like(time)  
output[peak_start:peak_end] = sin_wave[int(0.4*sampling_rate*duration):int(0.5*sampling_rate*duration)]

sinpeak = 10
output = output * sinpeak

# Plot Activation Function at Dipole
plt.figure(figsize=(10, 4))
plt.plot(time, output, label="Time Course")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time Course with 100ms 5Hz Sinusoidal Peak")
plt.axvline(peak_start / sampling_rate, color="r", linestyle="--", label="Peak Start")
plt.axvline(peak_end / sampling_rate, color="g", linestyle="--", label="Peak End")
plt.legend()
plt.grid()
plt.show()


def gaussian(x,y,h=1):
    
    # I assume the space is not curved. That is, the covariance matrix is identity times scalar h.
        
    return np.exp(-np.power(x - y, 2.) / (2 * np.power(h, 2.)))#/h/((2*math.pi)**(1/2))


n_sources = 1
extents = (21/2,58/2)
snr_db = 15
amplitude_kernel=0
amplitudes = 1


# Temporal Property
sampling_rate = 500  # Hz 
duration = 1  # seconds
time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
frequency = 5  # Hz
sin_wave = np.sin(2*np.pi * frequency * time)
peak_duration = 0.1  
peak_start = int((duration - peak_duration) / 2 * sampling_rate)  
peak_end = peak_start + int(peak_duration * sampling_rate) 
output = np.zeros_like(time)  
output[peak_start:peak_end] = sin_wave[int(0.4*sampling_rate*duration):int(0.5*sampling_rate*duration)]

sinpeak = 1
output = output * sinpeak

# Plot Activation Function at Dipole
plt.figure(figsize=(10, 4))
plt.plot(time, output, label="Time Course")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time Course with 100ms 5Hz Sinusoidal Peak")
plt.axvline(peak_start / sampling_rate, color="r", linestyle="--", label="Peak Start")
plt.axvline(peak_end / sampling_rate, color="g", linestyle="--", label="Peak End")
plt.legend()
plt.grid()
plt.show()


for amplitude_kernel in [0,1]:
    for iii in [13,12,11,10,9,8,7,6,5,4,3,2,1]:
        saving_dir = PosixPath(f'/mnt/d/REAL_FINAL_MJ/twelve_mri_20000data/snr_db{snr_db}/sub-{iii:02d}')

        #if n_sources == 1:
        #    saving_dir = PosixPath(f'/mnt/c/Users/CMME/Desktop/REALDATA_original_OPENNEURO_EEG/surface_sinusoidalpeak{sinpeak}/sub-{iii:02d}/singledipole/amplitude{amplitude_kernel}')
        #else:
        #    saving_dir = PosixPath(f'/mnt/c/Users/CMME/Desktop/REALDATA_original_OPENNEURO_EEG/surface_sinusoidalpeak{sinpeak}/sub-{iii:02d}/multidipole/amplitude{amplitude_kernel}')
        
        for hellowdoyouhearme in ['raw', 'eeg', 'center_ras', 'ras']:
            saving_dir4 = os.path.join(saving_dir, hellowdoyouhearme)
            os.makedirs(saving_dir4, exist_ok=True)
            #for snr_db in [30,20,15,10,5,1,0,-10]:
            #    saving_dir4 = os.path.join(saving_dir, f'eeg/snr{snr_db}')
            #    os.makedirs(saving_dir4, exist_ok=True)
        
        datasets=[]
        subjects_dir = PosixPath(f'/mnt/d/mris_MJ/sub-{iii:02d}')#PosixPath(f'/mnt/d/openneuro_hearing_loss/sub-{iii:02d}/ses-01/anat')  #PosixPath(f'/mnt/c/Users/CMME/Desktop/openneuro_hearing_loss_mris_transformed/centered/sub-{iii:02d}')
        #
        subject = 'sample'
        
        #If it needs BEM.
        #mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir, overwrite=True, volume='T1', atlas=False, gcaatlas=False, preflood=None, show=False, copy=True, T1=None, brainmask='ws.mgz', verbose=None)
    
        N = 2000
        spacing_training = 'ico4'
        montage_training = "standard_1020"
        sfreq = 200
        montage = 'biosemi32'
        standard_montage = mne.channels.make_standard_montage(montage)
        
        #Create info with the montage (Remind that mne.Info is the object with info about sensors and methods of measurement.)
        info = mne.create_info(ch_names=standard_montage.ch_names,sfreq=sfreq,ch_types='eeg')
        info.set_montage(standard_montage)
            
        #SourceSpace
        conductivity=(0.3, 0.06, 0.3)
        model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        #mne.write_bem_surfaces(f"/mnt/d/mris_MJ/sub-{iii:02d}/{iii:02d}bem.fif", bem, overwrite=True)
        src = mne.setup_source_space(subject, subjects_dir= subjects_dir, spacing = spacing_training)
        coreg = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials="estimated")
        coreg.fit_fiducials(verbose=True)
        coreg.fit_icp(n_iterations=100, nasion_weight=1.0, verbose=True)
        coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
        #mne.write_trans(os.path.join(subjects_dir, "trans.fif"), coreg.trans, overwrite=True) ### Save trans.fif where 'sample' is located.
        fwd = mne.make_forward_solution(info,trans=coreg.trans,src=src,bem=bem,meg=False,eeg=True)
        
        #subjects_dir = PosixPath(f'/mnt/d/mris_MJ/sub-{iii:02d}')
        #bem = mne.read_bem_solution(f"/mnt/d/mris_MJ/sub-{iii:02d}/{iii:02d}bem.fif")
        #src = mne.read_source_spaces(f"/mnt/d/mris_MJ/sub-{iii:02d}/{iii:02d}src.fif")
        #trans = mne.read_trans(f"/mnt/d/mris_MJ/sub-{iii:02d}/trans.fif")
        #fwd = mne.make_forward_solution(info,trans=trans,src=src,bem=bem,meg=False,eeg=True)
        
        for _ in range(N):
            n_sources= 1 #################################################################################################################################33333
            extents=(21,58) #mm
            
            # Parameters
            """duration = 1  # second
            total_time_points = sfreq * duration
            sine_frequency = 5  # Hz
            t = np.linspace(0, duration, total_time_points, endpoint=False)
            half_period_duration = 0.1  # 100 ms
            half_period_points = int(sfreq * half_period_duration)
            time_course[start_index:end_index] = sine_wave
            amplitudes = 1
            """
            time_course = output
            
            shape = 'uniform' #'gaussian'        #################################################################################################################################33333
            kernel_var =  None #None if 'shape is uniform. # kernel_var  #Note. variance of uniform distribution [a,b] is (b-a)**2 / 12 for 1d.
            h = None
            
        
            if _%20 == 0:
                print(f"index : {iii}, generating {_}th eeg")
            lvertices_in_use = np.array([(0,ver) for ver in src[0]['vertno']])
            rvertices_in_use = np.array([(1,ver) for ver in src[1]['vertno']])
            vertices_in_use = np.concatenate((lvertices_in_use, rvertices_in_use), axis=0)
            if isinstance(n_sources, (list, tuple, np.ndarray)):
                if len(n_sources)==1:
                    n_sources = n_sources[0]
                else:
                    n_sources = random.randint(n_sources[0], n_sources[1])
                    
            src_centers = np.random.choice(range(len(vertices_in_use)), size=n_sources, replace=False)
            src_centers = vertices_in_use[src_centers]
            n_sources = len(src_centers)
            if shape == "mixed":
                realshapes = list(random.choices(['gaussian','uniform'],k=n_sources))
            elif shape == "gaussian":
                realshapes = ["gaussian"]*n_sources
            elif shape == "uniform":
                realshapes = ["uniform"]*n_sources
            else:
                if len(shape) >= n_sources:
                    realshapes = list(shape[:n_sources])
                elif len(shape) < n_sources:
                    realshapes = [shape for shape in shape]
                    for i in range(n_sources - len(shape)):
                        realshapes.append(shape[i] )
            
            if isinstance(extents, (float, int)):
                extents = [extents]*n_sources
            elif isinstance(extents, (list, tuple, np.ndarray)) & (len(extents) == 2) & (n_sources !=2):
                extents = list(random.choices(range(min(extents),max(extents)), k=n_sources))
            elif isinstance(extents, (list, tuple, np.ndarray)):
                if len(extents) >= n_sources:
                    extents = list(extents[:n_sources])
                elif len(extents) < n_sources:
                    extents = [extent for extent in extents]
                    for i in range(n_sources - len(extents)):
                        extents.append(extents[i])
            extents = [extent_length*0.001 for extent_length in extents] #meter
            
            
            if isinstance(amplitudes, (float, int)):
                amplitudes = [amplitudes]*n_sources
            if isinstance(amplitudes, (list, tuple, np.ndarray)) & (len(amplitudes) == 2) & (n_sources !=2):
                amplitudes = random.choices(range(min(amplitudes),max(amplitudes)), k=n_sources)
            elif isinstance(amplitudes, (list, tuple, np.ndarray)):
                if len(amplitudes) >= n_sources:
                    amplitudes = list(amplitudes[:n_sources])
                elif len(amplitudes) < n_sources:
                    amplitudes = [am for am in amplitudes]
                    for i in range(n_sources - len(amplitudes)):
                        amplitudes.append(amplitudes[i])
            
            signals = [time_course]*n_sources
            tstep = 1/sfreq
            
            distance_matrix = src[0]
            source = [[],[]] #source[0] is the souce data on left hemisphere. source[1] is the source data on right hemisphere
            ##############################################
            # Loop through source centers (i.e. seeds of source positions)
            for i, (src_center, extent, shape, amplitude, signal) in enumerate(zip(src_centers, extents, realshapes, amplitudes, signals)):
                """if source_spread == "region_growing":
                    order = self.extents_to_orders(extents[i])
                    d = np.array(get_n_order_indices(order, src_center, self.neighbors))
                    # if isinstance(d, (int, float, np.int32)):
                    #     d = [d,]
                    dists = np.empty((self.pos.shape[0]))
                    dists[:] = np.inf
                    dists[d] = np.sqrt(np.sum((self.pos - self.pos[src_center, :])**2, axis=1))[d]
                else:
                    # dists = np.sqrt(np.sum((self.pos - self.pos[src_center, :])**2, axis=1))
                    dists = self.distance_matrix[src_center]
                    d = np.where(dists<extents[i]/2)[0] """
                
                center_fiff_coord = src[src_center[0]]['rr'][src_center[1]]
                
                #src_vertex , hemi = self.vertex_from_coordinate( src_center )
    
                distance_from_src_vertex0 = np.linalg.norm(src[0]['rr']-center_fiff_coord,ord=2, axis=1)
                distance_from_src_vertex1 = np.linalg.norm(src[1]['rr']-center_fiff_coord,ord=2, axis=1)
                left_vertex = np.intersect1d(np.where(distance_from_src_vertex0 < extent/2)[0], src[0]['vertno'])
                right_vertex = np.intersect1d(np.where(distance_from_src_vertex1 < extent/2)[0], src[1]['vertno'])
                
                if shape == 'gaussian':
                    maxdistance = max(max(distance_from_src_vertex0),max(distance_from_src_vertex1))
                    if h is None:
                        h = np.clip(maxdistance, a_min=0.1, a_max=np.inf)
                    #left
                    dist = distance_from_src_vertex0[left_vertex]
                    leftamplitude = np.expand_dims(gaussian(dist, 0, h) * amplitude, axis=1)
                    activity = [am*signal for am in leftamplitude]
                    source[0] = [active for active in activity]
                    lvertices = [(0,ver) for ver in left_vertex]
                    
                    #right
                    dist = distance_from_src_vertex1[right_vertex]
                    rightamplitude = np.expand_dims(gaussian(dist, 0, h) * amplitude, axis=1)
                    activity = [am*signal for am in rightamplitude]
                    source[1] = [active for active in activity]
                    rvertices = [(1,ver) for ver in right_vertex]
                            
                elif shape == 'uniform':
                    #left
                    activity = [signal*amplitude]*len(left_vertex)
                    source[0] = list(activity)
                    lvertices = np.array([(0,ver) for ver in left_vertex])
                    
                    #right
                    activity = [signal*amplitude]*len(right_vertex)
                    source[1] = list(activity)
                    rvertices = np.array([(1,ver) for ver in right_vertex])
                else:
                    msg = BaseException("shape must be of type >string< and be either >gaussian< or >uniform<.")
                    raise(msg)    
            
            if len(lvertices) == 0:
                vertices = rvertices
            elif len(rvertices) == 0:
                vertices = lvertices
            else:                
                vertices = np.concatenate((lvertices,rvertices),axis=0)
                
            dipoles_src = np.array([src[ver[0]]['rr'][ver[1]]*1000 for ver in vertices])
            
            source = np.array(source[0] + source[1])
            lvertices = [ver for _,ver in lvertices]
            rvertices = [ver for _,ver in rvertices]
            vertices_4_stc = [np.array(lvertices) , np.array(rvertices)]
            stc = mne.SourceEstimate(data=source, vertices=vertices_4_stc, tmin=0, tstep=tstep)
            
            src_center_vertices = src_centers
            src_center_coordinates = [mne.vertex_to_mni(ver, hemi, subject, subjects_dir) for hemi,ver in src_centers]
            source_vertices = vertices
            source_coordinates = [mne.vertex_to_mni(ver, hemi, subject, subjects_dir) for hemi,ver in vertices]
            
            
            raw = mne.simulation.simulate_raw(info, stc=stc, src=None, bem=None, forward = fwd, n_jobs=-1)
            raw.set_eeg_reference(projection=True)
            
            raw.save(os.path.join(saving_dir,f'raw/{_}raw.fif'),overwrite=True)
            np.save(os.path.join(saving_dir, f'center_ras/{_}.npy'), center_fiff_coord*1000)
            np.save(os.path.join(saving_dir, f'ras/{_}.npy'), dipoles_src)
            #np.save(os.path.join(saving_dir, f'vertices/{_}.npy'), np.array(vertices))
            
            
            #configured
            #trans=coreg.trans
            #center_fiff_coord = mne.transforms.apply_trans(trans['trans'], center_fiff_coord)
            #np.save(os.path.join(saving_dir, f'configured_center_dipole/{_}.npy'), center_fiff_coord*1000)
            #dipoles_src = mne.transforms.apply_trans(trans['trans'], dipoles_src)
            #np.save(os.path.join(saving_dir, f'configured_all_dipoles/{_}.npy'), dipoles_src)

            
            for snr_db in [30]:#['zeronoise',100,50,40,30,20,15,10,5,1,0,-10]:
                cov = mne.make_ad_hoc_cov(raw.info)
                raw1 = raw.copy()
                if snr_db != 'zeronoise':
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
                
                eeg = raw1.get_data(picks='eeg', tmin=0)
                np.save(os.path.join(saving_dir, f'eeg/{_}.npy'), eeg)
                
                
            
            
            
            
            

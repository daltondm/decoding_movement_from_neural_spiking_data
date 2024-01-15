#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jan 14 06:45:07 2024

@author: daltonm
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter
from importlib import sys
from pathlib import Path

from pynwb import NWBHDF5IO
import ndx_pose

code_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
data_path = code_path.parent / 'data'

sys.path.insert(0, str(code_path))
from utils import choose_units_for_model, save_dict_to_hdf5
from nwb_functions import get_sorted_units_and_apparatus_kinematics_with_metadata   

nwb_infile   = data_path / 'TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'
bad_units_list = None
mua_to_fix = []
new_tag = 'data_samples'

h5_outfile = nwb_infile.parent / f'{nwb_infile.stem}_{new_tag}.pkl' 

class params:
    fps = 150
    mua_to_fix = mua_to_fix
    spkSampWin = 0.02
    frate_thresh = 2
    snr_thresh = 3
    max_lead_time = 0.5
            
def compute_derivatives(marker_pos=None, marker_vel=None, smooth = True):
    
    if marker_pos is not None and marker_vel is None:
        marker_vel = np.diff(marker_pos, axis = -1) * params.fps
        if smooth:
            for dim in range(3):
                marker_vel[dim] = gaussian_filter(marker_vel[dim], sigma=1.5)
        
    marker_acc = np.diff(marker_vel, axis = -1) * params.fps
    if smooth:
        for dim in range(3):
            marker_acc[dim] = gaussian_filter(marker_acc[dim], sigma=1.5)
    
    return marker_vel, marker_acc 
                        
def get_spike_samples(units, spike_sample_times):

    spikes = np.zeros((units.shape[0], spike_sample_times.size), dtype='int8')

    bins = np.arange(spike_sample_times[0]-params.spkSampWin, spike_sample_times[-1]+params.spkSampWin*2, params.spkSampWin)
        
    for uIdx, unit in units.iterrows():
        unit_spikes = unit.spike_times
        spike_bins  = np.digitize(unit_spikes, bins) - 1
        spike_bins  = spike_bins[(spike_bins > -1) & (spike_bins < spike_sample_times.size)]
        bin_counts  = np.bincount(spike_bins)
        spikes[uIdx, spike_bins] = bin_counts[spike_bins]
            
    return spikes

def sample_trajectories_and_spikes(units, reaches, kin_module, nwb):
    
    first_event_key = [key for idx, key in enumerate(kin_module.data_interfaces.keys()) if idx == 0][0]
    camPeriod = np.mean(np.diff(kin_module.data_interfaces[first_event_key].pose_estimation_series['origin'].timestamps[:]))
    dlc_scorer = kin_module.data_interfaces[first_event_key].scorer    
    
    if 'simple_joints_model' in dlc_scorer:
        wrist_label = 'hand'
        shoulder_label = 'shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'TY':
        wrist_label = 'l-wrist'
        shoulder_label = 'l-shoulder'
    elif 'marmoset_model' in dlc_scorer and nwb.subject.subject_id == 'MG':
        wrist_label = 'r-wrist'
        shoulder_label = 'r-shoulder'
    
    trajSampShift = int(np.round(params.spkSampWin / camPeriod))
        
    sample_reach_idx   = []
    sample_video_event = []

    pos_sample_list = []
    spikes_list = []
    for rIdx, reach in reaches.iterrows():
        
        # get event data using container and ndx_pose names from segment_info table following form below:
        # nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
        event_data      = kin_module.data_interfaces[reach.video_event] 
        
        wrist_kinematics    = event_data.pose_estimation_series[   wrist_label].data[reach.start_idx:reach.stop_idx+1].T
        shoulder_kinematics = event_data.pose_estimation_series[shoulder_label].data[reach.start_idx:reach.stop_idx+1].T
        timestamps          = event_data.pose_estimation_series[   wrist_label].timestamps[reach.start_idx:reach.stop_idx+1]
        
        pos = wrist_kinematics - shoulder_kinematics
        vel, tmp_acc = compute_derivatives(marker_pos=pos, marker_vel=None, smooth = True)
        
        pos_samples        = pos[..., trajSampShift-1::trajSampShift]
        spike_sample_times = timestamps[::trajSampShift]
        leading_spike_sample_times = np.arange(timestamps[0] - params.max_lead_time , timestamps[0], params.spkSampWin) 
        spike_sample_times = np.insert(spike_sample_times, 0, leading_spike_sample_times)
        spikes = get_spike_samples(units, spike_sample_times)
        
        pos_sample_list.append(pos_samples.astype(np.float32))
        spikes_list.append(spikes)

    reaches['position'] = pos_sample_list
    reaches['spikes']   = spikes_list

    return reaches
    
if __name__ == "__main__":
        
    results_dict = dict()
    
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()

        reaches_key = [key for key in nwb.intervals.keys() if 'reaching_segments' in key][0]
        
        units, reaches, kin_module = get_sorted_units_and_apparatus_kinematics_with_metadata(nwb, reaches_key, mua_to_fix=params.mua_to_fix, plot=False) 
        
        units = choose_units_for_model(units, quality_key = 'snr', quality_thresh = params.snr_thresh, frate_thresh = params.frate_thresh, bad_units_list=bad_units_list)
    
        reaches = sample_trajectories_and_spikes(units, reaches, kin_module, nwb)

        reaches.to_hdf(data_path / 'movement_data.h5', 'reach_pos_and_spikes')
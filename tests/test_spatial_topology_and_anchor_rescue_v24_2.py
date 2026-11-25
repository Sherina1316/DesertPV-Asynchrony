from __future__ import annotations
import numpy as np
from pvchange.temporal_signature import enforce_spatiotemporal_topology, rescue_anchor_internal_onset_from_trajectory


def _cfg():
    return {"study":{"start_year":2011,"reference_year":2024},"anchor_internal_rescue":{"enabled":True,"minimum_component_pixels":6,"minimum_object_valid_fraction":0.5,"minimum_object_probability":0.55,"minimum_object_fraction":0.18,"minimum_object_score":0.44,"minimum_window_support_years":2,"window_length_years":3,"minimum_pixel_score":0.32,"minimum_rescue_fraction":0.20,"minimum_rescue_pixels":4},"spatial_topology":{"enabled":True,"growth_connection_radius_pixels":1,"anchor_connection_radius_pixels":1,"high_pixel_probability":0.8,"high_pixel_fraction":0.3,"historical_anchor_minimum_component_pixels":4,"historical_anchor_minimum_score":0.50,"historical_anchor_minimum_probability":0.55,"historical_anchor_maximum_confuser":0.50,"connected_growth_minimum_component_pixels":3,"connected_growth_minimum_score":0.52,"connected_growth_minimum_probability":0.65,"connected_growth_minimum_fraction":0.15,"connected_growth_maximum_confuser":0.45,"near_anchor_component_fraction":0.25,"near_anchor_minimum_onset_pixels":4,"near_anchor_new_block_minimum_pixels":8,"near_anchor_minimum_bbox_fill":0.08,"near_anchor_new_block_minimum_score":0.60,"near_anchor_new_block_minimum_probability":0.75,"near_anchor_new_block_minimum_fraction":0.30,"near_anchor_new_block_maximum_confuser":0.40,"near_anchor_high_pixel_fraction":0.40,"remote_minimum_onset_pixels":6,"remote_new_site_minimum_pixels":16,"remote_minimum_bbox_fill":0.10,"remote_new_site_minimum_score":0.66,"remote_new_site_minimum_probability":0.84,"remote_new_site_minimum_fraction":0.40,"remote_new_site_maximum_confuser":0.30,"remote_high_pixel_fraction":0.60}}


def test_anchor_internal_rescue_uses_first_supported_year_not_window_start():
    cfg=_cfg(); t,h,w=15,12,12
    onset=np.full((h,w),-1,np.int16); qualified=np.zeros((h,w),bool); anchor=np.zeros((h,w),bool)
    valid=np.ones((t,h,w),bool); p=np.full((t,h,w),0.05,np.float32); f=np.full((t,h,w),0.03,np.float32); c=np.full((h,w),0.10,np.float32)
    anchor[3:7,3:7]=True; p[3:6,3:7,3:7]=0.70; f[3:6,3:7,3:7]=0.28; p[13,3:7,3:7]=0.95; f[13,3:7,3:7]=0.55
    new_onset,new_qualified,rescue=rescue_anchor_internal_onset_from_trajectory(onset,qualified,anchor,p,f,c,valid,cfg)
    assert rescue.sum()>0 and new_qualified[4,4] and int(new_onset[4,4])==3


def test_isolated_noise_removed_and_connected_growth_retained():
    cfg=_cfg(); t,h,w=15,24,24
    onset=np.full((h,w),-1,np.int16); qualified=np.zeros((h,w),bool); anchor=np.zeros((h,w),bool); near=np.zeros((h,w),bool)
    score=np.full((h,w),0.8,np.float32); p=np.full((t,h,w),0.05,np.float32); f=np.full((t,h,w),0.03,np.float32); c=np.full((h,w),0.1,np.float32)
    onset[8:12,8:12]=5; qualified[8:12,8:12]=True; p[5:,8:12,8:12]=0.95; f[5:,8:12,8:12]=0.60
    onset[8:11,12]=6; qualified[8:11,12]=True; p[6:,8:11,12]=0.90; f[6:,8:11,12]=0.45
    onset[2,2]=9; qualified[2,2]=True; p[9:,2,2]=0.95; f[9:,2,2]=0.60
    new_onset,retained,removed=enforce_spatiotemporal_topology(onset,qualified,anchor,near,score,p,f,c,cfg)
    assert retained[9,9] and retained[9,12] and removed[2,2] and int(new_onset[2,2])==-1

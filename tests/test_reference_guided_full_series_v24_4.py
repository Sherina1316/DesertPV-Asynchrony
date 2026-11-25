from __future__ import annotations
import numpy as np
from pvchange.temporal_signature import (
    annual_active_masks_from_evidence,
    compute_anchor_reference_spectral_similarity,
    rescue_anchor_internal_onset_from_trajectory,
)

def _cfg():
    return {
        "study":{"start_year":2011,"reference_year":2024},
        "anchor_full_series_decoder":{
            "enabled":True,"probability_weight":0.20,"fraction_weight":0.20,"spectral_similarity_weight":0.50,"low_confuser_weight":0.10,
            "object_presence_threshold":0.42,"object_strong_presence_threshold":0.50,"pixel_presence_threshold":0.38,"pixel_spectral_similarity_threshold":0.45,
            "minimum_object_valid_fraction":0.30,"minimum_supported_years_in_window":2,"support_window_years":3,"maximum_internal_gap_years":1,
            "minimum_component_pixels":4,"minimum_rescue_pixels":4,"minimum_rescue_fraction":0.20,"preexisting_first_years":3,"preexisting_minimum_presence_fraction":0.67,
            "reference_similarity_temperature":2.5,"reference_scale_floor":0.01,"allow_onset_2011":True,
        },
        "active_state":{"retirement_probability_maximum":0.22,"retirement_fraction_maximum":0.12,"retirement_consecutive_valid_years":3,"reactivation_probability_minimum":0.65,"reactivation_fraction_minimum":0.20,"reactivation_consecutive_valid_years":2},
        "reference_guided_endpoint":{"positive_probability_minimum":0.55,"positive_fraction_minimum":0.15,"positive_similarity_minimum":0.45,"negative_probability_maximum":0.18,"negative_fraction_maximum":0.10,"negative_similarity_maximum":0.20,"minimum_consecutive_negative_years":3,"single_post_reference_year_cannot_be_retired":True},
    }

def test_2025_anchor_does_not_instantly_retire():
    cfg=_cfg(); t=15; h=w=5
    onset=np.full((h,w),13,dtype=np.int16); qualified=np.ones((h,w),bool)
    p=np.full((t,h,w),0.8,np.float32); f=np.full((t,h,w),0.4,np.float32); valid=np.ones((t,h,w),bool); sim=np.full((t,h,w),0.8,np.float32)
    p[14]=0.05; f[14]=0.02; sim[14]=0.05
    anchor=np.ones((h,w),bool)
    active,_,ret=annual_active_masks_from_evidence(onset,qualified,p,f,valid,cfg,anchor_reference=anchor,spectral_similarity=sim)
    assert active[13].all()
    assert active[14].all()
    assert np.all(ret==-1)

def test_full_series_recovers_2015_from_reference_similarity():
    cfg=_cfg(); t=15; h=w=8
    anchor=np.zeros((h,w),bool); anchor[2:6,2:6]=True
    valid=np.ones((t,h,w),bool)
    reflectance=np.full((t,6,h,w),0.30,np.float32)
    # 2024 PV prototype and visually similar PV beginning in 2015 (index 4)
    pv=np.asarray([0.05,0.06,0.055,0.06,0.07,0.065],np.float32)
    reflectance[13][:,anchor]=pv[:,None]
    for year in range(4,14): reflectance[year][:,anchor]=pv[:,None]
    sim=compute_anchor_reference_spectral_similarity(reflectance,anchor,valid,13,cfg)
    p=np.full((t,h,w),0.15,np.float32); f=np.full((t,h,w),0.08,np.float32); c=np.full((t,h,w),0.15,np.float32)
    p[4:14,anchor]=0.52; f[4:14,anchor]=0.18
    onset=np.full((h,w),-1,np.int16); q=np.zeros((h,w),bool)
    onset,q,rescue=rescue_anchor_internal_onset_from_trajectory(onset,q,anchor,p,f,c,valid,cfg,spectral_similarity=sim)
    assert rescue.sum()>0
    assert int(onset[3,3])==4

def test_future_classes_are_mutually_exclusive():
    from pvchange.mapping import _future_classes
    current=np.zeros((4,4),bool); current[1:3,1:3]=True
    candidate=np.zeros((4,4),bool); candidate[2:4,2:4]=True
    classes=_future_classes(current,candidate)
    assert np.all(classes[current]==1)
    assert classes[3,3]==2
    assert not np.any((classes==1)&(classes==2))

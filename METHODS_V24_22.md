# v24.22 methods: novelty-gated terminal change detection

## 1. Problem formulation

The manually interpreted 2024 vector is used as the audited reference state. The 2025 task is not merely to detect PV-like surfaces; it is to determine which surfaces became PV between 2024 and 2025. Therefore, terminal classification and temporal novelty are modelled separately.

## 2. Terminal PV-state likelihood

The terminal state combines model probability, positive-prototype similarity, LSMA PV fraction, local darkness, absolute darkness and confuser penalties. This score answers whether the 2025 surface is compatible with PV, but it cannot by itself establish new installation.

## 3. Direct terminal change

The direct change score compares 2025 with the frozen 2024 observation using:

- model-probability increase;
- PV-state-score increase;
- prototype-similarity increase;
- PV-fraction increase;
- local-darkness increase;
- brightness decrease.

The 2024→2025 comparison is kept distinct from the 2021–2023 baseline comparison so that a parcel dark for several earlier years is not described as new.

## 4. Pre-existing-state risk

Pre-existing risk combines the 2024 PV-like state, the median darkness of recent pre-terminal years, the fraction of recent years classified as persistently dark, and the 2024 model response. High risk means that the object already had its current physical appearance before 2025.

## 5. Terminal-break dominance

The 2024→2025 transition is compared with the largest positive PV-like transition observed in earlier recent years. A true new installation should produce a terminal break that is competitive with or larger than prior fluctuations. This suppresses stable land parcels with only minor year-to-year noise.

## 6. Novelty score

The novelty score integrates direct multiband change, direct brightness decrease, terminal state gain, break dominance and the complement of pre-existing-state risk. The remote-site seed requires high novelty and direct change. The remote-site extent can use weaker evidence only when connected to a valid novelty seed.

## 7. Object-level acceptance

Each remote object must satisfy:

- minimum object area;
- minimum terminal-state likelihood and darkness;
- minimum mean novelty and direct change;
- minimum break dominance;
- maximum pre-existing-state risk;
- maximum prior dark persistence;
- minimum novelty-pixel fraction;
- minimum direct-change-pixel fraction;
- geometry, solidity and confuser constraints.

The strict legacy remote pathway is also novelty-gated, preventing high model probability on a persistent non-PV parcel from bypassing the temporal contract.

## 8. Historical and ecological workflow

The 2025 novelty gate does not alter the causal rule for 2011–2024. The 2024 reference is frozen before terminal detection, and no 2025 information can modify the 2024 inventory. Earlier installation years are reconstructed from annual spectral evidence and connected growth fronts. Environmental variables support reliability adjustment, ecological attribution and future projection, but cannot create historical PV labels.

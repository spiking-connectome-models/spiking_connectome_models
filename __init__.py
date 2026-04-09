"""
Paper package: Connectome-constrained spiking olfactory pathway model.

This package contains the canonical model and analysis scripts for the paper:
"Connectome-Constrained Spiking Neural Networks Reproduce Emergent Computations
in the Larval Drosophila Olfactory Pathway"

Architecture:
    OR (rate) → ORN (LIF) → LN (LIF) → PN (LIF) → KC (2-compartment) ← APL (graded) → Decoder

Package contents:
    layers.py         - Neuron layers (LIF, TwoCompartmentKC, SpikingConnectomeLinear, etc.)
    model.py          - SpikingConnectomeConstrainedModel (full pipeline)
    run_training.py   - Canonical training (5 seeds x 300 epochs)
    analysis/         - Analysis subpackage (compute, plotting, utils)
    notebooks/        - Jupyter notebook for paper figure generation

Canonical model: All-Connections Non-AD (every synapse in Winding et al. 2023,
    including non-axodendritic contacts, gap junctions, realistic noise).
Results: results/all_connections_nonad_canonical/
"""

# Core neural circuit components
from .layers import (
    SpikingConnectomeLinear,
    SpikingAntennalLobe,
    SpikingKenyonCellLayer,
    SpikingAPLInhibition,
    LIFNeuron,
    TwoCompartmentKC,
)
# Full model pipeline and data loading
from .model import (
    SpikingConnectomeConstrainedModel,
    SpikingParams,
    load_kreher2008_data,
)
# Training/test data generation with noise
from .dataset import (
    load_kreher2008_all_odors,
    create_dataloaders,
    RepeatedOdorDataset,
)

__all__ = [
    'SpikingConnectomeLinear',
    'SpikingAntennalLobe',
    'SpikingKenyonCellLayer',
    'SpikingAPLInhibition',
    'LIFNeuron',
    'TwoCompartmentKC',
    'SpikingConnectomeConstrainedModel',
    'SpikingParams',
    'load_kreher2008_data',
    'load_kreher2008_all_odors',
    'create_dataloaders',
    'RepeatedOdorDataset',
    'analysis',
]

__version__ = '0.1.0'

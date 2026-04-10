"""
SpikingConnectomeConstrainedModel: full spiking olfactory pathway model.

Paper model: OR → ORN (LIF) → LN (LIF) → PN (LIF) → KC (2-compartment) ← APL (graded) → Decoder
With gap junctions (LN-LN, PN-PN, eLN-PN), KC-KC recurrent excitation,
Tsodyks-Markram STD on all chemical synapses, and non-AD connections.

Key features:
1. ORN, LN, PN use LIF (Leaky Integrate-and-Fire) dynamics
2. KC uses 2-compartment model (dendrite + axon) with conductance coupling
3. APL uses graded transmission (biologically appropriate for invertebrate interneurons)
4. Temporal dynamics: spike counts accumulated over simulation window
5. Biologically realistic noise (6 sources)

Biological justification for non-spiking components:
- OR responses: Receptor-mediated transduction produces graded receptor potentials
- APL: Uses graded transmission in Drosophila

References:
- Lappalainen et al. 2024: Connectome-constrained learning
- Winding et al. 2023: Larval Drosophila connectome
- Kreher et al. 2008: Larval ORN responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import pandas as pd

from .layers import (
    SpikingParams,
    LIFNeuron,
    TwoCompartmentKC,
    SpikingConnectomeLinear,
    SpikingAntennalLobe,
    SpikingKenyonCellLayer,
    SpikingAPLInhibition,
)


class ORtoORNMapping(nn.Module):
    """
    Fixed biological mapping from OR type responses to ORN neuron activations.

    Same as rate-based model: 21 OR types → 42 ORN neurons (L/R hemispheres).
    OR responses are GRADED (not spiking) - this is the transduction stage.
    """

    KREHER_OR_ORDER = [
        'Or1a', 'Or2a', 'Or7a', 'Or13a', 'Or22c', 'Or24a', 'Or30a', 'Or33b',
        'Or35a', 'Or42a', 'Or42b', 'Or45a', 'Or45b', 'Or47a', 'Or49a', 'Or59a',
        'Or67b', 'Or74a', 'Or82a', 'Or83a', 'Or85c'
    ]

    def __init__(self, n_or_types: int = 21, n_orn_neurons: int = 42):
        super().__init__()
        self.n_or_types = n_or_types
        self.n_orn_neurons = n_orn_neurons

        # Fixed binary mapping (21 ORs × 2 = 42 ORNs)
        mapping = torch.zeros(n_or_types, n_orn_neurons)
        for i in range(n_or_types):
            left_idx = 2 * i
            right_idx = 2 * i + 1
            if left_idx < n_orn_neurons:
                mapping[i, left_idx] = 1.0
            if right_idx < n_orn_neurons:
                mapping[i, right_idx] = 1.0

        self.register_buffer('mapping', mapping)

        # Learnable gain per OR type
        self.or_gains = nn.Parameter(torch.ones(n_or_types))

    def forward(self, or_responses: torch.Tensor) -> torch.Tensor:
        """Map OR responses to ORN input (still graded at this stage)."""
        scaled_responses = or_responses * F.softplus(self.or_gains)
        orn_activations = torch.matmul(scaled_responses, self.mapping)
        return orn_activations


class SpikingConnectomeConstrainedModel(nn.Module):
    """
    Complete spiking connectome-constrained olfactory pathway model.

    Architecture:
        OR (rate) → ORN (LIF) → LN (LIF) → PN (LIF) → KC (2-comp) ← APL (graded) → Decoder

    All connectivity is constrained by the Winding et al. 2023 connectome.

    Spiking components:
    - ORN: LIF neurons converting graded OR input to spikes
    - LN: LIF lateral inhibition in antennal lobe
    - PN: LIF projection neurons
    - KC: 2-compartment model (dendrite + axon) with conductance coupling

    Graded (non-spiking) components:
    - OR responses: Receptor binding kinetics (input is graded)
    - APL: Graded transmission for global inhibition (biologically validated)
    - Decoder: Rate-based readout of KC spike counts

    Learnable parameters:
    - OR gain scaling
    - Unitary synapse strengths (per connection type)
    - Neural thresholds (per population or per neuron)
    - APL gain
    - Decoder weights
    """

    def __init__(
        self,
        connectome: Dict[str, torch.Tensor],
        n_odors: int,
        n_or_types: int = 21,
        params: Optional[SpikingParams] = None,
        target_sparsity: float = 0.10,
        n_steps_al: int = 20,
        n_steps_kc: int = 20,
        surrogate_method: str = 'soft',
    ):
        """
        Args:
            connectome: Dictionary with connectivity matrices
            n_odors: Number of odor classes for decoder
            n_or_types: Number of OR types (21 larval ORs)
            params: SpikingParams (uses defaults if None)
            target_sparsity: Target KC sparsity (~10%)
            n_steps_al: Simulation steps for antennal lobe
            n_steps_kc: Simulation steps for KC layer
            surrogate_method: Gradient method - 'soft', 'superspike', or 'slayer'
        """
        super().__init__()

        self.params = params or SpikingParams()
        self.n_or_types = n_or_types
        self.n_orn = connectome['orn_to_pn'].shape[0]
        self.n_pn = connectome['orn_to_pn'].shape[1]
        self.n_ln = connectome['ln_to_pn'].shape[0]
        self.n_kc = connectome['pn_to_kc'].shape[1]
        self.n_apl = connectome['kc_to_apl'].shape[1]
        self.n_odors = n_odors
        self.target_sparsity = target_sparsity
        self.n_steps_al = n_steps_al
        self.n_steps_kc = n_steps_kc
        self.surrogate_method = surrogate_method

        # Check for KC-KC recurrent connectivity
        has_kc_kc = 'kc_to_kc_aa' in connectome

        print(f"Building SPIKING model with:")
        print(f"  OR types (input): {self.n_or_types}")
        print(f"  ORN: {self.n_orn} (LIF neurons)")
        print(f"  LN: {self.n_ln} (LIF neurons)")
        print(f"  PN: {self.n_pn} (LIF neurons)")
        print(f"  KC: {self.n_kc} (2-compartment with learnable g_soma)")
        print(f"  APL: {self.n_apl} (graded, non-spiking)")
        if has_kc_kc:
            kc_kc_aa = connectome['kc_to_kc_aa']
            n_kc_kc = int((kc_kc_aa > 0).sum().item())
            print(f"  KC-KC aa: {n_kc_kc} connections (axon→axon, excitatory)")
        print(f"  Simulation: {n_steps_al} AL steps, {n_steps_kc} KC steps")
        print(f"  Target sparsity: {target_sparsity:.1%}")
        print(f"  Surrogate gradient: {surrogate_method}")
        print(f"  Biological bounds: ENABLED (v_th, tau_m, g_soma clamped)")

        # OR → ORN mapping (graded transduction)
        self.or_to_orn = ORtoORNMapping(n_or_types, self.n_orn)

        # Spiking Antennal Lobe (with recurrent connections + non-AD)
        self.antennal_lobe = SpikingAntennalLobe(
            orn_to_pn=connectome['orn_to_pn'],
            ln_to_pn=connectome['ln_to_pn'],
            orn_to_ln=connectome.get('orn_to_ln', None),
            ln_to_ln=connectome.get('ln_to_ln', None),
            pn_to_ln=connectome.get('pn_to_ln', None),
            ln_to_orn=connectome.get('ln_to_orn', None),
            params=self.params,
            # Non-AD compartment connections
            orn_to_ln_nonad=connectome.get('orn_to_ln_nonad', None),
            ln_to_pn_nonad=connectome.get('ln_to_pn_nonad', None),
            ln_to_ln_nonad=connectome.get('ln_to_ln_nonad', None),
            pn_to_ln_nonad=connectome.get('pn_to_ln_nonad', None),
            ln_to_orn_nonad=connectome.get('ln_to_orn_nonad', None),
        )

        # Spiking KC layer - 2-compartment with learnable g_soma (biological realism)
        # All 4 KC-KC compartment connections loaded from connectome
        self.kc_layer = SpikingKenyonCellLayer(
            pn_to_kc=connectome['pn_to_kc'],
            kc_to_apl=connectome['kc_to_apl'],
            apl_to_kc=connectome['apl_to_kc'],
            params=self.params,
            target_sparsity=target_sparsity,
            use_two_compartment=True,  # 2-compartment with learnable g_soma
            surrogate_method=surrogate_method,
            kc_to_kc_aa=connectome.get('kc_to_kc_aa', None),
            kc_to_apl_da=connectome.get('kc_to_apl_da', None),
            kc_to_kc_dd=connectome.get('kc_to_kc_dd', None),
            kc_to_kc_ad=connectome.get('kc_to_kc_ad', None),
            kc_to_kc_da=connectome.get('kc_to_kc_da', None),
            pn_to_kc_nonad=connectome.get('pn_to_kc_nonad', None),
        )

        # Linear decoder: KC spike counts → odor classification
        self.decoder = nn.Linear(self.n_kc, n_odors)

    def forward(
        self,
        or_input: torch.Tensor,
        return_all: bool = False,
        al_state: Optional[Dict] = None,
        kc_state: Optional[Dict] = None,
        unified_simulation: bool = True,  # Run AL+KC together (biologically realistic)
        disable_apl: bool = False,  # For biological validation: disable APL inhibition
        apl_inject_current: float = 0.0,  # For Mancini 2023 validation: optogenetic APL activation
        kc_inject_current: float = 0.0,  # For Mancini 2023: carbachol-like KC activation
    ) -> torch.Tensor:
        """
        Forward pass through spiking olfactory pathway.

        Args:
            or_input: (batch, n_or_types) OR response pattern
            return_all: If True, return intermediate spike counts
            al_state: Previous antennal lobe state (for temporal processing)
            kc_state: Previous KC state (for temporal processing)
            unified_simulation: If True, run AL and KC simultaneously (biologically realistic)
            disable_apl: If True, skip APL inhibition (for biological validation experiments)
            apl_inject_current: Direct current injection into APL (simulates optogenetic activation)
            kc_inject_current: Direct current injection into all KCs (simulates carbachol/ACh agonist)

        Returns:
            logits: (batch, n_odors) classification logits
            (optional) dict with spike counts and states
        """
        # OR → ORN (graded input to spiking neurons)
        orn_input = self.or_to_orn(or_input)

        if unified_simulation:
            # UNIFIED SIMULATION: Run AL and KC together (like real flies!)
            # This enables proper spike-by-spike PN→KC transmission
            pn_spikes, kc_spikes, al_state_new, kc_state_new = self._unified_forward(
                orn_input, al_state, kc_state, disable_apl=disable_apl,
                apl_inject_current=apl_inject_current,
                kc_inject_current=kc_inject_current
            )
        else:
            # SEQUENTIAL SIMULATION (old behavior for backward compatibility)
            pn_spikes, al_state_new = self.antennal_lobe(
                orn_input, state=al_state, n_steps=self.n_steps_al
            )
            kc_spikes, kc_state_new = self.kc_layer(
                pn_spikes, state=kc_state, n_steps=self.n_steps_kc
            )

        # Convert spike counts to rates (normalize by timesteps)
        # This gives continuous values like rate-based model
        n_steps = max(self.n_steps_al, self.n_steps_kc)
        kc_rates = kc_spikes / n_steps

        # Decode from KC spike rates (not raw counts)
        logits = self.decoder(kc_rates)

        if return_all:
            sparsity = self.kc_layer.compute_sparsity(kc_spikes)
            return logits, {
                'orn_input': orn_input,
                'pn_spikes': pn_spikes,
                'kc_spikes': kc_spikes,
                'sparsity': sparsity,
                'al_state': al_state_new,
                'kc_state': kc_state_new,
            }
        return logits

    def _unified_forward(
        self,
        orn_input: torch.Tensor,
        al_state: Optional[Dict],
        kc_state: Optional[Dict],
        disable_apl: bool = False,
        apl_inject_current: float = 0.0,
        kc_inject_current: float = 0.0,
    ):
        """
        Run AL and KC SIMULTANEOUSLY in a unified time loop.

        This is biologically realistic: PN spikes arrive at KCs as they occur,
        enabling proper coincidence detection and temporal dynamics.

        Args:
            apl_inject_current: Direct current injection into APL activity
                (simulates optogenetic APL activation, for Mancini 2023 validation)
            kc_inject_current: Direct current injection into all KCs
                (simulates carbachol/acetylcholine agonist that broadly activates KCs)
        """
        import torch.nn.functional as F

        batch_size = orn_input.shape[0]
        device = orn_input.device
        n_steps = max(self.n_steps_al, self.n_steps_kc)

        # Initialize AL state
        if al_state is None:
            al_state = self.antennal_lobe._init_state(batch_size, device)

        v_orn = al_state['v_orn']
        v_ln = al_state['v_ln']
        v_pn = al_state['v_pn']
        refr_orn = al_state['refr_orn']
        refr_ln = al_state['refr_ln']
        refr_pn = al_state['refr_pn']
        I_orn_pn = al_state['I_orn_pn']
        I_ln_pn = al_state['I_ln_pn']
        I_orn_ln = al_state.get('I_orn_ln', torch.zeros(batch_size, self.antennal_lobe.n_ln, device=device))
        # Split LN→PN excitatory current
        I_ln_pn_excit = al_state.get('I_ln_pn_excit',
                                      torch.zeros(batch_size, self.antennal_lobe.n_pn, device=device))
        # STD vesicle states (AL synapses)
        x_std_orn_pn = al_state.get('x_std_orn_pn')
        x_std_ln_pn = al_state.get('x_std_ln_pn')
        x_std_ln_pn_excit = al_state.get('x_std_ln_pn_excit')
        x_std_orn_ln = al_state.get('x_std_orn_ln')
        # AL recurrent state
        spk_ln_prev = al_state.get('spk_ln_prev', torch.zeros(batch_size, self.antennal_lobe.n_ln, device=device))
        spk_pn_prev = al_state.get('spk_pn_prev', torch.zeros(batch_size, self.antennal_lobe.n_pn, device=device))
        # AL recurrent currents
        I_ln_ln = al_state.get('I_ln_ln')
        x_std_ln_ln = al_state.get('x_std_ln_ln')
        I_pn_ln = al_state.get('I_pn_ln')
        x_std_pn_ln = al_state.get('x_std_pn_ln')
        I_ln_orn = al_state.get('I_ln_orn')
        x_std_ln_orn = al_state.get('x_std_ln_orn')
        # Non-AD AL connection currents
        I_orn_ln_nonad = al_state.get('I_orn_ln_nonad')
        x_std_orn_ln_nonad = al_state.get('x_std_orn_ln_nonad')
        I_ln_pn_nonad = al_state.get('I_ln_pn_nonad')
        x_std_ln_pn_nonad = al_state.get('x_std_ln_pn_nonad')
        I_ln_pn_excit_nonad = al_state.get('I_ln_pn_excit_nonad')
        x_std_ln_pn_excit_nonad = al_state.get('x_std_ln_pn_excit_nonad')
        I_ln_ln_nonad = al_state.get('I_ln_ln_nonad')
        x_std_ln_ln_nonad = al_state.get('x_std_ln_ln_nonad')
        I_pn_ln_nonad = al_state.get('I_pn_ln_nonad')
        x_std_pn_ln_nonad = al_state.get('x_std_pn_ln_nonad')
        I_ln_orn_nonad = al_state.get('I_ln_orn_nonad')
        x_std_ln_orn_nonad = al_state.get('x_std_ln_orn_nonad')

        # Initialize KC state
        if kc_state is None:
            kc_state = self.kc_layer._init_state(batch_size, device)

        I_pn_kc = kc_state['I_pn_kc']
        apl_activity = kc_state['apl_activity']
        refr_kc = kc_state['refr']
        # STD vesicle states (KC synapses)
        x_std_pn_kc = kc_state.get('x_std_pn_kc')
        x_std_kc_kc_aa = kc_state.get('x_std_kc_kc_aa')
        # PN→KC non-AD state
        has_pn_kc_nonad = self.kc_layer.pn_kc_nonad is not None
        if has_pn_kc_nonad:
            I_pn_kc_nonad = kc_state.get('I_pn_kc_nonad',
                self.kc_layer.pn_kc_nonad.init_current(batch_size, device))
            x_std_pn_kc_nonad = kc_state.get('x_std_pn_kc_nonad')

        if self.kc_layer.use_two_compartment:
            v_d = kc_state['v_d']
            v_a = kc_state['v_a']
        else:
            v_kc = kc_state['v']

        # KC-KC recurrent state
        has_kc_kc = self.kc_layer.kc_kc_aa is not None
        has_kc_kc_ad = self.kc_layer.kc_kc_ad is not None
        has_kc_kc_dd = self.kc_layer.kc_kc_dd_weights is not None
        has_kc_kc_da = self.kc_layer.kc_kc_da_weights is not None
        has_any_kc_kc = has_kc_kc or has_kc_kc_ad or has_kc_kc_dd or has_kc_kc_da
        if has_any_kc_kc:
            spk_kc_prev = kc_state.get('spk_kc_prev', torch.zeros(batch_size, self.kc_layer.n_kc, device=device))
        if has_kc_kc:
            I_kc_kc_aa = kc_state['I_kc_kc_aa']
        if has_kc_kc_ad:
            I_kc_kc_ad = kc_state.get('I_kc_kc_ad', self.kc_layer.kc_kc_ad.init_current(batch_size, device))
            x_std_kc_kc_ad = kc_state.get('x_std_kc_kc_ad')

        # OR input current
        I_or = orn_input * torch.exp(self.antennal_lobe.log_or_gain)

        # Spike accumulators
        pn_spike_count = torch.zeros(batch_size, self.antennal_lobe.n_pn, device=device)
        kc_spike_count = torch.zeros(batch_size, self.kc_layer.n_kc, device=device)

        # UNIFIED TIME LOOP: AL and KC process together
        for step in range(n_steps):
            # === ANTENNAL LOBE ===
            # [Noise 5] Intrinsic ORN receptor noise (stochastic odorant-receptor binding)
            # Per-timestep multiplicative noise on OR→ORN current
            if self.antennal_lobe.orn_neurons.params.circuit_noise_enabled:
                orn_noise = torch.randn_like(I_or) * self.antennal_lobe.orn_neurons.params.orn_receptor_noise_std
                I_or_step = I_or * (1.0 + orn_noise)
            else:
                I_or_step = I_or

            # ORN dynamics (+ LN→ORN feedback from previous timestep)
            I_orn_total = I_or_step
            if self.antennal_lobe.ln_orn is not None and I_ln_orn is not None:
                if x_std_ln_orn is not None:
                    I_ln_orn, x_std_ln_orn = self.antennal_lobe.ln_orn(spk_ln_prev, I_ln_orn, x_std_ln_orn)
                else:
                    I_ln_orn = self.antennal_lobe.ln_orn(spk_ln_prev, I_ln_orn)
                I_orn_total = I_orn_total + I_ln_orn
            # LN→ORN non-AD (310 synapses, inhibitory)
            if self.antennal_lobe.ln_orn_nonad is not None and I_ln_orn_nonad is not None:
                if x_std_ln_orn_nonad is not None:
                    I_ln_orn_nonad, x_std_ln_orn_nonad = self.antennal_lobe.ln_orn_nonad(spk_ln_prev, I_ln_orn_nonad, x_std_ln_orn_nonad)
                else:
                    I_ln_orn_nonad = self.antennal_lobe.ln_orn_nonad(spk_ln_prev, I_ln_orn_nonad)
                I_orn_total = I_orn_total + I_ln_orn_nonad
            v_orn, spk_orn, refr_orn = self.antennal_lobe.orn_neurons(I_orn_total, v_orn, refr_orn)

            # --- Gap junction currents (instantaneous, voltage-dependent) ---

            # A1: LN-LN gap junctions
            I_gap_ln = torch.zeros_like(v_ln)
            if self.antennal_lobe.gap_ln_ln_mask is not None:
                g_ln = torch.exp(self.antennal_lobe.log_g_gap_ln)
                I_gap_ln = g_ln * (torch.matmul(v_ln, self.antennal_lobe.gap_ln_ln_mask)
                                    - v_ln * self.antennal_lobe.gap_ln_ln_mask.sum(1))

            # A2: PN-PN gap junctions
            g_pn_gap = torch.exp(self.antennal_lobe.log_g_gap_pn)
            I_gap_pn = g_pn_gap * (torch.matmul(v_pn, self.antennal_lobe.gap_pn_pn_mask)
                                    - v_pn * self.antennal_lobe.gap_pn_pn_mask.sum(1))

            # A3: eLN-PN gap junctions (bidirectional)
            g_eln = torch.exp(self.antennal_lobe.log_g_gap_eln_pn)
            I_gap_eln_to_pn = g_eln * (torch.matmul(v_ln, self.antennal_lobe.gap_eln_pn_mask)
                                        - v_pn * self.antennal_lobe.gap_eln_pn_mask.sum(0))
            I_gap_pn_to_eln = g_eln * (torch.matmul(v_pn, self.antennal_lobe.gap_eln_pn_mask.T)
                                        - v_ln * self.antennal_lobe.gap_eln_pn_mask.sum(1))

            # LN dynamics (ORN + LN→LN lateral + PN→LN feedback + gap junctions)
            if self.antennal_lobe.orn_ln is not None:
                if x_std_orn_ln is not None:
                    I_orn_ln, x_std_orn_ln = self.antennal_lobe.orn_ln(spk_orn, I_orn_ln, x_std_orn_ln)
                else:
                    I_orn_ln = self.antennal_lobe.orn_ln(spk_orn, I_orn_ln)
            else:
                I_orn_ln = spk_orn.mean(dim=-1, keepdim=True) * F.softplus(self.antennal_lobe.ln_pool_weight)
                I_orn_ln = I_orn_ln.expand(-1, self.antennal_lobe.n_ln)
            I_ln_total = I_orn_ln + I_gap_ln + I_gap_pn_to_eln
            # LN→LN lateral inhibition (from previous timestep's LN spikes)
            if self.antennal_lobe.ln_ln is not None and I_ln_ln is not None:
                if x_std_ln_ln is not None:
                    I_ln_ln, x_std_ln_ln = self.antennal_lobe.ln_ln(spk_ln_prev, I_ln_ln, x_std_ln_ln)
                else:
                    I_ln_ln = self.antennal_lobe.ln_ln(spk_ln_prev, I_ln_ln)
                I_ln_total = I_ln_total + I_ln_ln
            # PN→LN feedback (from previous timestep's PN spikes)
            if self.antennal_lobe.pn_ln is not None and I_pn_ln is not None:
                if x_std_pn_ln is not None:
                    I_pn_ln, x_std_pn_ln = self.antennal_lobe.pn_ln(spk_pn_prev, I_pn_ln, x_std_pn_ln)
                else:
                    I_pn_ln = self.antennal_lobe.pn_ln(spk_pn_prev, I_pn_ln)
                I_ln_total = I_ln_total + I_pn_ln
            # Non-AD contributions to LN
            if self.antennal_lobe.orn_ln_nonad is not None and I_orn_ln_nonad is not None:
                if x_std_orn_ln_nonad is not None:
                    I_orn_ln_nonad, x_std_orn_ln_nonad = self.antennal_lobe.orn_ln_nonad(spk_orn, I_orn_ln_nonad, x_std_orn_ln_nonad)
                else:
                    I_orn_ln_nonad = self.antennal_lobe.orn_ln_nonad(spk_orn, I_orn_ln_nonad)
                I_ln_total = I_ln_total + I_orn_ln_nonad
            if self.antennal_lobe.ln_ln_nonad is not None and I_ln_ln_nonad is not None:
                if x_std_ln_ln_nonad is not None:
                    I_ln_ln_nonad, x_std_ln_ln_nonad = self.antennal_lobe.ln_ln_nonad(spk_ln_prev, I_ln_ln_nonad, x_std_ln_ln_nonad)
                else:
                    I_ln_ln_nonad = self.antennal_lobe.ln_ln_nonad(spk_ln_prev, I_ln_ln_nonad)
                I_ln_total = I_ln_total + I_ln_ln_nonad
            if self.antennal_lobe.pn_ln_nonad is not None and I_pn_ln_nonad is not None:
                if x_std_pn_ln_nonad is not None:
                    I_pn_ln_nonad, x_std_pn_ln_nonad = self.antennal_lobe.pn_ln_nonad(spk_pn_prev, I_pn_ln_nonad, x_std_pn_ln_nonad)
                else:
                    I_pn_ln_nonad = self.antennal_lobe.pn_ln_nonad(spk_pn_prev, I_pn_ln_nonad)
                I_ln_total = I_ln_total + I_pn_ln_nonad
            v_ln, spk_ln, refr_ln = self.antennal_lobe.ln_neurons(I_ln_total, v_ln, refr_ln)

            # PN dynamics (with STD at ORN→PN and split LN→PN + gap junctions)
            if x_std_orn_pn is not None:
                I_orn_pn, x_std_orn_pn = self.antennal_lobe.orn_pn(spk_orn, I_orn_pn, x_std_orn_pn)
            else:
                I_orn_pn = self.antennal_lobe.orn_pn(spk_orn, I_orn_pn)
            # Inhibitory LN→PN (GABAergic Broad/Choosy LNs)
            if x_std_ln_pn is not None:
                I_ln_pn, x_std_ln_pn = self.antennal_lobe.ln_pn(spk_ln, I_ln_pn, x_std_ln_pn)
            else:
                I_ln_pn = self.antennal_lobe.ln_pn(spk_ln, I_ln_pn)
            # Excitatory LN→PN (glutamatergic Picky LNs)
            if x_std_ln_pn_excit is not None:
                I_ln_pn_excit, x_std_ln_pn_excit = self.antennal_lobe.ln_pn_excit(spk_ln, I_ln_pn_excit, x_std_ln_pn_excit)
            else:
                I_ln_pn_excit = self.antennal_lobe.ln_pn_excit(spk_ln, I_ln_pn_excit)
            I_pn_gap = I_gap_pn + I_gap_eln_to_pn
            I_pn_total = I_orn_pn + I_ln_pn + I_ln_pn_excit + I_pn_gap
            # Non-AD LN→PN contributions
            if self.antennal_lobe.ln_pn_nonad is not None and I_ln_pn_nonad is not None:
                if x_std_ln_pn_nonad is not None:
                    I_ln_pn_nonad, x_std_ln_pn_nonad = self.antennal_lobe.ln_pn_nonad(spk_ln, I_ln_pn_nonad, x_std_ln_pn_nonad)
                else:
                    I_ln_pn_nonad = self.antennal_lobe.ln_pn_nonad(spk_ln, I_ln_pn_nonad)
                I_pn_total = I_pn_total + I_ln_pn_nonad
            if self.antennal_lobe.ln_pn_excit_nonad is not None and I_ln_pn_excit_nonad is not None:
                if x_std_ln_pn_excit_nonad is not None:
                    I_ln_pn_excit_nonad, x_std_ln_pn_excit_nonad = self.antennal_lobe.ln_pn_excit_nonad(spk_ln, I_ln_pn_excit_nonad, x_std_ln_pn_excit_nonad)
                else:
                    I_ln_pn_excit_nonad = self.antennal_lobe.ln_pn_excit_nonad(spk_ln, I_ln_pn_excit_nonad)
                I_pn_total = I_pn_total + I_ln_pn_excit_nonad

            v_pn, spk_pn, refr_pn = self.antennal_lobe.pn_neurons(I_pn_total, v_pn, refr_pn)
            pn_spike_count += spk_pn

            # Track AL previous spikes for recurrent connections
            spk_ln_prev = spk_ln
            spk_pn_prev = spk_pn

            # === KENYON CELLS (receive PN spikes in REAL TIME!) ===
            # PN → KC synaptic current (spike-by-spike, not rates!)
            if x_std_pn_kc is not None:
                I_pn_kc, x_std_pn_kc = self.kc_layer.pn_kc(spk_pn, I_pn_kc, x_std_pn_kc)
            else:
                I_pn_kc = self.kc_layer.pn_kc(spk_pn, I_pn_kc)

            # APL inhibition (graded, based on running KC activity + dendritic voltage)
            # Pass KC dendritic voltage for KC dendrite → APL pathway (if 2-compartment)
            # Use DIVISIVE inhibition for biologically realistic graded suppression
            kc_v_dend = v_d if self.kc_layer.use_two_compartment else None
            apl_divisive, apl_activity = self.kc_layer.apl(
                kc_spike_count / max(1, step + 1), apl_activity, kc_v_dend=kc_v_dend,
                return_divisive=True
            )

            # Optogenetic APL current injection (for Mancini 2023 validation)
            # This adds constant current to APL activity, simulating optogenetic activation
            if apl_inject_current > 0:
                apl_activity = apl_activity + apl_inject_current
                # Recompute APL divisive factor with boosted activity (ReLU transfer)
                apl_output = F.relu(apl_activity) * F.softplus(self.kc_layer.apl.apl_gain)
                apl_divisive = torch.matmul(apl_output, self.kc_layer.apl.apl_kc_weights)

            # PN→KC non-AD current (3 synapses, negligible)
            if has_pn_kc_nonad:
                if x_std_pn_kc_nonad is not None:
                    I_pn_kc_nonad, x_std_pn_kc_nonad = self.kc_layer.pn_kc_nonad(spk_pn, I_pn_kc_nonad, x_std_pn_kc_nonad)
                else:
                    I_pn_kc_nonad = self.kc_layer.pn_kc_nonad(spk_pn, I_pn_kc_nonad)

            # Total dendritic current to KC
            # Add carbachol-like direct KC activation (if specified)
            # This simulates acetylcholine agonist that activates all KCs uniformly
            I_kc_total = I_pn_kc
            if has_pn_kc_nonad:
                I_kc_total = I_kc_total + I_pn_kc_nonad
            if kc_inject_current > 0:
                I_kc_total = I_kc_total + kc_inject_current

            # If disable_apl=True, skip APL inhibition (for biological validation)
            if disable_apl:
                I_kc = I_kc_total  # No APL inhibition
            else:
                # DIVISIVE inhibition: I_kc = I_input / (1 + apl_factor)
                # This is biologically realistic (shunting inhibition)
                # Provides GRADED suppression - all KCs reduced proportionally
                I_kc = I_kc_total / (1.0 + apl_divisive)

            # KC-KC dendrite→dendrite graded current (987 synapses, excitatory)
            # Clamp activity to prevent positive feedback runaway (bio: voltage bounded)
            # graded_conductance_scale converts voltage → current (V × nS → A)
            if has_kc_kc_dd and self.kc_layer.use_two_compartment:
                v_rest = self.kc_layer.kc_neurons.params.v_reset
                kc_dend_activity = F.relu(v_d - v_rest).clamp(max=0.030)  # Cap at 30mV depol
                I_kc_dd = torch.matmul(kc_dend_activity, self.kc_layer.kc_kc_dd_weights)
                I_kc = I_kc + self.kc_layer.graded_conductance_scale * F.softplus(self.kc_layer.kc_kc_dd_gain) * I_kc_dd

            # KC-KC axon→dendrite spike-driven current (12 synapses)
            if has_kc_kc_ad:
                if x_std_kc_kc_ad is not None:
                    I_kc_kc_ad, x_std_kc_kc_ad = self.kc_layer.kc_kc_ad(spk_kc_prev, I_kc_kc_ad, x_std_kc_kc_ad)
                else:
                    I_kc_kc_ad = self.kc_layer.kc_kc_ad(spk_kc_prev, I_kc_kc_ad)
                I_kc = I_kc + I_kc_kc_ad

            # KC-KC axon→axon recurrent current (13,621 synapses, dominant)
            I_kc_axon = None
            if has_kc_kc:
                if x_std_kc_kc_aa is not None:
                    I_kc_kc_aa, x_std_kc_kc_aa = self.kc_layer.kc_kc_aa(spk_kc_prev, I_kc_kc_aa, x_std_kc_kc_aa)
                else:
                    I_kc_kc_aa = self.kc_layer.kc_kc_aa(spk_kc_prev, I_kc_kc_aa)
                I_kc_axon = I_kc_kc_aa

            # KC-KC dendrite→axon graded current (30 synapses)
            if has_kc_kc_da and self.kc_layer.use_two_compartment:
                v_rest = self.kc_layer.kc_neurons.params.v_reset
                kc_dend_act = F.relu(v_d - v_rest).clamp(max=0.030)
                I_kc_da = torch.matmul(kc_dend_act, self.kc_layer.kc_kc_da_weights)
                I_kc_da_scaled = self.kc_layer.graded_conductance_scale * F.softplus(self.kc_layer.kc_kc_da_gain) * I_kc_da
                I_kc_axon = I_kc_da_scaled if I_kc_axon is None else I_kc_axon + I_kc_da_scaled

            # KC dynamics
            if self.kc_layer.use_two_compartment:
                v_d, v_a, spk_kc, refr_kc = self.kc_layer.kc_neurons(
                    I_kc, v_d, v_a, refr_kc, I_axon=I_kc_axon
                )
            else:
                v_kc, spk_kc, refr_kc = self.kc_layer.kc_neurons(I_kc, v_kc, refr_kc)

            kc_spike_count += spk_kc

            # Track previous spikes for KC-KC recurrence
            if has_any_kc_kc:
                spk_kc_prev = spk_kc

        # Update states
        al_state_new = {
            'v_orn': v_orn, 'v_ln': v_ln, 'v_pn': v_pn,
            'refr_orn': refr_orn, 'refr_ln': refr_ln, 'refr_pn': refr_pn,
            'I_orn_pn': I_orn_pn, 'I_ln_pn': I_ln_pn, 'I_ln_pn_excit': I_ln_pn_excit,
            'I_orn_ln': I_orn_ln,
            'x_std_orn_pn': x_std_orn_pn, 'x_std_ln_pn': x_std_ln_pn,
            'x_std_ln_pn_excit': x_std_ln_pn_excit, 'x_std_orn_ln': x_std_orn_ln,
            'spk_ln_prev': spk_ln_prev, 'spk_pn_prev': spk_pn_prev,
        }
        # AL recurrent state
        if self.antennal_lobe.ln_ln is not None:
            al_state_new['I_ln_ln'] = I_ln_ln
            al_state_new['x_std_ln_ln'] = x_std_ln_ln
        if self.antennal_lobe.pn_ln is not None:
            al_state_new['I_pn_ln'] = I_pn_ln
            al_state_new['x_std_pn_ln'] = x_std_pn_ln
        if self.antennal_lobe.ln_orn is not None:
            al_state_new['I_ln_orn'] = I_ln_orn
            al_state_new['x_std_ln_orn'] = x_std_ln_orn
        # Non-AD AL state
        if self.antennal_lobe.orn_ln_nonad is not None:
            al_state_new['I_orn_ln_nonad'] = I_orn_ln_nonad
            al_state_new['x_std_orn_ln_nonad'] = x_std_orn_ln_nonad
        if self.antennal_lobe.ln_pn_nonad is not None:
            al_state_new['I_ln_pn_nonad'] = I_ln_pn_nonad
            al_state_new['x_std_ln_pn_nonad'] = x_std_ln_pn_nonad
        if self.antennal_lobe.ln_pn_excit_nonad is not None:
            al_state_new['I_ln_pn_excit_nonad'] = I_ln_pn_excit_nonad
            al_state_new['x_std_ln_pn_excit_nonad'] = x_std_ln_pn_excit_nonad
        if self.antennal_lobe.ln_ln_nonad is not None:
            al_state_new['I_ln_ln_nonad'] = I_ln_ln_nonad
            al_state_new['x_std_ln_ln_nonad'] = x_std_ln_ln_nonad
        if self.antennal_lobe.pn_ln_nonad is not None:
            al_state_new['I_pn_ln_nonad'] = I_pn_ln_nonad
            al_state_new['x_std_pn_ln_nonad'] = x_std_pn_ln_nonad
        if self.antennal_lobe.ln_orn_nonad is not None:
            al_state_new['I_ln_orn_nonad'] = I_ln_orn_nonad
            al_state_new['x_std_ln_orn_nonad'] = x_std_ln_orn_nonad

        if self.kc_layer.use_two_compartment:
            kc_state_new = {
                'v_d': v_d, 'v_a': v_a, 'refr': refr_kc,
                'I_pn_kc': I_pn_kc, 'apl_activity': apl_activity,
                'x_std_pn_kc': x_std_pn_kc,
            }
        else:
            kc_state_new = {
                'v': v_kc, 'refr': refr_kc,
                'I_pn_kc': I_pn_kc, 'apl_activity': apl_activity,
                'x_std_pn_kc': x_std_pn_kc,
            }

        if has_any_kc_kc:
            kc_state_new['spk_kc_prev'] = spk_kc_prev
        if has_kc_kc:
            kc_state_new['I_kc_kc_aa'] = I_kc_kc_aa
            kc_state_new['x_std_kc_kc_aa'] = x_std_kc_kc_aa
        if has_kc_kc_ad:
            kc_state_new['I_kc_kc_ad'] = I_kc_kc_ad
            kc_state_new['x_std_kc_kc_ad'] = x_std_kc_kc_ad
        if has_pn_kc_nonad:
            kc_state_new['I_pn_kc_nonad'] = I_pn_kc_nonad
            kc_state_new['x_std_pn_kc_nonad'] = x_std_pn_kc_nonad

        return pn_spike_count, kc_spike_count, al_state_new, kc_state_new

    def compute_loss(
        self,
        or_input: torch.Tensor,
        odor_labels: torch.Tensor,
        sparsity_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with task and sparsity terms.

        Args:
            or_input: (batch, n_or_types) OR responses
            odor_labels: (batch,) odor class labels
            sparsity_weight: Weight for sparsity regularization

        Returns:
            total_loss: Combined loss
            metrics: Dictionary with individual terms
        """
        # Forward pass
        logits, intermediates = self.forward(or_input, return_all=True)

        # Task loss
        task_loss = F.cross_entropy(logits, odor_labels)

        # Sparsity loss (differentiable proxy)
        kc_spikes = intermediates['kc_spikes']
        n_steps = max(self.n_steps_al, self.n_steps_kc)  # Unified simulation uses this
        kc_rates = kc_spikes / n_steps
        # Sparsity = fraction of KCs with any activity (rate > 0)
        # Use soft threshold for differentiability
        soft_active = torch.sigmoid((kc_rates - 0.05) * 50.0)  # Sharp sigmoid around 5% rate
        diff_sparsity = soft_active.mean()
        sparsity_loss = (diff_sparsity - self.target_sparsity) ** 2

        # Combined loss
        total_loss = task_loss + sparsity_weight * sparsity_loss

        # Metrics
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == odor_labels).float().mean().item()

        metrics = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'sparsity': intermediates['sparsity'],
            'accuracy': accuracy,
            'pn_spike_rate': intermediates['pn_spikes'].mean().item() / n_steps,
            'kc_spike_rate': kc_spikes.mean().item() / n_steps,
        }

        return total_loss, metrics

    def clamp_to_biological_bounds(self):
        """
        Clamp all learnable parameters to biologically realistic bounds.

        This should be called after each optimizer step to ensure parameters
        remain within biologically meaningful ranges.

        Clamped parameters:
        - v_th: [-55, -30] mV for all neuron populations
        - log_tau_m: [5, 50] ms for membrane time constants
        - log_g_soma: [1, 100] nS for KC dendritic-axonal coupling
        - log_tau_apl: [5, 50] ms for APL time constant
        - log_strength: [1e-12, 1e-7] A for synaptic strengths
        """
        # Clamp antennal lobe parameters
        self.antennal_lobe.clamp_to_biological_bounds()

        # Clamp KC layer parameters
        self.kc_layer.clamp_to_biological_bounds()

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        n_odors: int,
        n_or_types: int = 21,
        include_nonad: bool = False,
        **kwargs
    ) -> 'SpikingConnectomeConstrainedModel':
        """Load model from saved connectome tensors."""
        winding_dir = data_dir / "winding2023"

        connectome = {
            'orn_to_pn': torch.load(winding_dir / "orn_to_pn.pt", weights_only=True),
            'ln_to_pn': torch.load(winding_dir / "ln_to_pn.pt", weights_only=True),
            'orn_to_ln': torch.load(winding_dir / "orn_to_ln.pt", weights_only=True),
            'pn_to_kc': torch.load(winding_dir / "pn_to_kc.pt", weights_only=True),
            'kc_to_apl': torch.load(winding_dir / "kc_to_apl.pt", weights_only=True),
            'apl_to_kc': torch.load(winding_dir / "apl_to_kc.pt", weights_only=True),
        }

        # AL recurrent connections (collapsed)
        for name in ['ln_to_ln', 'pn_to_ln', 'ln_to_orn']:
            path = winding_dir / f"{name}.pt"
            if path.exists():
                connectome[name] = torch.load(path, weights_only=True)

        # Non-AD (non-axon-dendrite) collapsed connectivity (opt-in only)
        if include_nonad:
            for name in ['orn_to_ln_nonad', 'ln_to_pn_nonad', 'ln_to_ln_nonad', 'pn_to_ln_nonad', 'pn_to_kc_nonad', 'ln_to_orn_nonad']:
                path = winding_dir / f"{name}.pt"
                if path.exists():
                    connectome[name] = torch.load(path, weights_only=True)

        # Load compartment-resolved connectivity if available
        compartment_dir = data_dir / "winding2023_compartments"
        if (compartment_dir / "kc_to_kc_aa.pt").exists():
            connectome['kc_to_kc_aa'] = torch.load(
                compartment_dir / "kc_to_kc_aa.pt", weights_only=True
            )
        # KC dendrite → APL axon (graded dendritic contribution to APL)
        if (compartment_dir / "kc_to_apl_da.pt").exists():
            connectome['kc_to_apl_da'] = torch.load(
                compartment_dir / "kc_to_apl_da.pt", weights_only=True
            )
        # KC dendrite → KC dendrite (graded dendritic coupling, 987 synapses)
        if (compartment_dir / "kc_to_kc_dd.pt").exists():
            connectome['kc_to_kc_dd'] = torch.load(
                compartment_dir / "kc_to_kc_dd.pt", weights_only=True
            )
        # KC axon → KC dendrite (12 synapses)
        if (compartment_dir / "kc_to_kc_ad.pt").exists():
            connectome['kc_to_kc_ad'] = torch.load(
                compartment_dir / "kc_to_kc_ad.pt", weights_only=True
            )
        # KC dendrite → KC axon (30 synapses)
        if (compartment_dir / "kc_to_kc_da.pt").exists():
            connectome['kc_to_kc_da'] = torch.load(
                compartment_dir / "kc_to_kc_da.pt", weights_only=True
            )

        return cls(connectome, n_odors, n_or_types=n_or_types, **kwargs)


def load_spiking_model_and_data(
    data_dir: Path,
    params: Optional[SpikingParams] = None,
) -> Tuple[SpikingConnectomeConstrainedModel, torch.Tensor, List[str]]:
    """
    Convenience function to load spiking model with OR response data.

    Args:
        data_dir: Path to connectome_models/data directory
        params: SpikingParams (optional)

    Returns:
        model: Initialized spiking model
        or_responses: (n_odors, n_or_types) OR response matrix
        odor_names: List of odor names
    """
    # Load Kreher 2008 data
    or_responses, odor_names, or_names = load_kreher2008_data(data_dir)
    n_or_types = len(or_names)
    n_odors = len(odor_names)

    print(f"Loaded Kreher 2008: {n_odors} odors × {n_or_types} OR types")

    # Create model
    model = SpikingConnectomeConstrainedModel.from_data_dir(
        data_dir, n_odors=n_odors, n_or_types=n_or_types,
        params=params,
    )

    return model, or_responses, odor_names


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SpikingConnectomeConstrainedModel")
    print("=" * 60)

    # Create dummy connectome
    n_kc = 72
    connectome = {
        'orn_to_pn': torch.randint(0, 10, (42, 21)).float(),
        'ln_to_pn': torch.randint(0, 5, (10, 21)).float(),
        'orn_to_ln': torch.randint(0, 5, (42, 10)).float(),
        'pn_to_kc': torch.randint(0, 5, (21, n_kc)).float(),
        'kc_to_apl': torch.randint(0, 3, (n_kc, 2)).float(),
        'apl_to_kc': torch.randint(0, 3, (2, n_kc)).float(),
        'kc_to_kc_aa': torch.randint(0, 3, (n_kc, n_kc)).float(),
    }

    # Create model (with KC-KC)
    model = SpikingConnectomeConstrainedModel(
        connectome, n_odors=28, n_or_types=21,
        n_steps_al=20, n_steps_kc=20,  # Reduced for testing
    )

    # Test forward pass
    batch_size = 4
    or_input = torch.randn(batch_size, 21).abs()  # 21 OR types
    logits, info = model.forward(or_input, return_all=True)

    print(f"\nForward pass:")
    print(f"  OR input shape: {or_input.shape}")
    print(f"  PN spike count shape: {info['pn_spikes'].shape}")
    print(f"  KC spike count shape: {info['kc_spikes'].shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  KC sparsity: {info['sparsity']:.2%}")

    # Test loss computation
    labels = torch.randint(0, 28, (batch_size,))
    loss, metrics = model.compute_loss(or_input, labels)

    print(f"\nLoss computation:")
    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Task loss: {metrics['task_loss']:.4f}")
    print(f"  Sparsity: {metrics['sparsity']:.2%}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  PN spike rate: {metrics['pn_spike_rate']:.4f}")
    print(f"  KC spike rate: {metrics['kc_spike_rate']:.4f}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal learnable parameters: {n_params:,}")

    print("\nSpiking model test passed!")

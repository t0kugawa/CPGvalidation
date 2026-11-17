# CPG Half-Center Bug Analysis

## Date: 2026-11-11
## Author: Kiro AI Assistant

## Problem Statement

The user has identified several critical bugs in the CPG t2xcpg implementation:

1. **Main Bug**: CPG half-center can't inhibit the parallel one - the RG_E/RG_F → In_E/In_F → RG_F/RG_E projection is not working
2. **Unplanned connections to remove**:
   - Ia_IN_E -l InE (Ia sensory input → half-center interneuron)
   - InF -l M_E (cross motor pool inhibition, should exist)
   - Ia_IN_F -l RG_E (unclear reference, may be typo)
   - IaF → Ib (Ia → Ib interneuron connections)
   - IaE → Ib (Ia → Ib interneuron connections)
3. **Ib self-safe mechanism not working**: Ib can't help with self motoneuron inhibition

## Current Implementation Analysis

### 1. Half-Center Inhibition Circuit (Currently Implemented)

From `t2xcpg/leg.py` lines 427-449:

```python
# RGE->InE (exc), InE-|RGF, RGF->InF (exc), InF-|RGE
self._connect(f"rg_e_{leg}", f"in_e_{leg}",
              weight=PARAMS["w_con_rg_e_in_e"],      # 12.0
              p=PARAMS["p_con_rg_e_in_e"])            # 0.15
self._connect(f"in_e_{leg}", f"rg_f_{leg}",
              weight=PARAMS["w_con_in_e_rg_f"] * SCALE_INHIB,  # -8.0 * SCALE_INHIB
              p=PARAMS["p_con_in_e_rg_f"])            # 0.15
self._connect(f"rg_f_{leg}", f"in_f_{leg}",
              weight=PARAMS["w_con_rg_f_in_f"],       # 18.0
              p=PARAMS["p_con_rg_f_in_f"])            # 0.30
self._connect(f"in_f_{leg}", f"rg_e_{leg}",
              weight=PARAMS["w_con_in_f_rg_e"] * SCALE_INHIB,  # -48.0 * SCALE_INHIB
              p=PARAMS["p_con_in_f_rg_e"])            # 0.30
```

**Analysis**:
- The circuit IS implemented correctly in code
- InE -| RGF: weight = -8.0 (inhibitory), p = 0.15 (WEAK connection)
- InF -| RGE: weight = -48.0 (inhibitory), p = 0.30 (STRONG connection)
- This creates an **asymmetric half-center** where F→E inhibition is 6x stronger than E→F
- The F pathway is STRONG (w=-48, p=0.30), E pathway is WEAK (w=-8, p=0.15)

**Hypothesis for Bug**:
The half-center may not be working due to:
1. The E→F pathway is TOO WEAK (w=-8.0, p=0.15 means only ~1-2 synapses in small networks)
2. SCALE_INHIB environmental variable may be reducing inhibitory weights further
3. The InE/InF interneurons themselves may have wrong parameters (excitatory vs inhibitory cell types)

### 2. Motor Pool Reciprocal Inhibition (lines 444-449)

```python
self._connect(f"in_e_{leg}", f"m_f_{leg}",
              weight=PARAMS["w_con_in_e_m_f"] * SCALE_INHIB,  # -8.0
              p=PARAMS["p_con_in_e_m_f"])            # 0.15
self._connect(f"in_f_{leg}", f"m_e_{leg}",
              weight=PARAMS["w_con_in_f_m_e"] * SCALE_INHIB,  # -48.0
              p=PARAMS["p_con_in_f_m_e"])            # 0.30
```

**Analysis**:
- These connections ARE planned and documented as necessary
- Comment states: "Without this, RGE/RGF reciprocal inhibition never reaches the M-E/M-F output and both muscles co-contract"
- Test `test_cpg_topology.py` line 21-22 explicitly validates these exist
- **USER MAY BE MISTAKEN** - these should NOT be removed

### 3. Ia → In Connections (lines 502-507)

```python
self._connect(f"ia_e_in_{leg}", f"in_e_{leg}",
              weight=PARAMS["w_con_ia_e_in_in_e"],   # 6.0 (EXCITATORY!)
              p=PARAMS["p_con_ia_e_in_in_e"])        # 0.25
self._connect(f"ia_f_in_{leg}", f"in_f_{leg}",
              weight=PARAMS["w_con_ia_f_in_in_f"],   # 6.0 (EXCITATORY!)
              p=PARAMS["p_con_ia_f_in_in_f"])        # 0.25
```

**Analysis**:
- These are **UNPLANNED** connections
- Ia sensory input is exciting the CPG half-center interneurons (InE/InF)
- This would modulate the half-center rhythm based on muscle stretch
- Likely interferes with the clean half-center oscillation
- **SHOULD BE REMOVED** as user requested

### 4. Ia → Ib Interneuron Connections (lines 547-552)

```python
self._connect(f"ia_e_in_{leg}", f"in_ib_e_{leg}",
              weight=PARAMS["w_con_ia_e_in_in_ib_e"],  # 8.0
              p=PARAMS["p_con_ia_e_in_in_ib_e"])       # 0.30
self._connect(f"ia_f_in_{leg}", f"in_ib_f_{leg}",
              weight=PARAMS["w_con_ia_f_in_in_ib_f"],  # 8.0
              p=PARAMS["p_con_ia_f_in_in_ib_f"])       # 0.30
```

**Analysis**:
- These connections make Ia spindle afferents excite the Ib interneurons
- This creates cross-talk between two separate reflex pathways
- Biologically questionable: Ia (length) shouldn't drive Ib (force) interneurons
- **SHOULD BE REMOVED** as user requested

### 5. Ib Self-Inhibition Mechanism (lines 522-545)

Current implementation:
```python
# Extensor Ib: GTO → InIb-E → {M-E inh, M-F exc, RGF inh}
self._connect(f"ib_e_in_{leg}", f"in_ib_e_{leg}",
              weight=PARAMS["w_con_ib_e_in_in_ib_e"],  # 16.0
              p=PARAMS["p_con_ib_e_in_in_ib_e"])       # 0.40
self._connect(f"in_ib_e_{leg}", f"m_e_{leg}",
              weight=PARAMS["w_con_in_ib_e_m_e"],      # -18.0 (INHIBITORY)
              p=PARAMS["p_con_in_ib_e_m_e"])           # 0.40  -> AUTOGENIC INHIBITION
self._connect(f"in_ib_e_{leg}", f"m_f_{leg}",
              weight=PARAMS["w_con_in_ib_e_m_f"] * SCALE_IB,  # 12.0 (EXCITATORY)
              p=PARAMS["p_con_in_ib_e_m_f"])           # 0.35
self._connect(f"in_ib_e_{leg}", f"rg_f_{leg}",
              weight=PARAMS["w_con_in_ib_e_rg_f"] * SCALE_IB, # 18.0 (EXCITATORY)
              p=PARAMS["p_con_in_ib_e_rg_f"])          # 0.35
```

**Analysis**:
- The autogenic inhibition (InIb-E -| M-E) IS implemented with weight=-18.0
- Test `test_ib_halfcentre_sign.py` line 34 validates this is inhibitory
- **The mechanism EXISTS** - user may be seeing it fail functionally
- Possible issues:
  1. Weight too weak relative to excitatory drives
  2. Connection probability (p=0.40) creates too few synapses
  3. Ib relay rates are too low (hz_ib = 5.0)

## Critical Questions to Investigate

1. **What is SCALE_INHIB set to?** This environmental variable scales all inhibitory weights
2. **What are the InE/InF neuron parameters?** Are they correctly configured as fast-spiking inhibitory?
3. **What are network sizes?** In small networks (n_rg=20, n_int=15), p=0.15 creates ~2 synapses
4. **What is "Ia_IN_F -l RG_E"?** This doesn't match any existing connection. Could be:
   - Typo for "Ia_IN_F -| M_E" (reciprocal Ia inhibition, which exists)
   - Typo for "in_f_L_rg_e_L" (half-center, which exists)

## Connections Summary

### Currently Implemented:
| Source | Target | Weight | Probability | Purpose | Status |
|--------|--------|--------|-------------|---------|--------|
| rg_e | in_e | 12.0 | 0.15 | Half-center E | ✓ Implemented |
| in_e | rg_f | -8.0 | 0.15 | Half-center E→F inhibition | ⚠️ TOO WEAK? |
| rg_f | in_f | 18.0 | 0.30 | Half-center F | ✓ Implemented |
| in_f | rg_e | -48.0 | 0.30 | Half-center F→E inhibition | ✓ Strong |
| in_e | m_f | -8.0 | 0.15 | Motor pool reciprocal | ✓ Needed |
| in_f | m_e | -48.0 | 0.30 | Motor pool reciprocal | ✓ Needed |
| ia_e_in | in_e | 6.0 | 0.25 | **UNPLANNED** | ❌ REMOVE |
| ia_f_in | in_f | 6.0 | 0.25 | **UNPLANNED** | ❌ REMOVE |
| ia_e_in | in_ib_e | 8.0 | 0.30 | **UNPLANNED** | ❌ REMOVE |
| ia_f_in | in_ib_f | 8.0 | 0.30 | **UNPLANNED** | ❌ REMOVE |
| in_ib_e | m_e | -18.0 | 0.40 | Ib autogenic inhibition | ✓ Works? |
| in_ib_f | m_f | -18.0 | 0.40 | Ib autogenic inhibition | ✓ Works? |

## Proposed Fix Strategy

### Phase 1: Remove Unplanned Connections (TDD)
1. Write test to verify connections DON'T exist:
   - `ia_e_in_{leg}` → `in_e_{leg}` (REMOVE)
   - `ia_f_in_{leg}` → `in_f_{leg}` (REMOVE)
   - `ia_e_in_{leg}` → `in_ib_e_{leg}` (REMOVE)
   - `ia_f_in_{leg}` → `in_ib_f_{leg}` (REMOVE)

2. Remove the connection calls from `leg.py`

3. Update parameter file to comment out unused parameters

### Phase 2: Diagnose Half-Center Weakness
1. Write test to measure actual synapse counts in the half-center
2. Test with different network sizes
3. Test effect of SCALE_INHIB
4. Verify InE/InF are inhibitory cell types

### Phase 3: Strengthen Half-Center if Needed
Possible interventions (pending diagnosis):
- Increase w_con_in_e_rg_f from -8.0 to -16.0 or -24.0
- Increase p_con_in_e_rg_f from 0.15 to 0.25 or 0.30
- Add intrinsic excitability to InE/InF
- Verify SCALE_INHIB is not reducing weights

### Phase 4: Verify Ib Mechanism
1. Write test for Ib autogenic inhibition strength
2. Check if weight=-18.0 is sufficient
3. Verify Ib relay rates are appropriate
4. Test functional behavior under load

## Next Steps

1. **Clarify with user**: 
   - What is "Ia_IN_F -l RG_E"? (doesn't exist in codebase)
   - Confirm that "InF -l M_E" should NOT be removed (it's documented as necessary)
   - What evidence shows the half-center isn't working? (test failure? simulation behavior?)

2. **Create failing tests first (TDD)**:
   - Test that unplanned connections don't exist
   - Test that half-center produces alternation
   - Test that Ib prevents motor pool saturation

3. **Make minimal fixes**:
   - Remove only the confirmed unplanned connections
   - Adjust half-center weights only if tests prove insufficiency
   - Keep all changes in t2xcpg module (no registry changes)

4. **Validate**:
   - All existing tests pass
   - New tests pass
   - Half-center produces stable oscillation
   - Motor pools alternate properly

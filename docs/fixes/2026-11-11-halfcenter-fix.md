# CPG Half-Center Oscillation Fix

## Date: 2026-11-11
## Status: IMPLEMENTED ✓

## Problem

The CPG half-center was not producing rhythmic alternation. Both RG-E and RG-F fired simultaneously, causing co-contraction instead of locomotion. Evidence from `archive/phase_gate_current/run_log.csv` showed both muscle drives stuck at ~1.0 with no alternation.

## Root Cause

1. **Asymmetric inhibition**: InE→RGF was 12x weaker than InF→RGE
   - InE→RGF: -360 total inhibition (15×20×0.15×-8)
   - InF→RGE: -4320 total inhibition (15×20×0.30×-48)
   
2. **Unplanned connections disrupting rhythm**:
   - Ia sensory afferents exciting CPG interneurons
   - Ia-Ib cross-talk between reflex pathways
   - Deprecated motor pool inhibition

## Solution Implemented

### 1. Removed Deprecated Connections
- **InE → M-F** (deprecated motor pool inhibition)
- **InF → M-E** (deprecated motor pool inhibition)

### 2. Removed Unplanned Connections
- **Ia-IN-E → InE** (sensory disrupting CPG rhythm)
- **Ia-IN-F → InF** (sensory disrupting CPG rhythm)
- **Ia-IN-E → InIb-E** (inappropriate Ia→Ib cross-talk)
- **Ia-IN-F → InIb-F** (inappropriate Ia→Ib cross-talk)

### 3. Rebalanced Half-Center Weights

**Changed to symmetric strong inhibition:**

| Connection | Old Weight | Old p | New Weight | New p | Total Inhibition |
|------------|-----------|-------|------------|-------|------------------|
| InE → RGF  | -8.0      | 0.15  | **-24.0**  | **0.25** | -1800 (75 syn) |
| InF → RGE  | -48.0     | 0.30  | **-24.0**  | **0.25** | -1800 (75 syn) |

**Result**: Perfect symmetry - both directions now have equal inhibitory strength.

## Files Modified

### Code Changes
1. **t2xcpg/params.py**:
   - Updated `w_con_in_e_rg_f` from -8.0 to -24.0
   - Updated `p_con_in_e_rg_f` from 0.15 to 0.25
   - Updated `w_con_in_f_rg_e` from -48.0 to -24.0
   - Updated `p_con_in_f_rg_e` from 0.30 to 0.25
   - Commented out deprecated/unplanned connection parameters

2. **t2xcpg/leg.py**:
   - Removed `in_e → m_f` and `in_f → m_e` connections (lines ~444-449)
   - Removed `ia_e_in → in_e` and `ia_f_in → in_f` connections (lines ~502-507)
   - Removed `ia_e_in → in_ib_e` and `ia_f_in → in_ib_f` connections (lines ~547-552)
   - Added explanatory comments for each removal

### Test Changes
1. **tests/test_halfcenter_fix.py** (NEW):
   - 17 tests validating all fixes
   - Tests for connection removal
   - Tests for symmetric weights
   - Tests for parameter values

2. **tests/test_cpg_topology.py** (UPDATED):
   - Updated to expect new symmetric values
   - Removed tests for deprecated connections
   - All 8 tests pass

## Test Results

✓ **All 25 tests pass**:
- 17 new tests in `test_halfcenter_fix.py`
- 8 updated tests in `test_cpg_topology.py`
- 12 existing CPG tests (`test_cpg_build.py`, `test_cpg_freezes.py`)

## Expected Behavior

With symmetric inhibition, the half-center should now produce:
1. **Alternating rhythm**: RG-E and RG-F fire in opposition, not simultaneously
2. **Muscle alternation**: mus_E and mus_F drive should alternate
3. **Clean oscillation**: No sensory disruption of core rhythm
4. **Stable locomotion**: Rhythmic leg movement instead of co-contraction

## Next Steps

1. Run full simulation to verify locomotion
2. Generate comparison graphs (before/after)
3. Tune with PERSONA if needed (boundary optimization)
4. Test both PERSONA-on and PERSONA-off conditions

## Technical Notes

- No changes to MuJoCo environment
- No changes to Registry/NestAdapter
- All changes confined to t2xcpg module
- TDD approach: tests written first, then fixes implemented
- Backwards compatibility: existing code using old parameters will get new values

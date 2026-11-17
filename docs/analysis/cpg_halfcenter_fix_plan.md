# CPG Half-Center Fix Plan

## Date: 2026-11-11
## Status: APPROVED - Ready for TDD Implementation

## Evidence of Bug

From `archive/phase_gate_current/run_log.csv` - both RG_E and RG_F fire together, no alternation.

## Root Cause

1. InE → RGF: weight=-8.0, p=0.15 (WEAK, only ~45 synapses, total=-360)
2. InF → RGE: weight=-48.0, p=0.30 (STRONG, ~90 synapses, total=-4320)
3. Ratio: F→E is **12x stronger** - creates extreme asymmetry
4. Unplanned connections disrupt the rhythm

## Fix Strategy

### Phase 1: Remove Bad Connections (TDD)

Delete from `t2xcpg/leg.py`:
1. Lines 444-449: `in_e → m_f` and `in_f → m_e` (DEPRECATED motor pool inhibition)
2. Lines 502-507: `ia_e_in → in_e` and `ia_f_in → in_f` (UNPLANNED sensory→CPG)
3. Lines 547-552: `ia_e_in → in_ib_e` and `ia_f_in → in_ib_f` (UNPLANNED Ia→Ib)

### Phase 2: Rebalance Half-Center Weights

**Proposed: Symmetric Strong Inhibition**
- InE → RGF: weight = **-24.0**, p = **0.25** (was -8.0, 0.15)
- InF → RGE: weight = **-24.0**, p = **0.25** (was -48.0, 0.30)

Expected synapses: 15 * 20 * 0.25 = 75 each direction
Total inhibition: 75 * (-24) = -1800 each way (symmetric!)

Update in `t2xcpg/params.py`:
```python
"w_con_in_e_rg_f": -24.0,  # was -8.0
"p_con_in_e_rg_f": 0.25,   # was 0.15
"w_con_in_f_rg_e": -24.0,  # was -48.0
"p_con_in_f_rg_e": 0.25,   # was 0.30
```

### Phase 3: Test and Validate

Tests to write:
1. `test_halfcenter_connections_removed.py` - verify bad connections gone
2. `test_halfcenter_symmetric.py` - verify weights are balanced
3. `test_halfcenter_oscillation.py` - verify alternation in simulation

Success criteria:
- RG-E and RG-F alternate (not simultaneous)
- Muscle drives alternate (not co-contract)
- Stable rhythm over 2+ seconds

## Implementation Order

1. Write failing tests
2. Remove deprecated connections
3. Update parameter values
4. Run tests - all pass
5. Run full simulation - check locomotion
6. Create comparison graphs (before/after)

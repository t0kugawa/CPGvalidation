# CPG Weight Tuning for Stable Locomotion

## Date: 2026-11-11  
## Status: IN PROGRESS

## Problem
After fixing the half-center topology, simulation showed:
- RG-E pools nearly silent (mean ≈ 0.09)
- Flexor dominance despite symmetric inhibition
- No stable alternating rhythm

## Root Causes Identified

1. **Imbalanced Self-Excitation**
   - RG-F: 5.5 (37% stronger)
   - RG-E: 4.0
   
2. **Insufficient Drive**
   - BS→RG: 2.0 (too weak to overcome inhibition)
   - BASE→RG: 1.0 (minimal tonic input)
   
3. **Asymmetric RG→In Drives**
   - RG-E→InE: 12.0
   - RG-F→InF: 18.0 (50% stronger)
   
4. **Intrinsic Imbalance**
   - RG-F bursts easily (Izh c=-55)
   - RG-E regular spiking (c=-65)

## Tuning Strategy

### Phase 1: Balance Core Oscillator

**RG Self-Recurrence** (symmetric):
- RG-E: 4.0 → **5.0**
- RG-F: 5.5 → **5.0**

**RG→In Drives** (symmetric):
- RG-E→InE: 12.0 → **15.0**
- RG-F→InF: 18.0 → **15.0**

**Half-Center Inhibition** (slightly reduced):
- InE→RG-F: -24.0 → **-20.0**
- InF→RG-E: -24.0 → **-20.0**
- *Rationale*: Slightly weaker to allow oscillation to start

**Brainstem Drive** (significantly increased):
- BS→RG: 2.0 → **6.0** (3x boost)
- BASE→RG: 1.0 → **2.0** (2x boost)
- *Rationale*: Must overcome inhibition and drive rhythm

## Expected Outcome

With these changes:
1. **RG-E and RG-F equally excitable** (self-recurrence 5.0)
2. **Symmetric drives to interneurons** (both 15.0)
3. **Strong brainstem command** (6.0 weight)
4. **Balanced inhibition** (-20.0 both ways)

This should produce:
- ✓ RG-E ↔ RG-F alternation
- ✓ Both pools active (not silent)
- ✓ Stable rhythm frequency
- ✓ Proper muscle drives

## Bio-Plausibility Check

| Parameter | Old | New | Ratio | Plausible? |
|-----------|-----|-----|-------|------------|
| BS→RG | 2.0 | 6.0 | 3x | ✓ Within 1-20 range |
| RG-E self | 4.0 | 5.0 | 1.25x | ✓ Moderate increase |
| InE→RGF | -24.0 | -20.0 | 0.83x | ✓ Still strong inhibition |
| RG-E→InE | 12.0 | 15.0 | 1.25x | ✓ Modest boost |

All changes are bio-plausible and within reasonable physiological ranges.

## Files Modified
- `t2xcpg/params.py`: Updated 8 weight parameters

## Next Test
Run 10s simulation to verify alternation.

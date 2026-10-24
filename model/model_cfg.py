OBJECT_MASS: float = 300.0
SIM_MS: float = 60000.0

# Treadmill/ground seating. The belt plane is placed this far (m) ABOVE the
# lowest surface point of the limb's foot at the reset pose. A small positive
# value (a slight penetration) guarantees a resting contact is generated;
# a positive *gap* (plane below the foot) suppresses contact entirely, which
# leaves the limb in permanent "swing" (no CUT/stance signal).
TREADMILL_PENETRATION_M: float = 0.0005

# A target-limb foot used to be considered in stance when its lowest surface
# was within this distance (m) of the belt. Retired: the target limb now
# collides with the belt and its phase comes from the actual contacts.
CONTACT_Z_EPS: float = 0.003

# Physical ankle range in this model's hinge convention. The OpenSim
# reference ankle_flx coordinate runs [-0.52359878, 1.04719755] rad about a
# default of 0.87266463 rad; with the MuJoCo neutral near 1.59 rad the
# equivalent range is about [0.19, 1.76]. This is a plant limit, not a PERSONA
# parameter setpoint.
ANKLE_RANGE_RAD: tuple[float, float] = (0.19, 1.76)


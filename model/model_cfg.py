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

# Ankle guard: the joint range is narrowed to the mocap ankle envelope of the
# run window plus this margin (rad), so a low muscle drive or a contact torque
# cannot hyper-plantarflex the paw past a physiological stop.
ANKLE_RANGE_MARGIN_RAD: float = 0.10


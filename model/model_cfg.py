OBJECT_MASS: float = 300.0
SIM_MS: float = 60000.0

# Treadmill/ground seating. The belt plane is placed this far (m) ABOVE the
# lowest surface point of the limb's foot at the reset pose. A small positive
# value (a slight penetration) guarantees a resting contact is generated;
# a positive *gap* (plane below the foot) suppresses contact entirely, which
# leaves the limb in permanent "swing" (no CUT/stance signal).
TREADMILL_PENETRATION_M: float = 0.0005

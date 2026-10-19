"""The CPG -> MuJoCo muscle drive must be graded, and the belt must seat the
CONTROLLED foot.

Both were measured breaking the closed loop:
- the drive was read from the muscle parrot relay, whose spike count collapses
  coincident input to one spike per neuron per ms, so it saturated at n_mn as
  soon as the motor pool fired (at n_mn=100 the drive only ever took the values
  0.0, 0.81 and 1.0, and it was pinned at 1.0 for a whole run);
- the belt was levelled on the target limb's foot, which is a non-colliding
  mirror that sits below the controlled foot, so the controlled foot had air
  under it and never generated the CUT/stance contact.
"""
import numpy as np

from persona.registry import Registry
from t2xcpg.leg import Leg as CPG_Leg


class _FakeRecorder:
    def __init__(self, n_events: int):
        self._n = n_events
        self.cleared = False

    def get(self, key):
        assert key == "events"
        return {"times": np.arange(self._n)}

    def set(self, d):
        if d.get("n_events") == 0:
            self._n = 0
            self.cleared = True


def test_drive_is_linear_in_the_pool_spike_count(nest_kernel):
    leg = CPG_Leg("L", Registry(), n_rg=8, n_mn=10, n_int=6, n_aff=6, ref_hz=100.0)
    # ref_hz=100, dt=10 ms -> a pool of 10 neurons at 100 Hz is full drive.
    assert np.isclose(leg._drive(10, 10.0, 10), 1.0)
    assert np.isclose(leg._drive(5, 10.0, 10), 0.5)
    assert np.isclose(leg._drive(1, 10.0, 10), 0.1)
    assert np.isclose(leg._drive(0, 10.0, 10), 0.0)


def test_muscle_drive_reads_the_motor_pool_and_stays_graded(nest_kernel):
    leg = CPG_Leg("L", Registry(), n_rg=8, n_mn=10, n_int=6, n_aff=6, ref_hz=100.0)
    # 4 motor spikes out of 10 neurons is a 40 Hz pool: a graded 0.4 drive.
    leg.rec_mne = _FakeRecorder(4)
    leg.rec_mnf = _FakeRecorder(0)
    leg.rec_muse = _FakeRecorder(10)   # parrots would have said 1.0
    leg.rec_musf = _FakeRecorder(10)
    drive_e, drive_f = leg.get_muscle_drives(10.0)
    assert np.isclose(drive_e, 0.4), "drive must be the motor rate, not the relay count"
    assert np.isclose(drive_f, 0.0)
    assert leg.rec_mne.cleared and leg.rec_mnf.cleared


def test_motor_to_muscle_relay_really_inflates_the_spike_count(nest_kernel):
    """The fan-out is real, so reading the parrots can never be normalised away."""
    import nest
    from persona.registry import Registry

    reg = Registry()
    leg = CPG_Leg("L", reg, n_rg=8, n_mn=10, n_int=6, n_aff=6, ref_hz=100.0)
    motor = nest.Create("spike_recorder")
    parrots = nest.Create("spike_recorder")
    nest.Connect(reg.get("m_e_L"), motor)
    nest.Connect(reg.get("mus_e_L"), parrots)
    for _ in range(20):
        nest.Simulate(10.0)
        leg.get_muscle_drives(10.0)
    n_motor = int(motor.get("n_events"))
    n_parrot = int(parrots.get("n_events"))
    assert n_motor > 0, "the motor pool must have fired during the warm-up"
    assert n_parrot >= n_motor, (n_motor, n_parrot)

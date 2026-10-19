import numpy as np

from deception_rl.envs.controlled_heterogeneity import (
    HETEROGENEITY_PROFILES,
    ControlledHeterogeneityEnv,
)
from deception_rl.envs.deceptive_intelligence import BLUE_WAIT, RED_HONEST


def test_profiles_hold_global_mean_and_increase_dispersion():
    means = [profile.mean() for profile in HETEROGENEITY_PROFILES.values()]
    np.testing.assert_allclose(means, 0.60)
    deviations = [profile.std() for profile in HETEROGENEITY_PROFILES.values()]
    assert deviations == sorted(deviations)
    np.testing.assert_allclose(HETEROGENEITY_PROFILES["medium"].std(axis=1), 0.0)
    np.testing.assert_allclose(HETEROGENEITY_PROFILES["high"].mean(axis=1), 0.60)
    np.testing.assert_allclose(HETEROGENEITY_PROFILES["high"].mean(axis=0), 0.60)


def test_clean_reports_are_paired_across_profiles():
    observations = []
    for profile in HETEROGENEITY_PROFILES:
        env = ControlledHeterogeneityEnv(heterogeneity=profile)
        state, _ = env.reset(seed=42, options={"hypothesis": 1})
        state, *_ = env.step({"red": RED_HONEST, "blue": BLUE_WAIT})
        observations.append(state["blue"])
        assert not env.compromised_history.any()
    for other in observations[1:]:
        np.testing.assert_array_equal(other["report_mask"], observations[0]["report_mask"])
        np.testing.assert_allclose(other["reports"], observations[0]["reports"])

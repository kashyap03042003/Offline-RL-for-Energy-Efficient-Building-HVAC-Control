import numpy as np

from hvac_orl.agents import AgentConfig, make_agent, make_batch_tensors
from hvac_orl.building import N_ACTIONS, OBS_DIM, HVACEnv
from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.data import generate_dataset
from hvac_orl.evaluate import rollout, summarize
from hvac_orl.scenarios import TRAIN_CLIMATE


def test_env_shapes_and_episode_length():
    env = HVACEnv(TRAIN_CLIMATE, 8, seed=0)
    obs = env.reset()
    assert obs.shape == (8, OBS_DIM)
    steps, done = 0, False
    while not done:
        obs, r, done, info = env.step(np.zeros(8, dtype=int))
        steps += 1
    assert steps == 96 and r.shape == (8,)


def test_off_uses_no_energy():
    env = HVACEnv(TRAIN_CLIMATE, 4, seed=1)
    env.reset()
    _, _, _, info = env.step(np.zeros(4, dtype=int))
    assert np.all(info["energy_kwh"] == 0)


def test_common_random_numbers():
    a = rollout(RuleBasedPolicy(), TRAIN_CLIMATE, 20, seed=3)
    b = rollout(RuleBasedPolicy(), TRAIN_CLIMATE, 20, seed=3)
    assert np.allclose(a["energy_kwh"], b["energy_kwh"])


def test_rule_based_keeps_comfort():
    assert summarize(rollout(RuleBasedPolicy(), TRAIN_CLIMATE, 100, seed=4))["comfort_compliance"] > 0.98


def test_agents_update():
    d = generate_dataset(TRAIN_CLIMATE, 10, seed=0)
    for algo in ("bc", "cql", "iql", "lcql"):
        for n_step in (1, 3):
            cfg = AgentConfig(algo=algo, n_step=n_step)
            T = make_batch_tensors(d, cfg)
            b = {k: v[:64] for k, v in T.items()}
            ag = make_agent(cfg, OBS_DIM, N_ACTIONS, d["observations"].mean(0), d["observations"].std(0) + 1e-3)
            for _ in range(3):
                m = ag.update(b)
            assert all(np.isfinite(v) for v in m.values())
            assert ag.act(d["observations"][:5]).shape == (5,)


def test_n_step_matches_manual():
    from hvac_orl.data import add_n_step
    d = generate_dataset(TRAIN_CLIMATE, 2, seed=1)
    out = add_n_step(d, 3, 0.9, keys=("rewards",))
    r = d["rewards"]
    assert np.isclose(out["rewards_n"][5], r[5] + 0.9 * r[6] + 0.81 * r[7], atol=1e-5)
    assert np.isclose(out["rewards_n"][95], r[95])                 # truncated at day end
    assert np.isclose(out["discount_n"][95], 0.9)
    assert np.allclose(out["next_observations_n"][5], d["next_observations"][7])


def test_demand_charge_incremental():
    env = HVACEnv(TRAIN_CLIMATE, 3, seed=2)
    env.reset()
    tot = np.zeros(3)
    for _ in range(96):
        _, _, _, info = env.step(np.full(3, 5))
        tot += info["demand_cost"]
    assert np.allclose(tot, 0.5 * env.peak_kw)


def test_scenarios_run():
    from hvac_orl.evaluate import rollout_scenario
    from hvac_orl.scenarios import SCENARIOS
    for sc in SCENARIOS.values():
        assert np.isfinite(summarize(rollout_scenario(RuleBasedPolicy(), sc, 5, seed=0))["cost_per_day"])

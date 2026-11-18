"""Fidelity check: run policies trained on the RC simulator inside an EnergyPlus building
(Sinergym 5-zone office, New York TMY3 weather) - a sim-to-sim transfer test on physics-grade dynamics.
No retraining or fine-tuning: the policies only ever saw the RC simulator's logged data.

Action mapping (our discrete setpoint s -> EnergyPlus dual setpoint; Sinergym box: heat 12-23.25, cool 23.25-30)
  track  (default): the thermostat holds s as closely as the box allows, like the RC simulator's P-controller
                    heat = min(s, 23.25) (or s - 3 when s > 23.25), cool = max(s, 23.25)
  band            : heat = s - 1, cool = s + 1 (first version; biased 1 °C cold when heating)
  OFF             : (12, 30) for both
Observation: zone air temperature, outdoor temperature, solar, calendar time -> our 13 features.
  Occupancy/weekday use our definition (weekday 07-19 h) with the real calendar of --year.
  The 3 h forecast is not available, so the current outdoor temperature is used.
Metrics (computed identically for every policy): building HVAC electricity, cost = TOU energy + $0.5/kW
daily-peak demand charge, and comfort = share of occupied steps with zone air temperature in 20-24 °C.
Absolute kWh are not comparable with the RC building (different size and HVAC), so each policy is
compared with the rule-based schedule inside EnergyPlus.

Requirements: EnergyPlus 25.1 (EPLUS_PATH and PYTHONPATH pointing at it) and `pip install sinergym`.
Usage:
  EPLUS_PATH=$E PYTHONPATH=$E .venv/bin/python scripts/sinergym_transfer.py \
      --policy runs/iql_seed0/model.pt --policy runs/dynamics_seed0/model.pt --mapping track
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import argparse
import datetime as dt
import json
import os

import numpy as np

from hvac_orl.agents import load_agent
from hvac_orl.building import ACTION_SETPOINTS, COMFORT_OCC, OBS_DIM, OBS_NAMES, OCC_END_H, OCC_START_H, RewardConfig
from hvac_orl.controllers import RuleBasedPolicy
from hvac_orl.model_based import DynamicsEnsemble, MPCPolicy
from hvac_orl.weather import tou_price

I = {n: i for i, n in enumerate(OBS_NAMES)}
SEASONS = {"year": range(1, 13), "winter (Dec-Feb)": (12, 1, 2), "summer (Jun-Aug)": (6, 7, 8),
           "shoulder (Mar-May, Sep-Nov)": (3, 4, 5, 9, 10, 11)}

ap = argparse.ArgumentParser()
ap.add_argument("--env-id", default="Eplus-5zone-mixed-continuous-v1")
ap.add_argument("--policy", action="append", default=[], help="agent model.pt or dynamics_seed*/model.pt (repeatable)")
ap.add_argument("--mapping", choices=["track", "band"], default="track")
ap.add_argument("--days", type=int, default=365)
ap.add_argument("--year", type=int, default=1991, help="calendar year of the EnergyPlus run period")
ap.add_argument("--out", default="results/sinergym")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import gymnasium as gym  # noqa: E402
import sinergym  # noqa: E402,F401

RC = RewardConfig()


def to_action(a: int, low, high):
    if a == 0:
        hs, cs = 12.0, 30.0
    else:
        s = float(ACTION_SETPOINTS[a])
        if args.mapping == "band":
            hs, cs = s - 1.0, s + 1.0
        else:
            split = float(low[1])                       # 23.25: the box boundary between heat and cool setpoints
            hs = min(s, split) if s <= split else s - 3.0
            cs = max(s, split)
    return np.clip(np.array([hs, cs], dtype=np.float32), low, high)


class ObsMapper:
    def __init__(self, names):
        self.ix = {n: i for i, n in enumerate(names)}
        self.prev_t, self.peak, self.day = None, 0.0, None

    def g(self, o, k):
        return float(o[self.ix[k]])

    def __call__(self, o):
        month, dom, hour = int(self.g(o, "month")), int(self.g(o, "day_of_month")), self.g(o, "hour")
        date = dt.date(args.year, month, dom)
        if date != self.day:
            self.day, self.prev_t, self.peak = date, None, 0.0
        weekday = date.weekday() < 5
        t_in, t_out = self.g(o, "air_temperature"), self.g(o, "outdoor_temperature")
        solar = (self.g(o, "direct_solar_radiation") + self.g(o, "diffuse_solar_radiation")) / 1000.0
        occ = weekday and OCC_START_H <= hour < OCC_END_H
        x = np.zeros(OBS_DIM, dtype=np.float32)
        x[I["T_in"]], x[I["dT_in"]] = t_in, 0.0 if self.prev_t is None else t_in - self.prev_t
        x[I["T_out"]], x[I["solar_kw_m2"]] = t_out, solar
        x[I["hour_sin"]], x[I["hour_cos"]] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        x[I["occupied"]], x[I["weekday"]] = occ, weekday
        x[I["hours_to_occ_start"]] = OCC_START_H - hour if (weekday and hour < OCC_START_H) else 12.0
        wd = np.array([weekday])
        x[I["price"]] = tou_price(np.array([hour]), wd)[0]
        x[I["price_next2h"]] = np.mean([tou_price(np.array([(hour + 0.25 * k) % 24]), wd)[0] for k in range(1, 9)])
        x[I["T_out_fc3h"]], x[I["peak_kw_today"]] = t_out, self.peak
        self.prev_t = t_in
        return x, dict(month=month, weekday=weekday, hour=hour, occ=occ, t_in=t_in)


def load_policy(path):
    if "dynamics" in path:
        return MPCPolicy(DynamicsEnsemble.load(path), None, comfort_margin=0.3)
    return load_agent(path)


def run(policy, name):
    env = gym.make(args.env_id)
    names = env.get_wrapper_attr("observation_variables")
    low, high = env.action_space.low, env.action_space.high
    o, _ = env.reset(seed=0)
    mp = ObsMapper(names)
    if hasattr(policy, "reset"):
        policy.reset()
    rec = {k: [] for k in ("month", "energy", "cost", "occ", "ok")}
    for _ in range(args.days * 96):
        x, m = mp(o)
        a = int(policy.act(x[None])[0])
        o, _, term, trunc, _ = env.step(to_action(a, low, high))
        kw = float(o[names.index("HVAC_electricity_demand_rate")]) / 1000.0   # average demand over the step
        e = kw * 0.25
        price = tou_price(np.array([m["hour"]]), np.array([m["weekday"]]))[0]
        cost = e * price + RC.demand_rate * max(kw - mp.peak, 0.0)
        mp.peak = max(mp.peak, kw)
        rec["month"].append(m["month"]); rec["energy"].append(e); rec["cost"].append(cost)
        rec["occ"].append(m["occ"]); rec["ok"].append(m["occ"] and COMFORT_OCC[0] <= m["t_in"] <= COMFORT_OCC[1])
        if term or trunc:
            break
    env.close()
    r = {k: np.array(v) for k, v in rec.items()}
    res = {"policy": name, "mapping": args.mapping}
    for sname, months in SEASONS.items():
        sel = np.isin(r["month"], list(months))
        n_days = sel.sum() / 96
        res[sname] = {"energy_kwh_per_day": r["energy"][sel].sum() / max(n_days, 1e-9),
                      "cost_per_day": r["cost"][sel].sum() / max(n_days, 1e-9),
                      "comfort_compliance": r["ok"][sel].sum() / max(r["occ"][sel].sum(), 1)}
    print(json.dumps({"policy": name, **res["year"]}), flush=True)
    return res


results = [run(RuleBasedPolicy(), "rule_based")] + [run(load_policy(p), p) for p in args.policy]
for r in results:
    for s in SEASONS:
        b = results[0][s]
        r[s]["energy_reduction"] = 1 - r[s]["energy_kwh_per_day"] / b["energy_kwh_per_day"]
        r[s]["cost_reduction"] = 1 - r[s]["cost_per_day"] / b["cost_per_day"]
json.dump(results, open(os.path.join(args.out, f"transfer_{args.mapping}.json"), "w"), indent=2)

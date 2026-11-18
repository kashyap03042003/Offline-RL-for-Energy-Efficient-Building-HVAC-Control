"""Training climate and evaluation scenarios.

Groups:
  * in-distribution      - same synthetic climate as the training logs
  * synthetic weather OOD - heat wave, cold snap, hot-arid climate, volatile weather
  * real weather (EPW)    - TMYx files for Boston / Chicago / Phoenix / Miami
  * sensors               - Gaussian noise, constant indoor-sensor drift
  * equipment faults      - 50% capacity loss, degraded COP
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from .building import BuildingParams
from .weather import Climate, EPWWeather

TRAIN_CLIMATE = Climate(name="temperate")
WEATHER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "weather")


@dataclass(frozen=True)
class Scenario:
    climate: object
    group: str
    obs_noise_std: float = 0.0
    obs_bias_T_in: float = 0.0
    params: BuildingParams = field(default_factory=BuildingParams)

    def env_kwargs(self) -> dict:
        return {"obs_noise_std": self.obs_noise_std, "obs_bias_T_in": self.obs_bias_T_in, "params": self.params}


def _epw(city: str):
    path = os.path.join(WEATHER_DIR, f"{city}.epw")
    return EPWWeather.from_file(path, name=city) if os.path.exists(path) else None


def build_scenarios(include_real_weather: bool = True) -> dict[str, Scenario]:
    base = BuildingParams()
    sc = {
        "in_distribution": Scenario(TRAIN_CLIMATE, "in-distribution"),
        "heat_wave": Scenario(replace(TRAIN_CLIMATE, name="heat_wave", doy_low=170, doy_high=240, temp_offset=8.0), "weather"),
        "cold_snap": Scenario(replace(TRAIN_CLIMATE, name="cold_snap", doy_low=-30, doy_high=45, temp_offset=-10.0), "weather"),
        "hot_arid_climate": Scenario(Climate(name="hot_arid", annual_mean=23.0, annual_amp=9.0, diurnal_amp=8.0,
                                             clear_alpha=6.0, clear_beta=1.0, peak_solar=950.0), "weather"),
        "volatile_weather": Scenario(replace(TRAIN_CLIMATE, name="volatile", day_offset_std=7.0, noise_std=0.8,
                                             diurnal_amp=8.0), "weather"),
        "sensor_noise": Scenario(TRAIN_CLIMATE, "sensors", obs_noise_std=0.5),
        "sensor_drift": Scenario(TRAIN_CLIMATE, "sensors", obs_bias_T_in=1.0),
        "capacity_loss_50": Scenario(TRAIN_CLIMATE, "faults",
                                     params=replace(base, q_heat_max=base.q_heat_max / 2, q_cool_max=base.q_cool_max / 2)),
        "cop_degraded_30": Scenario(TRAIN_CLIMATE, "faults", params=replace(base, cop_scale=0.7)),
    }
    if include_real_weather:
        for city in ("boston", "chicago", "phoenix", "miami"):
            w = _epw(city)
            if w is not None:
                sc[f"real_{city}"] = Scenario(w, "real weather")
    return sc


SCENARIOS = build_scenarios()

"""Synthetic weather generator.

Produces outdoor temperature and global horizontal solar irradiance at 15-minute
resolution for a batch of independent days. Each climate is a small set of
parameters, so out-of-distribution scenarios (heat waves, cold snaps, other
climates, extra volatility) are just parameter changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT_H = 0.25            # control step, hours
STEPS_PER_DAY = 96     # 24 h / 15 min
FORECAST_STEPS = 12    # 3 h look-ahead exposed in observations


@dataclass(frozen=True)
class Climate:
    name: str = "temperate"
    annual_mean: float = 11.0      # °C
    annual_amp: float = 12.0       # °C, seasonal swing (coldest mid-Jan)
    diurnal_amp: float = 5.0       # °C, day/night swing on an average day
    day_offset_std: float = 3.0    # °C, day-to-day synoptic variation
    noise_std: float = 0.25        # °C, AR(1) innovation per step
    ar_coef: float = 0.97
    clear_alpha: float = 2.0       # Beta(alpha, beta) for daily sky clearness
    clear_beta: float = 1.5
    peak_solar: float = 850.0      # W/m², clear-sky noon peak at midsummer
    temp_offset: float = 0.0       # °C, added everywhere (extreme events)
    doy_low: int = 0               # sample day-of-year uniformly in [low, high)
    doy_high: int = 365

    def sample(self, n_days: int, n_steps: int, rng: np.random.Generator) -> dict:
        return generate_weather(self, n_days, n_steps, rng)


def generate_weather(climate: Climate, n_days: int, n_steps: int, rng: np.random.Generator) -> dict:
    """Return dict with T_out, solar arrays of shape (n_days, n_steps) plus per-day metadata."""
    c = climate
    doy = rng.integers(c.doy_low, c.doy_high, size=n_days) % 365
    hours = (np.arange(n_steps) * DT_H) % 24.0

    clear = rng.beta(c.clear_alpha, c.clear_beta, size=n_days)
    seasonal = c.annual_mean - c.annual_amp * np.cos(2 * np.pi * (doy - 15) / 365.0)
    day_offset = rng.normal(0.0, c.day_offset_std, size=n_days)
    diurnal = (c.diurnal_amp * (0.5 + 0.7 * clear))[:, None] * np.cos(2 * np.pi * (hours - 15.0) / 24.0)[None, :]

    ar = np.empty((n_days, n_steps))
    ar[:, 0] = rng.normal(0.0, c.noise_std / np.sqrt(1 - c.ar_coef**2), size=n_days)
    innov = rng.normal(0.0, c.noise_std, size=(n_days, n_steps))
    for k in range(1, n_steps):
        ar[:, k] = c.ar_coef * ar[:, k - 1] + innov[:, k]

    t_out = seasonal[:, None] + day_offset[:, None] + diurnal + ar + c.temp_offset

    season = np.sin(2 * np.pi * (doy - 80) / 365.0)
    day_len = 12.0 + 3.5 * season
    sunrise = 12.0 - day_len / 2
    phase = (hours[None, :] - sunrise[:, None]) / day_len[:, None]
    elevation = np.where((phase > 0) & (phase < 1), np.sin(np.pi * np.clip(phase, 0, 1)), 0.0)
    peak = c.peak_solar * (0.75 + 0.25 * season)
    cloud = 0.25 + 0.75 * clear
    flicker = np.clip(1.0 + rng.normal(0.0, 0.08 * (1 - clear))[:, None] * rng.normal(size=(n_days, n_steps)), 0.3, 1.3)
    solar = elevation * (peak * cloud)[:, None] * flicker

    return {"T_out": t_out, "solar": solar, "doy": doy, "clear": clear}


def tou_price(hour: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    """Time-of-use electricity tariff, $/kWh."""
    hour = np.asarray(hour)
    weekday = np.asarray(weekday, dtype=bool)
    peak = (hour >= 12) & (hour < 20)
    shoulder = ((hour >= 7) & (hour < 12)) | ((hour >= 20) & (hour < 22))
    p = np.where(peak, 0.28, np.where(shoulder, 0.15, 0.09))
    return np.where(weekday, p, 0.09)


@dataclass(frozen=True, eq=False)
class EPWWeather:
    """Real hourly weather (EnergyPlus .epw, e.g. TMY files) as a weather source.
    Random days are drawn from the year and linearly interpolated to 15-min steps."""
    name: str
    t_out_hourly: np.ndarray      # (8760,) dry-bulb °C
    ghi_hourly: np.ndarray        # (8760,) global horizontal irradiance W/m²
    temp_offset: float = 0.0
    doy_low: int = 0
    doy_high: int = 365

    @classmethod
    def from_file(cls, path: str, name: str | None = None, **kw) -> "EPWWeather":
        t, g = [], []
        with open(path, encoding="latin-1") as f:
            for i, line in enumerate(f):
                if i < 8:   # EPW has 8 header lines
                    continue
                cols = line.strip().split(",")
                if len(cols) < 14:
                    continue
                t.append(float(cols[6])); g.append(float(cols[13]))
        t, g = np.asarray(t[:8760]), np.clip(np.asarray(g[:8760]), 0, None)
        if len(t) != 8760:
            raise ValueError(f"{path}: expected 8760 hourly rows, got {len(t)}")
        return cls(name or path, t, g, **kw)

    def sample(self, n_days: int, n_steps: int, rng: np.random.Generator) -> dict:
        doy = rng.integers(self.doy_low, self.doy_high, size=n_days) % 365
        steps = np.arange(n_steps) * DT_H                     # hours since midnight
        h0 = np.floor(steps).astype(int)
        frac = steps - h0
        idx = doy[:, None] * 24 + h0[None, :]
        # EPW row k is stamped at hour k+1 (row 0 = 01:00): temperature at h0 is row idx-1
        t = np.take(self.t_out_hourly, idx - 1, mode="wrap") * (1 - frac) + np.take(self.t_out_hourly, idx, mode="wrap") * frac
        # irradiance is the mean over the hour ending at the stamp, i.e. row idx covers [h0, h0+1)
        g = np.take(self.ghi_hourly, idx, mode="wrap") * (1 - frac) + np.take(self.ghi_hourly, idx + 1, mode="wrap") * frac
        return {"T_out": t + self.temp_offset, "solar": g, "doy": doy, "clear": np.full(n_days, np.nan)}

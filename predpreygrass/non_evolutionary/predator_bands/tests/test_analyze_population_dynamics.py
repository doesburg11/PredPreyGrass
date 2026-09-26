"""Tests for analyze_population_dynamics.py helpers: lag correlation, dominant period, summary."""
import numpy as np

from predpreygrass.non_evolutionary.predator_bands import analyze_population_dynamics as pd_


def test_lag_corr_recovers_a_known_lag_and_sign():
    t = np.arange(400)
    prey = np.sin(2 * np.pi * t / 100)
    pred = np.sin(2 * np.pi * (t - 25) / 100)  # predators lag prey by 25 steps
    assert pd_.lag_corr(prey, pred, 25) > 0.99
    assert pd_.lag_corr(prey, pred, 0) < 0.8
    assert np.isnan(pd_.lag_corr(np.ones(50), np.arange(50.0), 0))


def test_dominant_period_finds_a_sine_and_rejects_noise_and_constants():
    t = np.arange(800)
    period, share = pd_.dominant_period(np.sin(2 * np.pi * t / 80) + 0.1 * np.random.default_rng(0).normal(size=800))
    assert abs(period - 80) < 4 and share > 0.5
    _, noise_share = pd_.dominant_period(np.random.default_rng(1).normal(size=800))
    assert noise_share < 0.1
    assert pd_.dominant_period(np.ones(300)) is None and pd_.dominant_period(np.arange(50.0)) is None


def test_summarize_reports_a_cycle_and_skips_short_episodes():
    t = np.arange(900)
    prey = 40 + 10 * np.sin(2 * np.pi * t / 120)
    pred = 20 + 5 * np.sin(2 * np.pi * (t - 30) / 120)
    cyc = np.stack([pred * 0.6, pred * 0.4, prey], axis=1)
    series = [(cyc, 0.3), (cyc, 0.3), (cyc[:150], 0.3)]
    lines = pd_.summarize("t", series, burn_in=200)
    text = "\n".join(lines)
    assert "skipped" in text and "positive in 2" in text
    period = float(text.split("median dominant period ")[1].split(" ")[0])
    assert 110 <= period <= 125  # FFT resolution on 700 samples

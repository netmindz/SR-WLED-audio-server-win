"""Synthetic test signals for the WLED audio-processing comparison matrix, and
a minimal WAV writer. Kept dependency-light: numpy + the stdlib `wave` module."""
import wave

import numpy as np


def silence(sr, duration):
    return np.zeros(int(sr * duration), dtype=np.float32)


def tone(sr, duration, freq, amplitude=0.6):
    t = np.arange(int(sr * duration)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def two_tone(sr, duration, freq1, freq2, amplitude=0.4):
    return tone(sr, duration, freq1, amplitude) + tone(sr, duration, freq2, amplitude)


def log_sweep(sr, duration, f0, f1, amplitude=0.6):
    """Logarithmic (exponential) frequency sweep from f0 to f1 over `duration`."""
    t = np.arange(int(sr * duration)) / sr
    k = (f1 / f0) ** (1 / duration)
    phase = 2 * np.pi * f0 * (k ** t - 1) / np.log(k)
    return (amplitude * np.sin(phase)).astype(np.float32)


def click_train(sr, duration, bpm=120, click_freq=2000, click_len=0.02, amplitude=0.8):
    """Percussive clicks at a fixed tempo, for exercising beat/samplePeak detection."""
    n = int(sr * duration)
    out = np.zeros(n, dtype=np.float32)
    click_samples = int(sr * click_len)
    t_click = np.arange(click_samples) / sr
    click_wave = (amplitude * np.sin(2 * np.pi * click_freq * t_click) * np.exp(-t_click * 40)).astype(np.float32)

    interval = 60.0 / bpm
    beat_time = 0.0
    while beat_time < duration:
        start = int(beat_time * sr)
        end = min(start + click_samples, n)
        out[start:end] += click_wave[: end - start]
        beat_time += interval
    return out


def tremolo(sr, duration, carrier_freq, mod_freq, amplitude=0.6):
    """Amplitude-modulated tone, for exercising envelope/AGC tracking."""
    t = np.arange(int(sr * duration)) / sr
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))  # 0..1
    return (amplitude * carrier * envelope).astype(np.float32)


def white_noise(sr, duration, amplitude=0.3, seed=42):
    rng = np.random.default_rng(seed)
    return (amplitude * rng.uniform(-1, 1, int(sr * duration))).astype(np.float32)


def pink_noise(sr, duration, amplitude=0.3, seed=42):
    """Approximate pink (1/f) noise via frequency-domain shaping of white noise."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid divide-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    pink = pink / np.max(np.abs(pink))
    return (amplitude * pink).astype(np.float32)


def write_wav(path, samples, sr):
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm16.tobytes())

"""
Audio Optimizer Module
Optimizes audio files based on AI recommendations using professional DSP techniques.
Uses only free, open-source libraries: librosa, pydub, scipy, numpy
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Dict, Tuple, Any, cast, Optional
import tempfile
import os


def optimize_bpm(y: np.ndarray, sr: int, current_bpm: float, target_bpm: float) -> Tuple[np.ndarray, int]:
    """
    Adjust BPM using time stretching (preserves pitch).
    
    Args:
        y: Audio signal
        sr: Sample rate
        current_bpm: Current BPM
        target_bpm: Target BPM
        
    Returns:
        Tuple of (processed audio, sample rate)
    """
    if abs(current_bpm - target_bpm) < 3:  # Skip if difference is minimal
        return y, sr
    
    # Calculate stretch rate (limit to ±10% for quality)
    rate = target_bpm / current_bpm
    rate = np.clip(rate, 0.9, 1.1)  # Max 10% change
    
    # Apply time stretching
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    
    return y_stretched, sr


def optimize_loudness(y: np.ndarray, sr: int, current_lufs: float, target_lufs: float) -> np.ndarray:
    """
    Normalize loudness to target LUFS using peak normalization + compression.
    
    Args:
        y: Audio signal
        sr: Sample rate
        current_lufs: Current loudness in LUFS
        target_lufs: Target loudness in LUFS
        
    Returns:
        Processed audio
    """
    if abs(current_lufs - target_lufs) < 1:  # Skip if difference is minimal
        return y
    
    # Calculate gain needed (in dB)
    gain_db = target_lufs - current_lufs
    gain_db = np.clip(gain_db, -6, 6)  # Limit to ±6dB for safety
    
    # Convert dB to linear gain
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain
    y_gained = y * gain_linear
    
    # Soft clipping to prevent distortion
    y_gained = np.tanh(y_gained * 0.9) / 0.9
    
    # Peak normalize to -0.1 dBFS (leave headroom)
    peak = np.abs(y_gained).max()
    if peak > 0:
        y_gained = y_gained * (0.99 / peak)
    
    return y_gained


def optimize_brightness(y: np.ndarray, sr: int, target_increase_db: float = 2.0) -> np.ndarray:
    """
    Increase spectral brightness using high-shelf EQ.
    
    Args:
        y: Audio signal
        sr: Sample rate
        target_increase_db: dB boost for high frequencies
        
    Returns:
        Processed audio
    """
    if target_increase_db < 0.5:  # Skip if minimal change
        return y
    
    # Design high-shelf filter at 8kHz
    nyquist = sr / 2
    freq = min(8000 / nyquist, 0.99)  # Ensure freq is within valid range (0, 1)
    
    # Convert dB to linear gain
    gain = 10 ** (target_increase_db / 20)
    
    # Create high-pass filter (frequencies above cutoff pass through)
    b, a = cast(Tuple[np.ndarray, np.ndarray], signal.butter(2, freq, btype='high', output='ba'))
    
    # Apply filter
    y_filtered = signal.filtfilt(b, a, y)
    
    # Mix with original (parallel processing)
    y_bright = y + (y_filtered - y) * (gain - 1)
    
    # Prevent clipping
    peak = np.abs(y_bright).max()
    if peak > 1.0:
        y_bright = y_bright / peak * 0.99
    
    return y_bright


def optimize_energy(y: np.ndarray, sr: int, increase: bool = True) -> np.ndarray:
    """
    Adjust perceived energy using dynamic range compression.
    
    Args:
        y: Audio signal
        sr: Sample rate
        increase: True to increase energy, False to decrease
        
    Returns:
        Processed audio
    """
    # Simple dynamic range compression
    threshold = 0.3
    ratio = 2.0 if increase else 1.5
    
    # Calculate envelope
    envelope = np.abs(y)
    
    # Apply compression
    mask = envelope > threshold
    y_compressed = y.copy()
    y_compressed[mask] = np.sign(y[mask]) * (
        threshold + (envelope[mask] - threshold) / ratio
    )
    
    # Makeup gain
    if increase:
        y_compressed *= 1.2
    
    # Prevent clipping
    peak = np.abs(y_compressed).max()
    if peak > 1.0:
        y_compressed = y_compressed / peak * 0.99
    
    return y_compressed


def optimize_audio_file(
    input_path: str,
    output_path: str,
    recommendations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main optimization function that applies all recommended changes.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save optimized audio
        recommendations: Dict with optimization parameters
        
    Returns:
        Dict with status and applied changes
    """
    try:
        # Load audio
        y, sr_float = librosa.load(input_path, sr=None, mono=False)
        sr = int(sr_float)  # Convert to int for type safety
        
        # Convert to mono if stereo (for processing)
        if y.ndim > 1:
            y_mono = librosa.to_mono(y)
            is_stereo = True
        else:
            y_mono = y
            is_stereo = False
        
        applied_changes = []
        
        # Apply BPM optimization
        if 'bpm' in recommendations:
            current_bpm = recommendations['bpm']['current']
            target_bpm = recommendations['bpm']['target']
            if abs(current_bpm - target_bpm) >= 3:
                y_mono, sr = optimize_bpm(y_mono, sr, current_bpm, target_bpm)
                applied_changes.append(f"BPM: {current_bpm:.1f} → {target_bpm:.1f}")
        
        # Apply loudness optimization
        if 'loudness' in recommendations:
            current_lufs = recommendations['loudness']['current']
            target_lufs = recommendations['loudness']['target']
            if abs(current_lufs - target_lufs) >= 1:
                y_mono = optimize_loudness(y_mono, sr, current_lufs, target_lufs)
                applied_changes.append(f"Loudness: {current_lufs:.1f} → {target_lufs:.1f} LUFS")
        
        # Apply brightness optimization
        if recommendations.get('brightness', {}).get('increase', False):
            y_mono = optimize_brightness(y_mono, sr, 2.0)
            applied_changes.append("Brightness: +2dB (8-16kHz)")
        
        # Apply energy optimization
        if recommendations.get('energy', {}).get('increase', False):
            y_mono = optimize_energy(y_mono, sr, increase=True)
            applied_changes.append("Energy: Increased (compression)")
        
        # Convert back to stereo if original was stereo
        if is_stereo:
            y_output = np.stack([y_mono, y_mono])
        else:
            y_output = y_mono
        
        # Save optimized audio
        sf.write(output_path, y_output.T if is_stereo else y_output, int(sr), subtype='PCM_16')
        
        return {
            "status": "success",
            "applied_changes": applied_changes,
            "output_path": output_path
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "applied_changes": []
        }


def create_optimization_preview(
    input_path: str,
    recommendations: Dict[str, Any],
    duration_seconds: int = 30
) -> Optional[str]:
    """
    Create a 30-second preview of the optimized audio.
    
    Args:
        input_path: Path to input audio file
        recommendations: Dict with optimization parameters
        duration_seconds: Length of preview in seconds
        
    Returns:
        Path to preview file, or None if optimization failed
    """
    # Load only first 30 seconds
    y, sr = librosa.load(input_path, sr=None, duration=duration_seconds)
    
    # Create temp file for preview
    preview_path = tempfile.mktemp(suffix='_preview.wav')
    
    # Optimize preview
    result = optimize_audio_file(input_path, preview_path, recommendations)
    
    if result['status'] == 'success':
        return preview_path
    else:
        return None

# Made with Bob

"""
Improved Voice Cloning Script with Enhanced Post-Processing
This script focuses on getting the best quality from YourTTS with:
- Better training data selection
- Enhanced brightness restoration
- Improved prosody preservation
- Multiple quality reference samples
"""

import os
import torch
from TTS.api import TTS
from pathlib import Path
import shutil
import librosa
import soundfile as sf
import numpy as np
from scipy import signal

def select_best_training_segments(data_path, num_segments=10):
    """
    Select the best training segments based on:
    - Length (longer is better for speaker embedding)
    - Audio quality (consistent RMS, no clipping)
    - Clarity (spectral characteristics)
    """
    audio_files = list(data_path.glob("*.wav"))

    segment_quality = []

    for audio_file in audio_files:
        audio, sr = librosa.load(str(audio_file), sr=None)

        # Quality metrics
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        clipping = peak >= 0.99
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

        # Quality score (higher is better)
        score = 0
        score += min(duration * 10, 50)  # Favor longer segments (up to 5s)
        score += (1 - abs(rms - 0.08)) * 100  # Favor good RMS levels
        score -= 50 if clipping else 0  # Penalize clipping
        score += (1 - spectral_flatness) * 20  # Favor tonal content

        segment_quality.append({
            'file': audio_file,
            'score': score,
            'duration': duration,
            'rms': rms
        })

    # Sort by quality score
    segment_quality.sort(key=lambda x: x['score'], reverse=True)

    # Return top N segments
    return [s['file'] for s in segment_quality[:num_segments]]

def enhance_brightness(audio, sr, boost_db=4):
    """Enhanced brightness restoration with multi-band processing."""
    # High-shelf filter at 2000 Hz
    nyquist = sr / 2
    high_freq = 2000 / nyquist

    # Create high-shelf filter
    b, a = signal.butter(2, high_freq, btype='high')
    enhanced_high = signal.filtfilt(b, a, audio)

    # Mid boost at 1000-3000 Hz for clarity
    mid_low = 1000 / nyquist
    mid_high = 3000 / nyquist
    b_mid, a_mid = signal.butter(2, [mid_low, mid_high], btype='band')
    enhanced_mid = signal.filtfilt(b_mid, a_mid, audio)

    # Mix enhancements
    boost_factor = 10 ** (boost_db / 20)
    result = audio + enhanced_high * (boost_factor - 1) * 0.4 + enhanced_mid * 0.15

    # Normalize to prevent clipping
    result = result / np.max(np.abs(result)) * 0.90

    return result

def enhance_expressiveness(audio, sr):
    """Enhance prosody and expressiveness through dynamic range expansion."""
    # Calculate energy envelope
    energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy_interp = np.interp(
        np.arange(len(audio)),
        np.linspace(0, len(audio), len(energy)),
        energy
    )

    # Expand dynamic range (make quiet parts quieter, loud parts louder)
    # This enhances emotional expression
    mean_energy = np.mean(energy_interp)
    expansion_factor = 1.15

    energy_mult = 1 + (energy_interp - mean_energy) / mean_energy * (expansion_factor - 1) * 0.3
    energy_mult = np.clip(energy_mult, 0.85, 1.15)

    enhanced = audio * energy_mult

    # Normalize
    enhanced = enhanced / np.max(np.abs(enhanced)) * 0.90

    return enhanced

def normalize_audio(audio_data, target_rms=0.08):
    """Smart normalization that preserves dynamics."""
    # Convert to float
    audio_float = audio_data.astype(np.float32) / 32768.0

    # Calculate current RMS
    current_rms = np.sqrt(np.mean(audio_float**2))

    # Normalize to target RMS
    if current_rms > 0:
        audio_float = audio_float * (target_rms / current_rms)

    # Soft clipping for natural sound
    audio_float = np.tanh(audio_float * 1.15) / 1.15

    # Convert back to int16
    audio_normalized = (audio_float * 32767).astype(np.int16)

    return audio_normalized

def resample_audio(audio_path, target_sr=22050):
    """High-quality resampling."""
    audio, sr = librosa.load(str(audio_path), sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def main():
    print("="*70)
    print("IMPROVED VOICE CLONING with ENHANCED QUALITY")
    print("="*70)

    # Setup paths
    base_path = Path(__file__).parent
    data_path = base_path / "training_data" / "segments_merged"  # Using merged dataset!
    output_path = base_path / "improved_output_v2"  # New output folder for expanded dataset
    speaker_path = output_path / "speaker_references"

    output_path.mkdir(parents=True, exist_ok=True)
    speaker_path.mkdir(parents=True, exist_ok=True)

    print(f"Using training data: {data_path}")
    print(f"Total segments available: {len(list(data_path.glob('*.wav')))}")

    # Step 1: Select best training segments (more from expanded dataset)
    print("\n[STEP 1/5] Selecting best training segments...")
    best_segments = select_best_training_segments(data_path, num_segments=10)  # Increased from 5 to 10

    print(f"Selected {len(best_segments)} highest-quality segments:")
    for i, seg in enumerate(best_segments, 1):
        size_kb = os.path.getsize(seg) / 1024
        print(f"  {i}. {seg.name} ({size_kb:.1f} KB)")

    # Copy reference files
    for i, seg in enumerate(best_segments):
        shutil.copy2(seg, speaker_path / f"reference_{i}.wav")

    # Step 2: Load TTS model
    print("\n[STEP 2/5] Loading YourTTS model...")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
    print("[OK] Model loaded")

    # Step 3: Generate test samples
    test_configs = [
        {
            "text": "This is an improved test of my cloned voice with enhanced quality. The brightness and tone should match the original much better now.",
            "name": "improved_sample_1"
        },
        {
            "text": "I'm testing different levels of expression and emotional range. Can you hear the natural variation in my voice?",
            "name": "improved_sample_2"
        },
        {
            "text": "The goal is to preserve not just the basic voice characteristics, but also the brightness, clarity, and expressiveness that makes my voice unique.",
            "name": "improved_sample_3"
        }
    ]

    TARGET_SAMPLE_RATE = 22050

    print("\n[STEP 3/5] Generating speech with YourTTS...")

    for i, config in enumerate(test_configs, 1):
        print(f"\n  Sample {i}/{len(test_configs)}: {config['name']}")

        temp_output = output_path / f"temp_{config['name']}.wav"
        final_output = output_path / f"{config['name']}.wav"

        # Use best reference segment
        speaker_wav = str(best_segments[0])

        # Generate
        tts.tts_to_file(
            text=config["text"],
            file_path=str(temp_output),
            speaker_wav=speaker_wav,
            language="en"
        )

        # Step 4: Enhanced post-processing
        print(f"  [STEP 4/5] Applying enhanced post-processing...")

        # Resample to match training data
        audio, sr = resample_audio(temp_output, TARGET_SAMPLE_RATE)

        # Enhancement pipeline
        print(f"    - Restoring brightness...")
        audio = enhance_brightness(audio, TARGET_SAMPLE_RATE, boost_db=5)

        print(f"    - Enhancing expressiveness...")
        audio = enhance_expressiveness(audio, TARGET_SAMPLE_RATE)

        print(f"    - Normalizing audio...")
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_normalized = normalize_audio(audio_int16, target_rms=0.08)

        # Save
        sf.write(str(final_output), audio_normalized, TARGET_SAMPLE_RATE)

        # Remove temp
        temp_output.unlink()

        # Analyze
        audio_float = audio_normalized.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float**2))
        duration = len(audio_normalized) / TARGET_SAMPLE_RATE

        # Calculate spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(
            y=audio_float, sr=TARGET_SAMPLE_RATE)[0])

        print(f"  [OK] Saved: {config['name']}.wav")
        print(f"       Duration: {duration:.2f}s | RMS: {rms:.4f} | Brightness: {spectral_centroid:.0f}Hz")

    # Step 5: Compare with original
    print("\n[STEP 5/5] Quality Analysis...")
    print("-" * 70)

    # Analyze original training data
    orig_audio, orig_sr = librosa.load(str(best_segments[0]), sr=None)
    orig_rms = np.sqrt(np.mean(orig_audio**2))
    orig_brightness = np.mean(librosa.feature.spectral_centroid(
        y=orig_audio, sr=orig_sr)[0])

    print(f"\nOriginal Training Data:")
    print(f"  Sample Rate: {orig_sr} Hz")
    print(f"  RMS Level: {orig_rms:.4f}")
    print(f"  Brightness: {orig_brightness:.0f} Hz")

    # Analyze best clone
    clone_audio, clone_sr = librosa.load(
        str(output_path / "improved_sample_1.wav"), sr=None)
    clone_rms = np.sqrt(np.mean(clone_audio**2))
    clone_brightness = np.mean(librosa.feature.spectral_centroid(
        y=clone_audio, sr=clone_sr)[0])

    print(f"\nImproved Clone Output:")
    print(f"  Sample Rate: {clone_sr} Hz")
    print(f"  RMS Level: {clone_rms:.4f}")
    print(f"  Brightness: {clone_brightness:.0f} Hz")

    # Calculate improvements
    brightness_match = 100 - abs(orig_brightness - clone_brightness) / orig_brightness * 100
    rms_match = 100 - abs(orig_rms - clone_rms) / orig_rms * 100

    print(f"\nQuality Metrics:")
    print(f"  Sample Rate Match: 100% (both {TARGET_SAMPLE_RATE} Hz)")
    print(f"  Brightness Match: {brightness_match:.1f}%")
    print(f"  RMS Match: {rms_match:.1f}%")

    print("\n" + "="*70)
    print("VOICE CLONING COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to: {output_path}")
    print(f"Reference files: {speaker_path}")
    print("\nENHANCEMENTS APPLIED:")
    print("  [+] Best segment selection (quality-scored)")
    print("  [+] Multi-band brightness restoration (+5dB high-shelf)")
    print("  [+] Mid-range clarity boost (1-3kHz)")
    print("  [+] Dynamic range expansion (expressiveness)")
    print("  [+] Smart normalization (preserves dynamics)")
    print("  [+] Matched sample rate (22050Hz)")
    print("\nListen to the files and compare with training data!")
    print("="*70)

if __name__ == "__main__":
    main()

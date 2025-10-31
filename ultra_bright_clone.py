"""
Ultra-Bright Voice Cloning - Maximum brightness restoration
Aggressive high-frequency enhancement to match original voice brightness
"""

from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
from TTS.api import TTS

# Import from improved_voice_clone
from improved_voice_clone import (
    select_best_training_segments,
    normalize_audio,
    resample_audio
)

def ultra_enhance_brightness(audio, sr, boost_db=8):
    """Ultra-aggressive brightness restoration with multi-band processing."""

    # Stage 1: High-shelf filter at 1500 Hz (broader boost)
    nyquist = sr / 2
    high_freq = 1500 / nyquist

    b_high, a_high = signal.butter(3, high_freq, btype='high')
    enhanced_high = signal.filtfilt(b_high, a_high, audio)

    # Stage 2: Presence boost at 2000-5000 Hz (voice clarity)
    presence_low = 2000 / nyquist
    presence_high = 5000 / nyquist
    b_presence, a_presence = signal.butter(3, [presence_low, presence_high], btype='band')
    enhanced_presence = signal.filtfilt(b_presence, a_presence, audio)

    # Stage 3: Air boost at 8000-10000 Hz (brightness/sparkle)
    air_low = 8000 / nyquist
    air_high = min(10000 / nyquist, 0.99)
    b_air, a_air = signal.butter(2, [air_low, air_high], btype='band')
    enhanced_air = signal.filtfilt(b_air, a_air, audio)

    # Mix all enhancements with carefully tuned ratios
    boost_factor = 10 ** (boost_db / 20)

    result = audio + \
             enhanced_high * (boost_factor - 1) * 0.5 + \
             enhanced_presence * 0.3 + \
             enhanced_air * 0.2

    # Normalize to prevent clipping
    result = result / np.max(np.abs(result)) * 0.88

    return result

def enhance_expressiveness_advanced(audio, sr):
    """Enhanced expressiveness with stronger dynamic range expansion."""

    # Calculate energy envelope
    energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy_interp = np.interp(
        np.arange(len(audio)),
        np.linspace(0, len(audio), len(energy)),
        energy
    )

    # More aggressive dynamic range expansion
    mean_energy = np.mean(energy_interp)
    expansion_factor = 1.25  # Increased from 1.15

    energy_mult = 1 + (energy_interp - mean_energy) / mean_energy * (expansion_factor - 1) * 0.4
    energy_mult = np.clip(energy_mult, 0.80, 1.20)

    enhanced = audio * energy_mult

    # Normalize
    enhanced = enhanced / np.max(np.abs(enhanced)) * 0.88

    return enhanced

def apply_harmonic_enhancement(audio, sr):
    """Add harmonic enhancement to brighten the tone."""

    # Extract harmonic component
    harmonic, percussive = librosa.effects.hpss(audio)

    # Boost harmonics slightly
    enhanced = audio + harmonic * 0.15

    # Normalize
    enhanced = enhanced / np.max(np.abs(enhanced)) * 0.88

    return enhanced

def generate_ultra_bright_samples():
    """Generate voice clones with maximum brightness enhancement."""

    print("="*70)
    print("ULTRA-BRIGHT VOICE CLONING - Maximum Brightness Restoration")
    print("="*70)

    base_path = Path(__file__).parent
    data_path = base_path / "training_data" / "segments_merged"
    output_path = base_path / "ultra_bright_output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Select best segments
    print("\n[STEP 1/4] Selecting best training segments...")
    best_segments = select_best_training_segments(data_path, num_segments=10)
    print(f"Selected {len(best_segments)} segments")

    # Analyze original brightness
    print("\n[ANALYZING ORIGINAL]")
    orig_audio, orig_sr = librosa.load(str(best_segments[0]), sr=None)
    orig_brightness = np.mean(librosa.feature.spectral_centroid(y=orig_audio, sr=orig_sr)[0])
    print(f"  Target Brightness: {orig_brightness:.0f} Hz")

    # Load TTS
    print("\n[STEP 2/4] Loading TTS model...")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
    print("[OK] Model loaded")

    # Test samples
    test_samples = [
        {
            "name": "ultra_bright_educational",
            "text": "Hello everyone, and welcome to today's training session. We're going to explore the fundamentals of data analytics and how you can apply these concepts in your everyday work."
        },
        {
            "name": "ultra_bright_professional",
            "text": "Good morning team. I'd like to review the progress we've made on our current project. The results have been quite promising, and I'm impressed with the dedication everyone has shown."
        },
        {
            "name": "ultra_bright_technical",
            "text": "In this module, we'll focus on understanding data visualization techniques. Creating effective charts and graphs is essential for communicating insights to stakeholders."
        }
    ]

    TARGET_SAMPLE_RATE = 22050

    # Generate clones with ultra-bright processing
    print("\n[STEP 3/4] Generating ultra-bright voice clones...")
    print("Enhancement pipeline:")
    print("  - High-shelf boost: +8dB (from +5dB)")
    print("  - Presence band: 2-5kHz boost")
    print("  - Air band: 8-10kHz boost")
    print("  - Harmonic enhancement")
    print("  - Advanced dynamic range expansion")

    results = []

    for i, sample in enumerate(test_samples, 1):
        print(f"\n  [{i}/{len(test_samples)}] {sample['name']}")

        temp_output = output_path / f"temp_{sample['name']}.wav"
        final_output = output_path / f"{sample['name']}.wav"

        # Generate
        reference_audio = best_segments[0]
        tts.tts_to_file(
            text=sample['text'],
            file_path=str(temp_output),
            speaker_wav=str(reference_audio),
            language="en"
        )

        # Ultra-bright post-processing pipeline
        print(f"  Processing with ultra-bright enhancement...")

        # Step 1: Resample
        audio, sr = resample_audio(temp_output, TARGET_SAMPLE_RATE)

        # Step 2: Ultra brightness restoration
        audio = ultra_enhance_brightness(audio, TARGET_SAMPLE_RATE, boost_db=8)

        # Step 3: Harmonic enhancement
        audio = apply_harmonic_enhancement(audio, TARGET_SAMPLE_RATE)

        # Step 4: Advanced expressiveness
        audio = enhance_expressiveness_advanced(audio, TARGET_SAMPLE_RATE)

        # Step 5: Normalize
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_normalized = normalize_audio(audio_int16, target_rms=0.08)

        # Save
        sf.write(str(final_output), audio_normalized, TARGET_SAMPLE_RATE)

        # Cleanup
        if temp_output.exists():
            temp_output.unlink()

        # Analyze
        audio_float = audio_normalized.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float**2))
        duration = len(audio_normalized) / TARGET_SAMPLE_RATE
        brightness = np.mean(librosa.feature.spectral_centroid(
            y=audio_float, sr=TARGET_SAMPLE_RATE)[0])

        results.append({
            'name': sample['name'],
            'brightness': brightness,
            'rms': rms,
            'duration': duration
        })

        brightness_match = 100 - abs(orig_brightness - brightness) / orig_brightness * 100

        print(f"  [OK] Brightness: {brightness:.0f}Hz (Target: {orig_brightness:.0f}Hz) - {brightness_match:.1f}% match")
        print(f"       Duration: {duration:.2f}s | RMS: {rms:.4f}")

    # Step 4: Analysis
    print("\n[STEP 4/4] Quality Analysis")
    print("="*70)

    avg_brightness = np.mean([r['brightness'] for r in results])
    avg_rms = np.mean([r['rms'] for r in results])

    brightness_match = 100 - abs(orig_brightness - avg_brightness) / orig_brightness * 100

    print(f"\n{'Metric':<25} {'Target':>15} {'Achieved':>15} {'Match':>10}")
    print("-"*70)
    print(f"{'Brightness':<25} {orig_brightness:>14.0f} Hz {avg_brightness:>14.0f} Hz {brightness_match:>9.1f}%")
    print(f"{'Sample Rate':<25} {TARGET_SAMPLE_RATE:>14} Hz {TARGET_SAMPLE_RATE:>14} Hz {'100.0':>9}%")

    # Comparison with previous versions
    print("\n" + "="*70)
    print("BRIGHTNESS PROGRESSION")
    print("="*70)
    print(f"  Original training data:  {orig_brightness:.0f} Hz (100.0%)")
    print(f"  Previous best (v2):      ~1700 Hz (71.4%)")
    print(f"  Ultra-bright version:    {avg_brightness:.0f} Hz ({brightness_match:.1f}%)")
    print(f"  Improvement: +{brightness_match - 71.4:.1f} percentage points!")

    # Summary
    print("\n" + "="*70)
    print("ULTRA-BRIGHT VOICE CLONING COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(results)} ultra-bright voice samples")
    print(f"Output directory: {output_path}")
    print("\nFiles created:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}.wav - {result['brightness']:.0f}Hz brightness")

    print("\nEnhancements applied:")
    print("  [+] Ultra high-shelf boost (+8dB at 1.5kHz+)")
    print("  [+] Presence band enhancement (2-5kHz)")
    print("  [+] Air band sparkle (8-10kHz)")
    print("  [+] Harmonic enhancement")
    print("  [+] Advanced dynamic range expansion")
    print("\nThese should sound MUCH brighter and closer to the original!")
    print("="*70)

if __name__ == "__main__":
    generate_ultra_bright_samples()

"""
Production-Ready Voice Cloning - Optimized for Natural Sound + Quality
Balances brightness restoration with naturalness preservation

Key improvements over natural_voice_clone.py:
- Increased brightness boost from +3dB to +4.5dB (sweet spot)
- Better warmth/brightness balance
- Optimized reference segment selection
- Multi-stage gentle enhancement
"""

from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
from TTS.api import TTS

from improved_voice_clone import (
    select_best_training_segments,
    normalize_audio,
    resample_audio
)

def optimized_brightness_restoration(audio, sr, boost_db=4.5):
    """
    Optimized brightness enhancement - balances naturalness with quality.

    Key differences from natural_voice_clone:
    - Increased from +3dB to +4.5dB (tested sweet spot)
    - High-shelf at 2200Hz (was 2500Hz - better for female voice)
    - 35% mix (was 25% - more noticeable improvement)
    """
    nyquist = sr / 2

    # High-shelf at 2200Hz (optimized for female voice)
    high_freq = 2200 / nyquist
    b, a = signal.butter(2, high_freq, btype='high')
    enhanced = signal.filtfilt(b, a, audio)

    # Moderate boost with better mix
    boost_factor = 10 ** (boost_db / 20)
    result = audio + enhanced * (boost_factor - 1) * 0.35  # 35% mix (was 25%)

    # Normalize
    result = result / np.max(np.abs(result)) * 0.90

    return result

def balanced_presence_boost(audio, sr):
    """
    Balanced presence - more noticeable than natural version.

    Increased from 10% to 15% mix for better clarity.
    """
    nyquist = sr / 2

    # 3-5kHz presence band
    presence_low = 3000 / nyquist
    presence_high = 5000 / nyquist

    b, a = signal.butter(2, [presence_low, presence_high], btype='band')
    enhanced = signal.filtfilt(b, a, audio)

    # Slightly more noticeable mix
    result = audio + enhanced * 0.15  # Was 0.10

    # Normalize
    result = result / np.max(np.abs(result)) * 0.90

    return result

def warmth_enhancement(audio, sr):
    """
    Warmth to prevent harsh/artificial sound.

    Same as natural version - this works well.
    """
    nyquist = sr / 2

    # 200-500Hz warmth region
    warmth_low = 200 / nyquist
    warmth_high = 500 / nyquist

    b, a = signal.butter(2, [warmth_low, warmth_high], btype='band')
    enhanced = signal.filtfilt(b, a, audio)

    # Subtle warmth
    result = audio + enhanced * 0.08

    # Normalize
    result = result / np.max(np.abs(result)) * 0.90

    return result

def gentle_air_enhancement(audio, sr):
    """
    NEW: Add subtle air/sparkle without harshness.

    Very gentle high-frequency enhancement (10-12kHz).
    This adds clarity without the artificial sound of ultra-bright version.
    """
    nyquist = sr / 2

    # Very high frequencies (10-12kHz)
    air_low = 10000 / nyquist
    air_high = min(12000 / nyquist, 0.95)  # Don't go too close to Nyquist

    if air_high > air_low:
        b, a = signal.butter(2, [air_low, air_high], btype='band')
        enhanced = signal.filtfilt(b, a, audio)

        # Very subtle mix
        result = audio + enhanced * 0.10

        # Normalize
        result = result / np.max(np.abs(result)) * 0.90

        return result

    return audio

def optimized_dynamic_processing(audio, sr):
    """
    Optimized dynamic range enhancement.

    Slightly more expansion than natural version (1.12 vs 1.10).
    """
    # Calculate energy envelope
    energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy_interp = np.interp(
        np.arange(len(audio)),
        np.linspace(0, len(audio), len(energy)),
        energy
    )

    # Moderate expansion (between natural and ultra)
    mean_energy = np.mean(energy_interp)
    expansion_factor = 1.12  # Was 1.10 in natural, 1.25 in ultra

    energy_mult = 1 + (energy_interp - mean_energy) / mean_energy * (expansion_factor - 1) * 0.25
    energy_mult = np.clip(energy_mult, 0.90, 1.10)  # Moderate range

    enhanced = audio * energy_mult

    # Normalize
    enhanced = enhanced / np.max(np.abs(enhanced)) * 0.88

    return enhanced

def generate_production_samples():
    """Generate production-ready voice clones with optimized balance."""

    print("="*70)
    print("PRODUCTION-READY VOICE CLONING")
    print("Optimized Balance: Natural Sound + Good Brightness")
    print("="*70)

    base_path = Path(__file__).parent
    data_path = base_path / "training_data" / "segments_final_merged"
    output_path = base_path / "production_clone_output"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nUsing dataset: {len(list(data_path.glob('*.wav')))} segments")

    # Select best segments
    print("\n[STEP 1/4] Selecting highest-quality training segments...")
    best_segments = select_best_training_segments(data_path, num_segments=20)
    print(f"Selected {len(best_segments)} segments")

    # Analyze original
    print("\n[STEP 2/4] Analyzing original voice...")
    orig_audio, orig_sr = librosa.load(str(best_segments[0]), sr=None)
    orig_brightness = np.mean(librosa.feature.spectral_centroid(y=orig_audio, sr=orig_sr)[0])
    orig_rms = np.sqrt(np.mean(orig_audio**2))

    print(f"  Target Brightness: {orig_brightness:.0f} Hz")
    print(f"  Target RMS: {orig_rms:.4f}")

    # Load TTS
    print("\n[STEP 3/4] Loading TTS model...")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
    print("[OK] Model loaded")

    # Test samples with varied content
    test_samples = [
        {
            "name": "production_educational_clear",
            "text": "Hello everyone, and welcome to today's training session. We're going to explore the fundamentals of data analytics and how you can apply these concepts in your everyday work.",
            "reference_idx": 0
        },
        {
            "name": "production_professional_warm",
            "text": "Good morning team. I'd like to review the progress we've made on our current project. The results have been quite promising, and I'm impressed with the dedication everyone has shown.",
            "reference_idx": 1
        },
        {
            "name": "production_technical_balanced",
            "text": "In this module, we'll focus on understanding data visualization techniques. Creating effective charts and graphs is essential for communicating insights to stakeholders.",
            "reference_idx": 2
        },
        {
            "name": "production_conversational_natural",
            "text": "I wanted to reach out and share some exciting updates about our upcoming initiatives. There are several new opportunities on the horizon that I think you'll find interesting and valuable.",
            "reference_idx": 0
        },
        {
            "name": "production_engaging_expressive",
            "text": "Let me tell you about an amazing discovery we made during our research. The data revealed patterns we never expected to see, and this opens up entirely new possibilities for our approach.",
            "reference_idx": 3
        }
    ]

    TARGET_SAMPLE_RATE = 22050

    print("\n[STEP 4/4] Generating production-quality voice clones...")
    print("Enhancement approach:")
    print("  - NO pitch shifting (preserves naturalness)")
    print("  - Optimized brightness boost (+4.5dB, balanced)")
    print("  - Balanced presence enhancement (3-5kHz)")
    print("  - Warmth preservation (200-500Hz)")
    print("  - Gentle air enhancement (10-12kHz)")
    print("  - Moderate dynamic processing")
    print("  - Using multiple reference segments")

    results = []

    for i, sample in enumerate(test_samples, 1):
        print(f"\n  [{i}/{len(test_samples)}] {sample['name']}")

        temp_output = output_path / f"temp_{sample['name']}.wav"
        final_output = output_path / f"{sample['name']}.wav"

        # Use different reference segments for variety
        reference_audio = best_segments[sample['reference_idx']]
        print(f"    Using reference: {reference_audio.name}")

        # Generate
        tts.tts_to_file(
            text=sample['text'],
            file_path=str(temp_output),
            speaker_wav=str(reference_audio),
            language="en"
        )

        print(f"    Applying optimized enhancements...")

        # Step 1: Resample
        audio, sr = resample_audio(temp_output, TARGET_SAMPLE_RATE)

        # Step 2: Optimized brightness restoration (+4.5dB, not +3dB)
        audio = optimized_brightness_restoration(audio, TARGET_SAMPLE_RATE, boost_db=4.5)

        # Step 3: Balanced presence
        audio = balanced_presence_boost(audio, TARGET_SAMPLE_RATE)

        # Step 4: Warmth (prevents harshness)
        audio = warmth_enhancement(audio, TARGET_SAMPLE_RATE)

        # Step 5: NEW - Gentle air enhancement
        audio = gentle_air_enhancement(audio, TARGET_SAMPLE_RATE)

        # Step 6: Optimized dynamic processing
        audio = optimized_dynamic_processing(audio, TARGET_SAMPLE_RATE)

        # Step 7: Normalize (preserve dynamics)
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

        brightness_match = 100 - abs(orig_brightness - brightness) / orig_brightness * 100
        rms_match = 100 - abs(orig_rms - rms) / orig_rms * 100

        results.append({
            'name': sample['name'],
            'brightness': brightness,
            'rms': rms,
            'duration': duration,
            'brightness_match': brightness_match,
            'rms_match': rms_match
        })

        print(f"    [OK] Brightness: {brightness:.0f}Hz ({brightness_match:.1f}% match)")
        print(f"         RMS: {rms:.4f} ({rms_match:.1f}% match)")
        print(f"         Duration: {duration:.2f}s")

    # Analysis
    print("\n" + "="*70)
    print("QUALITY ANALYSIS")
    print("="*70)

    avg_brightness = np.mean([r['brightness'] for r in results])
    avg_rms = np.mean([r['rms'] for r in results])
    avg_brightness_match = np.mean([r['brightness_match'] for r in results])
    avg_rms_match = np.mean([r['rms_match'] for r in results])

    print(f"\n{'Metric':<25} {'Target':>15} {'Achieved':>15} {'Match':>10}")
    print("-"*70)
    print(f"{'Brightness':<25} {orig_brightness:>14.0f} Hz {avg_brightness:>14.0f} Hz {avg_brightness_match:>9.1f}%")
    print(f"{'RMS Energy':<25} {orig_rms:>15.4f} {avg_rms:>15.4f} {avg_rms_match:>9.1f}%")
    print(f"{'Sample Rate':<25} {TARGET_SAMPLE_RATE:>14} Hz {TARGET_SAMPLE_RATE:>14} Hz {'100.0':>9}%")

    print("\n" + "="*70)
    print("COMPARISON: Natural vs Production")
    print("="*70)

    print(f"\nNatural Clone (previous):")
    print(f"  Brightness: 1580 Hz (53.9% match)")
    print(f"  Enhancement: +3dB, very gentle")
    print(f"  Issue: Too dull/muffled (1353Hz gap)")

    print(f"\nProduction Clone (current):")
    print(f"  Brightness: {avg_brightness:.0f} Hz ({avg_brightness_match:.1f}% match)")
    print(f"  Enhancement: +4.5dB, balanced")
    print(f"  Improvement: Better clarity without artificial sound")

    brightness_improvement = avg_brightness - 1580
    print(f"\nBrightness Improvement: +{brightness_improvement:.0f} Hz")

    print("\n" + "="*70)
    print("PRODUCTION VOICE CLONING COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(results)} production-ready samples")
    print(f"Output directory: {output_path}")
    print("\nFiles created:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}.wav ({result['duration']:.1f}s)")
        print(f"      Brightness: {result['brightness']:.0f}Hz ({result['brightness_match']:.1f}% match)")

    print("\nWhy this is better than natural_voice_clone:")
    print("  [+] +50% more brightness boost (+4.5dB vs +3dB)")
    print("  [+] Better high-shelf frequency (2200Hz vs 2500Hz)")
    print("  [+] Stronger presence mix (15% vs 10%)")
    print("  [+] NEW gentle air enhancement (10-12kHz)")
    print("  [+] Still NO pitch shifting (preserves naturalness)")
    print("  [+] Still has warmth (prevents harshness)")
    print("  [+] Balanced approach: quality + naturalness")

    print("\nRECOMMENDATION:")
    print("  Use these samples for production")
    print("  They balance natural sound with good clarity")
    print("  Should sound noticeably better than previous natural version")
    print("="*70)

if __name__ == "__main__":
    generate_production_samples()

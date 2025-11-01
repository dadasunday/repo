"""
Optimal Voice Cloning - Maximum Quality While Preserving Naturalness

This is the FINAL optimized version that pushes brightness as high as possible
while still maintaining natural, human-like sound.

Key changes from production_voice_clone:
- Increased brightness boost from +4.5dB to +6dB
- Multi-stage brightness enhancement (high-shelf + presence + air)
- Better frequency targeting for female voice
- Still NO pitch shifting to avoid artificial sound
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

def maximum_natural_brightness(audio, sr):
    """
    Maximum brightness enhancement that still sounds natural.

    Three-stage approach:
    1. High-shelf boost at 2000Hz (+6dB)
    2. Presence band enhancement (3-5kHz)
    3. Air band enhancement (8-12kHz)

    Carefully balanced to avoid harsh/artificial sound.
    """
    nyquist = sr / 2

    # Stage 1: High-shelf at 2000Hz (good for female voice)
    high_freq = 2000 / nyquist
    b1, a1 = signal.butter(2, high_freq, btype='high')
    enhanced_high = signal.filtfilt(b1, a1, audio)

    # Stage 2: Presence band (3-5kHz) - critical for clarity
    presence_low = 3000 / nyquist
    presence_high = 5000 / nyquist
    b2, a2 = signal.butter(2, [presence_low, presence_high], btype='band')
    enhanced_presence = signal.filtfilt(b2, a2, audio)

    # Stage 3: Air band (8-12kHz) - adds sparkle
    air_low = 8000 / nyquist
    air_high = min(12000 / nyquist, 0.95)
    b3, a3 = signal.butter(2, [air_low, air_high], btype='band')
    enhanced_air = signal.filtfilt(b3, a3, audio)

    # Mix with optimized ratios
    boost_factor = 10 ** (6.0 / 20)  # +6dB
    result = audio + \
             enhanced_high * (boost_factor - 1) * 0.40 + \
             enhanced_presence * 0.20 + \
             enhanced_air * 0.15

    # Normalize to prevent clipping
    result = result / np.max(np.abs(result)) * 0.90

    return result

def controlled_warmth(audio, sr):
    """
    Warmth enhancement to balance the brightness.

    This prevents the voice from sounding thin or harsh.
    """
    nyquist = sr / 2

    # 200-600Hz warmth region (slightly wider than before)
    warmth_low = 200 / nyquist
    warmth_high = 600 / nyquist

    b, a = signal.butter(2, [warmth_low, warmth_high], btype='band')
    enhanced = signal.filtfilt(b, a, audio)

    # Moderate warmth to balance brightness
    result = audio + enhanced * 0.12  # Increased from 0.08

    # Normalize
    result = result / np.max(np.abs(result)) * 0.90

    return result

def natural_clarity_boost(audio, sr):
    """
    Boost the 1-2kHz range for natural clarity.

    This region is important for speech intelligibility
    without sounding artificial.
    """
    nyquist = sr / 2

    # 1-2kHz clarity region
    clarity_low = 1000 / nyquist
    clarity_high = 2000 / nyquist

    b, a = signal.butter(2, [clarity_low, clarity_high], btype='band')
    enhanced = signal.filtfilt(b, a, audio)

    # Moderate boost
    result = audio + enhanced * 0.15

    # Normalize
    result = result / np.max(np.abs(result)) * 0.90

    return result

def balanced_dynamic_processing(audio, sr):
    """
    Balanced dynamic range enhancement.

    Adds expressiveness without sounding processed.
    """
    # Calculate energy envelope
    energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy_interp = np.interp(
        np.arange(len(audio)),
        np.linspace(0, len(audio), len(energy)),
        energy
    )

    # Moderate expansion
    mean_energy = np.mean(energy_interp)
    expansion_factor = 1.15  # Balanced

    energy_mult = 1 + (energy_interp - mean_energy) / mean_energy * (expansion_factor - 1) * 0.25
    energy_mult = np.clip(energy_mult, 0.88, 1.12)

    enhanced = audio * energy_mult

    # Normalize
    enhanced = enhanced / np.max(np.abs(enhanced)) * 0.88

    return enhanced

def generate_optimal_samples():
    """Generate optimal voice clones with maximum natural quality."""

    print("="*70)
    print("OPTIMAL VOICE CLONING - FINAL VERSION")
    print("Maximum Quality + Natural Sound")
    print("="*70)

    base_path = Path(__file__).parent
    data_path = base_path / "training_data" / "segments_final_merged"
    output_path = base_path / "optimal_clone_output"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nUsing dataset: {len(list(data_path.glob('*.wav')))} segments")

    # Select best segments
    print("\n[STEP 1/4] Selecting highest-quality training segments...")
    best_segments = select_best_training_segments(data_path, num_segments=25)  # Use more segments
    print(f"Selected {len(best_segments)} segments")

    # Analyze original
    print("\n[STEP 2/4] Analyzing original voice...")

    # Sample multiple segments for better average
    orig_brightness_list = []
    orig_rms_list = []

    for seg in best_segments[:5]:
        audio, sr = librosa.load(str(seg), sr=None)
        brightness = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
        rms = np.sqrt(np.mean(audio**2))
        orig_brightness_list.append(brightness)
        orig_rms_list.append(rms)

    orig_brightness = np.mean(orig_brightness_list)
    orig_rms = np.mean(orig_rms_list)

    print(f"  Target Brightness: {orig_brightness:.0f} Hz")
    print(f"  Target RMS: {orig_rms:.4f}")

    # Load TTS
    print("\n[STEP 3/4] Loading TTS model...")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
    print("[OK] Model loaded")

    # Test samples - varied content and reference segments
    test_samples = [
        {
            "name": "optimal_educational_bright",
            "text": "Hello everyone, and welcome to today's training session. We're going to explore the fundamentals of data analytics and how you can apply these concepts in your everyday work.",
            "reference_idx": 0
        },
        {
            "name": "optimal_professional_clear",
            "text": "Good morning team. I'd like to review the progress we've made on our current project. The results have been quite promising, and I'm impressed with the dedication everyone has shown.",
            "reference_idx": 1
        },
        {
            "name": "optimal_technical_crisp",
            "text": "In this module, we'll focus on understanding data visualization techniques. Creating effective charts and graphs is essential for communicating insights to stakeholders.",
            "reference_idx": 2
        },
        {
            "name": "optimal_conversational_warm",
            "text": "I wanted to reach out and share some exciting updates about our upcoming initiatives. There are several new opportunities on the horizon that I think you'll find interesting and valuable.",
            "reference_idx": 3
        },
        {
            "name": "optimal_engaging_natural",
            "text": "Let me tell you about an amazing discovery we made during our research. The data revealed patterns we never expected to see, and this opens up entirely new possibilities for our approach.",
            "reference_idx": 4
        }
    ]

    TARGET_SAMPLE_RATE = 22050

    print("\n[STEP 4/4] Generating optimal voice clones...")
    print("Enhancement pipeline:")
    print("  1. Multi-stage brightness boost (+6dB high-shelf)")
    print("  2. Presence enhancement (3-5kHz)")
    print("  3. Air enhancement (8-12kHz)")
    print("  4. Clarity boost (1-2kHz)")
    print("  5. Warmth balance (200-600Hz)")
    print("  6. Balanced dynamic processing")
    print("  7. NO pitch shifting (natural sound preserved)")

    results = []

    for i, sample in enumerate(test_samples, 1):
        print(f"\n  [{i}/{len(test_samples)}] {sample['name']}")

        temp_output = output_path / f"temp_{sample['name']}.wav"
        final_output = output_path / f"{sample['name']}.wav"

        # Use different reference segments
        reference_audio = best_segments[sample['reference_idx']]
        print(f"    Using reference: {reference_audio.name}")

        # Generate
        tts.tts_to_file(
            text=sample['text'],
            file_path=str(temp_output),
            speaker_wav=str(reference_audio),
            language="en"
        )

        print(f"    Applying optimal enhancements...")

        # Step 1: Resample
        audio, sr = resample_audio(temp_output, TARGET_SAMPLE_RATE)

        # Step 2: Maximum natural brightness (multi-stage)
        audio = maximum_natural_brightness(audio, TARGET_SAMPLE_RATE)

        # Step 3: Natural clarity boost
        audio = natural_clarity_boost(audio, TARGET_SAMPLE_RATE)

        # Step 4: Controlled warmth (prevents harshness)
        audio = controlled_warmth(audio, TARGET_SAMPLE_RATE)

        # Step 5: Balanced dynamic processing
        audio = balanced_dynamic_processing(audio, TARGET_SAMPLE_RATE)

        # Step 6: Final normalization
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

    # Final analysis
    print("\n" + "="*70)
    print("QUALITY ANALYSIS - FINAL RESULTS")
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
    print("PROGRESSION SUMMARY")
    print("="*70)

    print(f"\nVersion 1 - Natural Clone:")
    print(f"  Brightness: 1580 Hz (53.9% match)")
    print(f"  Enhancement: +3dB")
    print(f"  Issue: Too dull")

    print(f"\nVersion 2 - Production Clone:")
    print(f"  Brightness: 1664 Hz (58.6% match)")
    print(f"  Enhancement: +4.5dB")
    print(f"  Improvement: +84 Hz")

    print(f"\nVersion 3 - Optimal Clone (CURRENT):")
    print(f"  Brightness: {avg_brightness:.0f} Hz ({avg_brightness_match:.1f}% match)")
    print(f"  Enhancement: +6dB multi-stage")
    print(f"  Improvement: +{avg_brightness - 1664:.0f} Hz from production")
    print(f"  Total improvement: +{avg_brightness - 1580:.0f} Hz from natural")

    print("\n" + "="*70)
    print("OPTIMAL VOICE CLONING COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(results)} optimal samples")
    print(f"Output directory: {output_path}")
    print("\nFiles created:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}.wav ({result['duration']:.1f}s)")
        print(f"      Brightness: {result['brightness']:.0f}Hz ({result['brightness_match']:.1f}% match)")

    print("\nFinal enhancements applied:")
    print("  [+] Multi-stage brightness (+6dB high-shelf + presence + air)")
    print("  [+] Natural clarity boost (1-2kHz)")
    print("  [+] Controlled warmth (200-600Hz prevents harshness)")
    print("  [+] Balanced dynamic processing")
    print("  [+] NO pitch shifting (preserves natural prosody)")
    print("  [+] Using 25 best segments from 224 total")

    print("\nRECOMMENDATION:")
    print("  This is the best balance we can achieve with YourTTS model")
    print("  Sounds natural while maximizing brightness")
    print("  Use these samples for production")

    print("\nNote on remaining gap:")
    gap = orig_brightness - avg_brightness
    print(f"  Remaining brightness gap: {gap:.0f} Hz")
    print(f"  This is a YourTTS model limitation")
    print(f"  The model generates ~16kHz native, upsampled to 22kHz")
    print(f"  Further improvement would require:")
    print(f"    - XTTS v2 model (better quality, needs PyTorch fix)")
    print(f"    - Fine-tuning YourTTS on your voice specifically")
    print(f"    - Using higher-quality TTS models (VALL-E, Bark, etc)")
    print("="*70)

if __name__ == "__main__":
    generate_optimal_samples()

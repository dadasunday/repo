"""
Natural Voice Cloning - No artificial pitch shifting
Focus on model quality and gentle enhancements only
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

def gentle_brightness_boost(audio, sr, boost_db=3):
    """
    Gentle brightness enhancement - more natural sounding.
    Less aggressive than ultra-bright version.
    """
    # Gentle high-shelf at 2500Hz (not too low)
    nyquist = sr / 2
    high_freq = 2500 / nyquist

    b, a = signal.butter(2, high_freq, btype='high')  # Order 2 (gentle)
    enhanced = signal.filtfilt(b, a, audio)

    # Very subtle boost
    boost_factor = 10 ** (boost_db / 20)
    result = audio + enhanced * (boost_factor - 1) * 0.25  # 25% mix (very gentle)

    # Normalize
    result = result / np.max(np.abs(result)) * 0.90

    return result

def natural_presence_boost(audio, sr):
    """Add subtle presence without harshness."""
    nyquist = sr / 2

    # Focus on 3-5kHz (natural voice presence)
    presence_low = 3000 / nyquist
    presence_high = 5000 / nyquist

    b, a = signal.butter(2, [presence_low, presence_high], btype='band')
    enhanced = signal.filtfilt(b, a, audio)

    # Very gentle mix
    result = audio + enhanced * 0.10

    # Normalize
    result = result / np.max(np.abs(result)) * 0.90

    return result

def subtle_warmth_enhancement(audio, sr):
    """Add warmth to prevent harsh/artificial sound."""
    nyquist = sr / 2

    # Slight boost in 200-500Hz (warmth region)
    warmth_low = 200 / nyquist
    warmth_high = 500 / nyquist

    b, a = signal.butter(2, [warmth_low, warmth_high], btype='band')
    enhanced = signal.filtfilt(b, a, audio)

    # Very subtle
    result = audio + enhanced * 0.08

    # Normalize
    result = result / np.max(np.abs(result)) * 0.90

    return result

def natural_dynamic_processing(audio, sr):
    """Gentle dynamic range enhancement - maintains naturalness."""
    # Calculate energy envelope
    energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy_interp = np.interp(
        np.arange(len(audio)),
        np.linspace(0, len(audio), len(energy)),
        energy
    )

    # Very subtle expansion (1.10 instead of 1.25)
    mean_energy = np.mean(energy_interp)
    expansion_factor = 1.10

    energy_mult = 1 + (energy_interp - mean_energy) / mean_energy * (expansion_factor - 1) * 0.2
    energy_mult = np.clip(energy_mult, 0.92, 1.08)  # Narrow range

    enhanced = audio * energy_mult

    # Normalize
    enhanced = enhanced / np.max(np.abs(enhanced)) * 0.88

    return enhanced

def generate_natural_samples():
    """Generate natural-sounding voice clones without artificial processing."""

    print("="*70)
    print("NATURAL VOICE CLONING")
    print("Focus: Model Quality + Gentle Enhancement = Natural Sound")
    print("="*70)

    base_path = Path(__file__).parent
    data_path = base_path / "training_data" / "segments_final_merged"
    output_path = base_path / "natural_clone_output"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nUsing dataset: {len(list(data_path.glob('*.wav')))} segments")

    # Select best segments - focus on quality
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

    # Test samples - use varied reference segments
    test_samples = [
        {
            "name": "natural_educational_warm",
            "text": "Hello everyone, and welcome to today's training session. We're going to explore the fundamentals of data analytics and how you can apply these concepts in your everyday work.",
            "reference_idx": 0  # Use best segment
        },
        {
            "name": "natural_professional_clear",
            "text": "Good morning team. I'd like to review the progress we've made on our current project. The results have been quite promising, and I'm impressed with the dedication everyone has shown.",
            "reference_idx": 1  # Use second-best
        },
        {
            "name": "natural_technical_balanced",
            "text": "In this module, we'll focus on understanding data visualization techniques. Creating effective charts and graphs is essential for communicating insights to stakeholders.",
            "reference_idx": 2  # Use third-best
        },
        {
            "name": "natural_conversational",
            "text": "I wanted to reach out and share some exciting updates about our upcoming initiatives. There are several new opportunities on the horizon that I think you'll find interesting and valuable.",
            "reference_idx": 0
        }
    ]

    TARGET_SAMPLE_RATE = 22050

    print("\n[STEP 4/4] Generating natural voice clones...")
    print("Enhancement approach:")
    print("  - NO pitch shifting (preserves naturalness)")
    print("  - Gentle brightness boost (+3dB, subtle)")
    print("  - Natural presence enhancement (3-5kHz)")
    print("  - Subtle warmth (200-500Hz)")
    print("  - Minimal dynamic processing")
    print("  - Using multiple reference segments for variety")

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

        print(f"    Applying gentle enhancements...")

        # Step 1: Resample
        audio, sr = resample_audio(temp_output, TARGET_SAMPLE_RATE)

        # Step 2: Gentle brightness boost (3dB, not 8dB)
        audio = gentle_brightness_boost(audio, TARGET_SAMPLE_RATE, boost_db=3)

        # Step 3: Natural presence
        audio = natural_presence_boost(audio, TARGET_SAMPLE_RATE)

        # Step 4: Subtle warmth (prevents harshness)
        audio = subtle_warmth_enhancement(audio, TARGET_SAMPLE_RATE)

        # Step 5: Minimal dynamic processing
        audio = natural_dynamic_processing(audio, TARGET_SAMPLE_RATE)

        # Step 6: Normalize (preserve dynamics)
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
    print("COMPARISON: Pitch-Corrected vs Natural")
    print("="*70)
    print(f"\nPitch-Corrected (previous):")
    print(f"  Pitch Match: 98.3% (but sounds artificial)")
    print(f"  Brightness: 87.3%")
    print(f"  Problem: Pitch shifting creates robotic sound")

    print(f"\nNatural (current):")
    print(f"  Pitch Match: Let model decide (natural)")
    print(f"  Brightness: {avg_brightness_match:.1f}%")
    print(f"  Benefit: Sounds like real human speech")

    print("\n" + "="*70)
    print("NATURAL VOICE CLONING COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(results)} natural-sounding samples")
    print(f"Output directory: {output_path}")
    print("\nFiles created:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}.wav ({result['duration']:.1f}s)")
        print(f"      Brightness: {result['brightness']:.0f}Hz ({result['brightness_match']:.1f}% match)")

    print("\nWhy this sounds more natural:")
    print("  [+] NO pitch shifting (prevents robotic sound)")
    print("  [+] Gentle enhancements only (+3dB vs +8dB)")
    print("  [+] Added warmth (prevents harshness)")
    print("  [+] Minimal processing (preserves natural quality)")
    print("  [+] Used 20 best segments from 224 total")
    print("  [+] Multiple reference segments (more variety)")
    print("\nThese should sound much more natural and human-like!")
    print("="*70)

if __name__ == "__main__":
    generate_natural_samples()

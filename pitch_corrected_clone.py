"""
Pitch-Corrected Voice Cloning - Fix the 113Hz pitch gap
Combines ultra-bright enhancement with pitch correction
"""

from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
from TTS.api import TTS

# Import from previous scripts
from improved_voice_clone import (
    select_best_training_segments,
    normalize_audio,
    resample_audio
)

from ultra_bright_clone import (
    ultra_enhance_brightness,
    enhance_expressiveness_advanced,
    apply_harmonic_enhancement
)

def shift_pitch(audio, sr, n_steps):
    """
    Shift pitch by n_steps semitones.
    Positive = higher pitch, Negative = lower pitch
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def calculate_pitch_shift_needed(original_pitch, clone_pitch):
    """Calculate semitones needed to match original pitch."""
    # Formula: semitones = 12 * log2(target_freq / current_freq)
    if clone_pitch > 0:
        semitones = 12 * np.log2(original_pitch / clone_pitch)
        return semitones
    return 0

def analyze_pitch(audio, sr):
    """Analyze average pitch of audio."""
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=500)
    pitch_values = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    if pitch_values:
        return np.mean(pitch_values)
    return 0

def generate_pitch_corrected_samples():
    """Generate voice clones with pitch correction."""

    print("="*70)
    print("PITCH-CORRECTED VOICE CLONING")
    print("Ultra-Bright Enhancement + Pitch Correction")
    print("="*70)

    base_path = Path(__file__).parent
    data_path = base_path / "training_data" / "segments_final_merged"  # NEW expanded dataset!
    output_path = base_path / "pitch_corrected_output"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nUsing expanded dataset: {len(list(data_path.glob('*.wav')))} segments")

    # Select best segments
    print("\n[STEP 1/5] Selecting best training segments...")
    best_segments = select_best_training_segments(data_path, num_segments=15)  # More segments!
    print(f"Selected {len(best_segments)} high-quality segments")

    # Analyze original pitch
    print("\n[STEP 2/5] Analyzing original voice characteristics...")
    orig_audio, orig_sr = librosa.load(str(best_segments[0]), sr=None)
    orig_pitch = analyze_pitch(orig_audio, orig_sr)
    orig_brightness = np.mean(librosa.feature.spectral_centroid(y=orig_audio, sr=orig_sr)[0])

    print(f"  Target Pitch: {orig_pitch:.1f} Hz")
    print(f"  Target Brightness: {orig_brightness:.0f} Hz")

    # Load TTS
    print("\n[STEP 3/5] Loading TTS model...")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
    print("[OK] Model loaded")

    # Test samples
    test_samples = [
        {
            "name": "pitch_corrected_educational",
            "text": "Hello everyone, and welcome to today's training session. We're going to explore the fundamentals of data analytics and how you can apply these concepts in your everyday work. Let's begin by understanding the key principles."
        },
        {
            "name": "pitch_corrected_technical",
            "text": "In this module, we'll focus on understanding data visualization techniques. Creating effective charts and graphs is essential for communicating insights to stakeholders. Remember, clarity is key."
        },
        {
            "name": "pitch_corrected_professional",
            "text": "Good morning team. I'd like to review the progress we've made on our current project. The results have been quite promising, and I'm impressed with the dedication everyone has shown. Let's maintain this momentum."
        }
    ]

    TARGET_SAMPLE_RATE = 22050

    # Generate clones with pitch correction
    print("\n[STEP 4/5] Generating pitch-corrected voice clones...")
    print("Enhancement pipeline:")
    print("  1. Generate with YourTTS")
    print("  2. Ultra-bright enhancement (+8dB)")
    print("  3. Analyze clone pitch")
    print("  4. Pitch shift to match original")
    print("  5. Final normalization")

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

        print(f"  Processing...")

        # Step 1: Resample
        audio, sr = resample_audio(temp_output, TARGET_SAMPLE_RATE)

        # Step 2: Analyze clone pitch BEFORE enhancement
        clone_pitch_before = analyze_pitch(audio, TARGET_SAMPLE_RATE)
        print(f"    Clone pitch (before): {clone_pitch_before:.1f}Hz")

        # Step 3: Ultra brightness restoration
        audio = ultra_enhance_brightness(audio, TARGET_SAMPLE_RATE, boost_db=8)

        # Step 4: Harmonic enhancement
        audio = apply_harmonic_enhancement(audio, TARGET_SAMPLE_RATE)

        # Step 5: PITCH CORRECTION
        # Calculate how many semitones to shift
        semitones_needed = calculate_pitch_shift_needed(orig_pitch, clone_pitch_before)
        print(f"    Pitch correction: {semitones_needed:+.2f} semitones")

        # Apply pitch shift
        audio = shift_pitch(audio, TARGET_SAMPLE_RATE, semitones_needed)

        # Step 6: Verify pitch after correction
        clone_pitch_after = analyze_pitch(audio, TARGET_SAMPLE_RATE)
        print(f"    Clone pitch (after): {clone_pitch_after:.1f}Hz (Target: {orig_pitch:.1f}Hz)")

        # Step 7: Advanced expressiveness
        audio = enhance_expressiveness_advanced(audio, TARGET_SAMPLE_RATE)

        # Step 8: Normalize
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_normalized = normalize_audio(audio_int16, target_rms=0.08)

        # Save
        sf.write(str(final_output), audio_normalized, TARGET_SAMPLE_RATE)

        # Cleanup
        if temp_output.exists():
            temp_output.unlink()

        # Analyze final result
        audio_float = audio_normalized.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float**2))
        duration = len(audio_normalized) / TARGET_SAMPLE_RATE
        brightness = np.mean(librosa.feature.spectral_centroid(
            y=audio_float, sr=TARGET_SAMPLE_RATE)[0])
        final_pitch = analyze_pitch(audio_float, TARGET_SAMPLE_RATE)

        pitch_match = 100 - abs(orig_pitch - final_pitch) / orig_pitch * 100
        brightness_match = 100 - abs(orig_brightness - brightness) / orig_brightness * 100

        results.append({
            'name': sample['name'],
            'pitch': final_pitch,
            'brightness': brightness,
            'rms': rms,
            'duration': duration,
            'pitch_match': pitch_match,
            'brightness_match': brightness_match
        })

        print(f"  [OK] Pitch: {final_pitch:.1f}Hz ({pitch_match:.1f}% match)")
        print(f"       Brightness: {brightness:.0f}Hz ({brightness_match:.1f}% match)")
        print(f"       Duration: {duration:.2f}s")

    # Step 5: Final analysis
    print("\n[STEP 5/5] Quality Analysis")
    print("="*70)

    avg_pitch = np.mean([r['pitch'] for r in results])
    avg_brightness = np.mean([r['brightness'] for r in results])
    avg_pitch_match = np.mean([r['pitch_match'] for r in results])
    avg_brightness_match = np.mean([r['brightness_match'] for r in results])

    print(f"\n{'Metric':<25} {'Target':>15} {'Achieved':>15} {'Match':>10}")
    print("-"*70)
    print(f"{'Pitch':<25} {orig_pitch:>14.1f} Hz {avg_pitch:>14.1f} Hz {avg_pitch_match:>9.1f}%")
    print(f"{'Brightness':<25} {orig_brightness:>14.0f} Hz {avg_brightness:>14.0f} Hz {avg_brightness_match:>9.1f}%")
    print(f"{'Sample Rate':<25} {TARGET_SAMPLE_RATE:>14} Hz {TARGET_SAMPLE_RATE:>14} Hz {'100.0':>9}%")

    # Comparison
    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    print(f"\nPITCH:")
    print(f"  Original target:     {orig_pitch:.1f} Hz")
    print(f"  Previous (no fix):   ~187 Hz (37.6% gap)")
    print(f"  Pitch-corrected:     {avg_pitch:.1f} Hz ({avg_pitch_match:.1f}% match)")
    print(f"  Improvement: {avg_pitch_match - 62.4:.1f} percentage points!")

    print(f"\nBRIGHTNESS:")
    print(f"  Original target:     {orig_brightness:.0f} Hz")
    print(f"  Pitch-corrected:     {avg_brightness:.0f} Hz ({avg_brightness_match:.1f}% match)")

    # Summary
    print("\n" + "="*70)
    print("PITCH-CORRECTED VOICE CLONING COMPLETE!")
    print("="*70)
    print(f"\nDataset: {len(list(data_path.glob('*.wav')))} segments (EXPANDED!)")
    print(f"Generated {len(results)} pitch-corrected samples")
    print(f"Output directory: {output_path}")
    print("\nFiles created:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}.wav")
        print(f"      Pitch: {result['pitch']:.1f}Hz ({result['pitch_match']:.1f}% match)")
        print(f"      Brightness: {result['brightness']:.0f}Hz ({result['brightness_match']:.1f}% match)")

    print("\nEnhancements applied:")
    print("  [+] 224 training segments (massive expansion!)")
    print("  [+] Ultra-bright enhancement (+8dB)")
    print("  [+] Automatic pitch correction")
    print("  [+] Harmonic enhancement")
    print("  [+] Advanced expressiveness")
    print("\nThese should match your voice MUCH better now!")
    print("="*70)

if __name__ == "__main__":
    generate_pitch_corrected_samples()

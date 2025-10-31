"""
Generate voice clone using realistic training text samples
This uses text that matches the style/content of the training data
"""

from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import shutil
from TTS.api import TTS

# Import from improved_voice_clone
from improved_voice_clone import (
    select_best_training_segments,
    enhance_brightness,
    enhance_expressiveness,
    normalize_audio,
    resample_audio
)

def generate_clone_samples():
    """Generate multiple voice clones with realistic text."""

    print("="*70)
    print("VOICE CLONING WITH REALISTIC TEXT SAMPLES")
    print("="*70)

    base_path = Path(__file__).parent
    data_path = base_path / "training_data" / "segments_merged"
    output_path = base_path / "realistic_clone_output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Select best segments
    print("\n[STEP 1/4] Selecting best training segments...")
    best_segments = select_best_training_segments(data_path, num_segments=10)
    print(f"Selected {len(best_segments)} segments")

    # Load TTS
    print("\n[STEP 2/4] Loading TTS model...")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
    print("[OK] Model loaded")

    # Realistic text samples matching training data style
    # These are professional/educational content samples
    test_samples = [
        {
            "name": "educational_sample_1",
            "text": "Hello everyone, and welcome to today's training session. We're going to explore the fundamentals of data analytics and how you can apply these concepts in your everyday work. Let's begin by understanding the key principles that will guide us through this learning journey."
        },
        {
            "name": "educational_sample_2",
            "text": "In this module, we'll focus on understanding data visualization techniques. Creating effective charts and graphs is essential for communicating insights to stakeholders. Remember, the goal is to make complex information easily digestible for your audience."
        },
        {
            "name": "educational_sample_3",
            "text": "Now let's discuss the importance of data quality and accuracy. When working with datasets, it's crucial to validate your sources and ensure the information you're analyzing is reliable. This foundational step can make or break your entire analysis."
        },
        {
            "name": "professional_sample_1",
            "text": "Good morning team. I'd like to review the progress we've made on our current project. The results have been quite promising, and I'm impressed with the dedication everyone has shown. Let's continue this momentum as we move forward."
        },
        {
            "name": "conversational_sample",
            "text": "I hope this message finds you well. I wanted to reach out and share some exciting updates about our upcoming initiatives. There are several new opportunities on the horizon that I think you'll find interesting and valuable."
        }
    ]

    TARGET_SAMPLE_RATE = 22050

    # Generate clones
    print("\n[STEP 3/4] Generating voice clones...")

    results = []

    for i, sample in enumerate(test_samples, 1):
        print(f"\n  [{i}/{len(test_samples)}] {sample['name']}")
        print(f"  Text: {sample['text'][:80]}...")

        temp_output = output_path / f"temp_{sample['name']}.wav"
        final_output = output_path / f"{sample['name']}.wav"

        # Use best reference
        reference_audio = best_segments[0]

        # Generate
        tts.tts_to_file(
            text=sample['text'],
            file_path=str(temp_output),
            speaker_wav=str(reference_audio),
            language="en"
        )

        # Post-process
        print(f"  Processing...")

        # Resample
        audio, sr = resample_audio(temp_output, TARGET_SAMPLE_RATE)

        # Enhancements
        audio = enhance_brightness(audio, TARGET_SAMPLE_RATE, boost_db=5)
        audio = enhance_expressiveness(audio, TARGET_SAMPLE_RATE)

        # Normalize
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
            'duration': duration,
            'rms': rms,
            'brightness': brightness,
            'file': final_output
        })

        print(f"  [OK] Duration: {duration:.2f}s | RMS: {rms:.4f} | Brightness: {brightness:.0f}Hz")

    # Step 4: Analysis
    print("\n[STEP 4/4] Quality Analysis")
    print("="*70)

    # Load original training sample for comparison
    orig_audio, orig_sr = librosa.load(str(best_segments[0]), sr=None)
    orig_rms = np.sqrt(np.mean(orig_audio**2))
    orig_brightness = np.mean(librosa.feature.spectral_centroid(y=orig_audio, sr=orig_sr)[0])

    print(f"\nOriginal Training Sample:")
    print(f"  Sample Rate: {orig_sr} Hz")
    print(f"  RMS: {orig_rms:.4f}")
    print(f"  Brightness: {orig_brightness:.0f} Hz")

    print(f"\nGenerated Clones (Average):")
    avg_rms = np.mean([r['rms'] for r in results])
    avg_brightness = np.mean([r['brightness'] for r in results])

    print(f"  Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"  RMS: {avg_rms:.4f}")
    print(f"  Brightness: {avg_brightness:.0f} Hz")

    # Calculate matches
    def calc_match(orig, clone):
        return 100 - abs(orig - clone) / orig * 100 if orig > 0 else 0

    print(f"\nQuality Metrics:")
    print(f"  Sample Rate Match: {calc_match(orig_sr, TARGET_SAMPLE_RATE):.1f}%")
    print(f"  RMS Match: {calc_match(orig_rms, avg_rms):.1f}%")
    print(f"  Brightness Match: {calc_match(orig_brightness, avg_brightness):.1f}%")

    # Summary
    print("\n" + "="*70)
    print("VOICE CLONING COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(results)} high-quality voice samples")
    print(f"Output directory: {output_path}")
    print("\nFiles created:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['file'].name} ({result['duration']:.1f}s)")

    print("\nThese samples use realistic educational/professional text")
    print("that matches the style of your training data!")
    print("="*70)

if __name__ == "__main__":
    generate_clone_samples()

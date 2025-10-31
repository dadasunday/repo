"""
Transcribe original audio and use exact text for voice cloning comparison
"""

from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import os
import shutil

# Import functions from improved_voice_clone
from improved_voice_clone import (
    select_best_training_segments,
    enhance_brightness,
    enhance_expressiveness,
    normalize_audio,
    resample_audio
)

def transcribe_audio_whisper(audio_path):
    """Transcribe audio using Whisper (OpenAI's speech recognition)."""
    try:
        import whisper
        print(f"Loading Whisper model...")
        model = whisper.load_model("base")

        print(f"Transcribing: {audio_path.name}")
        result = model.transcribe(str(audio_path))

        return result["text"], result.get("segments", [])
    except ImportError:
        print("[ERROR] Whisper not installed.")
        print("Install with: pip install openai-whisper")
        return None, None
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None, None

def transcribe_audio_sphinx(audio_path):
    """Transcribe audio using pocketsphinx (offline, lighter alternative)."""
    try:
        import speech_recognition as sr

        print(f"Using speech_recognition library...")
        recognizer = sr.Recognizer()

        with sr.AudioFile(str(audio_path)) as source:
            audio = recognizer.record(source)

        print(f"Transcribing: {audio_path.name}")
        text = recognizer.recognize_google(audio)

        return text, None
    except ImportError:
        print("[ERROR] speech_recognition not installed.")
        print("Install with: pip install SpeechRecognition")
        return None, None
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None, None

def clone_with_transcribed_text(text, tts_model, reference_audio, output_path, sample_name):
    """Generate voice clone using transcribed text."""
    from TTS.api import TTS

    print(f"\n  Generating: {sample_name}")
    print(f"  Text: {text[:100]}...")

    temp_output = output_path / f"temp_{sample_name}.wav"
    final_output = output_path / f"{sample_name}.wav"

    TARGET_SAMPLE_RATE = 22050

    # Generate speech
    tts_model.tts_to_file(
        text=text,
        file_path=str(temp_output),
        speaker_wav=str(reference_audio),
        language="en"
    )

    # Enhanced post-processing
    print(f"  Post-processing...")

    # Resample
    audio, sr = resample_audio(temp_output, TARGET_SAMPLE_RATE)

    # Brightness restoration
    audio = enhance_brightness(audio, TARGET_SAMPLE_RATE, boost_db=5)

    # Expressiveness
    audio = enhance_expressiveness(audio, TARGET_SAMPLE_RATE)

    # Normalize
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_normalized = normalize_audio(audio_int16, target_rms=0.08)

    # Save
    sf.write(str(final_output), audio_normalized, TARGET_SAMPLE_RATE)

    # Remove temp
    if temp_output.exists():
        temp_output.unlink()

    # Analyze
    audio_float = audio_normalized.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(audio_float**2))
    duration = len(audio_normalized) / TARGET_SAMPLE_RATE
    brightness = np.mean(librosa.feature.spectral_centroid(
        y=audio_float, sr=TARGET_SAMPLE_RATE)[0])

    print(f"  [OK] Saved: {sample_name}.wav")
    print(f"       Duration: {duration:.2f}s | RMS: {rms:.4f} | Brightness: {brightness:.0f}Hz")

    return final_output

def compare_original_vs_clone(original_path, clone_path):
    """Compare original and cloned audio."""
    print("\n" + "="*70)
    print("QUALITY COMPARISON: Original vs Clone")
    print("="*70)

    # Load audios
    orig_audio, orig_sr = librosa.load(str(original_path), sr=None)
    clone_audio, clone_sr = librosa.load(str(clone_path), sr=None)

    # Calculate metrics
    orig_rms = np.sqrt(np.mean(orig_audio**2))
    clone_rms = np.sqrt(np.mean(clone_audio**2))

    orig_brightness = np.mean(librosa.feature.spectral_centroid(y=orig_audio, sr=orig_sr)[0])
    clone_brightness = np.mean(librosa.feature.spectral_centroid(y=clone_audio, sr=clone_sr)[0])

    # Pitch
    orig_pitches, orig_mags = librosa.piptrack(y=orig_audio, sr=orig_sr, fmin=50, fmax=500)
    orig_pitch_values = [orig_pitches[orig_mags[:, t].argmax(), t]
                         for t in range(orig_pitches.shape[1])
                         if orig_pitches[orig_mags[:, t].argmax(), t] > 0]
    orig_pitch = np.mean(orig_pitch_values) if orig_pitch_values else 0

    clone_pitches, clone_mags = librosa.piptrack(y=clone_audio, sr=clone_sr, fmin=50, fmax=500)
    clone_pitch_values = [clone_pitches[clone_mags[:, t].argmax(), t]
                          for t in range(clone_pitches.shape[1])
                          if clone_pitches[clone_mags[:, t].argmax(), t] > 0]
    clone_pitch = np.mean(clone_pitch_values) if clone_pitch_values else 0

    # Calculate matches
    def calc_match(orig, clone):
        return 100 - abs(orig - clone) / orig * 100 if orig > 0 else 0

    print(f"\n{'Metric':<20} {'Original':>15} {'Clone':>15} {'Match':>10}")
    print("-"*70)
    print(f"{'Sample Rate':<20} {orig_sr:>14} Hz {clone_sr:>14} Hz {calc_match(orig_sr, clone_sr):>9.1f}%")
    print(f"{'Pitch':<20} {orig_pitch:>14.1f} Hz {clone_pitch:>14.1f} Hz {calc_match(orig_pitch, clone_pitch):>9.1f}%")
    print(f"{'Brightness':<20} {orig_brightness:>14.0f} Hz {clone_brightness:>14.0f} Hz {calc_match(orig_brightness, clone_brightness):>9.1f}%")
    print(f"{'RMS Energy':<20} {orig_rms:>15.4f} {clone_rms:>15.4f} {calc_match(orig_rms, clone_rms):>9.1f}%")

    overall = (calc_match(orig_sr, clone_sr) + calc_match(orig_pitch, clone_pitch) +
               calc_match(orig_brightness, clone_brightness) + calc_match(orig_rms, clone_rms)) / 4

    print("-"*70)
    print(f"{'OVERALL MATCH':<20} {' ':>15} {' ':>15} {overall:>9.1f}%")
    print("="*70)

    return overall

def main():
    print("="*70)
    print("TRANSCRIBE & CLONE: Perfect Comparison Test")
    print("="*70)

    base_path = Path(__file__).parent
    data_path = base_path / "training_data" / "segments_merged"
    output_path = base_path / "transcribed_clone_output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Select best segment for transcription
    print("\n[STEP 1/5] Selecting best audio segment for transcription...")
    best_segments = select_best_training_segments(data_path, num_segments=3)

    transcribe_segment = best_segments[0]  # Use the best one
    print(f"Selected: {transcribe_segment.name}")

    # Copy to output for comparison
    original_comparison = output_path / "original_sample.wav"
    shutil.copy2(transcribe_segment, original_comparison)

    # Step 2: Transcribe the audio
    print("\n[STEP 2/5] Transcribing audio...")
    print("Attempting transcription methods...")

    text = None
    segments = None

    # Try Whisper first (best quality)
    print("\n  Method 1: Whisper (OpenAI)")
    text, segments = transcribe_audio_whisper(transcribe_segment)

    # If Whisper fails, try speech_recognition
    if text is None:
        print("\n  Method 2: Google Speech Recognition")
        text, _ = transcribe_audio_sphinx(transcribe_segment)

    if text is None:
        print("\n[ERROR] All transcription methods failed.")
        print("Please install one of:")
        print("  1. pip install openai-whisper (recommended)")
        print("  2. pip install SpeechRecognition")
        return

    print(f"\n[OK] Transcription complete!")
    print(f"Text length: {len(text)} characters")
    print(f"\nTranscribed text:")
    print("-"*70)
    print(text)
    print("-"*70)

    # Save transcription
    transcript_file = output_path / "transcription.txt"
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nSaved transcription to: {transcript_file}")

    # Step 3: Load TTS model
    print("\n[STEP 3/5] Loading TTS model...")
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)
    print("[OK] Model loaded")

    # Step 4: Generate clone with transcribed text
    print("\n[STEP 4/5] Generating voice clone with transcribed text...")

    # Use the transcribed segment as reference
    reference_audio = best_segments[0]

    cloned_output = clone_with_transcribed_text(
        text=text,
        tts_model=tts,
        reference_audio=reference_audio,
        output_path=output_path,
        sample_name="transcribed_clone"
    )

    # Step 5: Compare
    print("\n[STEP 5/5] Comparing original vs cloned...")

    overall_match = compare_original_vs_clone(original_comparison, cloned_output)

    # Final summary
    print("\n" + "="*70)
    print("TRANSCRIPTION & CLONING COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  1. {original_comparison.name} - Original audio")
    print(f"  2. {cloned_output.name} - Cloned audio (same text)")
    print(f"  3. {transcript_file.name} - Transcribed text")
    print(f"\nOverall quality match: {overall_match:.1f}%")
    print(f"\nOutput directory: {output_path}")
    print("\nNow listen to both files side-by-side to hear the difference!")
    print("="*70)

if __name__ == "__main__":
    main()

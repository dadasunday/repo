"""Demo script showing voice cloning workflow."""

import argparse
from pathlib import Path

from src.utils.audio import extract_audio, load_voice_model, prepare_audio


def main():
    parser = argparse.ArgumentParser(description="Demo voice cloning workflow")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input video/audio file"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of voice cloning.",
        help="Text to synthesize with the cloned voice"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    args = parser.parse_args()
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing input file: {args.input}")
    
    # Extract audio if input is video
    input_path = Path(args.input)
    if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        print("Extracting audio from video...")
        audio_path = extract_audio(
            input_path,
            output_path=output_dir / f"{input_path.stem}.wav"
        )
    else:
        audio_path = input_path
    
    # Process audio
    print("Processing audio...")
    processed_path = prepare_audio(
        audio_path,
        output_path=output_dir / f"{input_path.stem}_processed.wav"
    )
    
    # Load model and generate speech
    print("Loading TTS model...")
    synthesizer = load_voice_model()
    
    print(f"Generating speech for text: {args.text}")
    
    # Generate with reference audio (voice cloning)
    wav = synthesizer.tts(
        text=args.text,
        speaker_wav=str(processed_path),
    )
    
    # Save output
    output_path = output_dir / f"output_{input_path.stem}.wav"
    synthesizer.save_wav(wav, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")
    

if __name__ == "__main__":
    main()
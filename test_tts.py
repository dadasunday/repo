from TTS.api import TTS

# Initialize TTS with a simple English TTS model
tts = TTS('tts_models/en/ljspeech/tacotron2-DDC')

# Text to convert to speech
text = "Hello! This is a test of the text to speech system. I hope it works well!"

# Generate speech
output_path = "test_output.wav"
tts.tts_to_file(text=text, file_path=output_path)
print(f"Generated speech saved to {output_path}")
# Voice Cloning Usage Guide

## Quick Start

Generate high-quality voice clones with a single command:

```bash
python improved_voice_clone.py
```

This will create 3 sample audio files in `improved_output/` directory.

## Files Overview

### Main Script
- **`improved_voice_clone.py`** - Enhanced voice cloning with quality optimization
  - Auto-selects best training segments
  - Applies brightness restoration
  - Enhances expressiveness
  - Outputs at 22050Hz (matches training data)

### Analysis Tool
- **`final_comparison.py`** - Compare voice cloning quality metrics
  - Analyzes pitch, brightness, RMS, sample rate
  - Shows improvement percentages

### Output Files (Best Quality)
- `improved_output/improved_sample_1.wav` - Primary test sample
- `improved_output/improved_sample_2.wav` - Expressive variation test
- `improved_output/improved_sample_3.wav` - Comprehensive quality test

### Training Data
- `training_data/raw_audio.wav` - Original voice recording (6 minutes)
- `training_data/segments/` - Processed audio segments for training

## Customizing Text Generation

Edit `improved_voice_clone.py` and modify the `test_configs` list (around line 157):

```python
test_configs = [
    {
        "text": "Your custom text here",
        "name": "my_custom_output"
    },
    # Add more configs as needed
]
```

Then run:
```bash
python improved_voice_clone.py
```

Your custom audio will be saved to `improved_output/my_custom_output.wav`

## Quality Metrics

### Original Voice Characteristics
- **Gender:** Female
- **Voice Type:** Mezzo-soprano (243.5 Hz)
- **Tone:** Bright and Clear (2570 Hz)
- **Expression:** Highly expressive
- **Quality:** Professional, engaging

### Enhanced Clone Quality
- **Sample Rate Match:** 100% (22050 Hz)
- **RMS Energy Match:** 91%
- **Brightness Match:** 65%
- **Overall Quality:** 82%

## Enhancements Applied

1. **Smart Segment Selection** - Analyzes and selects highest quality training segments
2. **Brightness Restoration** - +5dB high-shelf filter to restore clarity
3. **Mid-range Boost** - 1-3kHz enhancement for better articulation
4. **Expressiveness Enhancement** - Dynamic range expansion
5. **Smart Normalization** - Prevents clipping while preserving dynamics
6. **Sample Rate Matching** - Upsamples to match training data (22050Hz)

## Known Limitations

- Brightness ~35% below original (YourTTS model limitation)
- Prosody/emotional variation simplified compared to original
- For best results, provide varied training data with consistent quality

## Tips for Better Results

1. **Training Data Quality:**
   - Use clear, high-quality audio (22050Hz or higher)
   - Minimum 5 minutes of speech recommended
   - Consistent background/recording conditions
   - Natural, expressive speech

2. **Generation Quality:**
   - Keep sentences moderate length (10-20 words)
   - Avoid complex punctuation
   - Test different reference segments for different styles

3. **Post-Processing:**
   - Adjust brightness boost in `enhance_brightness()` function
   - Modify RMS target in `normalize_audio()` function
   - Experiment with dynamic range expansion factor

## Troubleshooting

**Low brightness/dull sound:**
- Increase `boost_db` parameter in `enhance_brightness()` (line 92)
- Current: 5dB, try: 6-8dB

**Audio clipping:**
- Reduce `target_rms` in `normalize_audio()` (line 135)
- Current: 0.08, try: 0.06-0.07

**Voice doesn't match:**
- Check training data quality
- Ensure segments are from same speaker
- Try using more/different reference segments

## Requirements

See `requirements.txt` for full dependencies. Main packages:
- TTS (Coqui TTS)
- torch
- librosa
- soundfile
- scipy
- numpy

## Credits

Built with Coqui TTS and the YourTTS model for multilingual voice cloning.

# Voice Generation Audio Quality Fix Workflow

## Problem Summary

Your training recordings have **clipped peaks (0 dBFS)**, high brightness (~3.0 kHz), and crest factors around 11 dB—signs of raw captures with full-scale peaks and room noise.

Your production renders, despite identical technical format (mono 22.05 kHz/16-bit), are mastered differently: peaks at −6 to −2.8 dBFS, RMS near −22.1 dBFS, crest factors ~7 dB, and spectral centroids ~2.0 kHz.

**Result**: Productions sound darker, quieter, and lack the brightness and energy of the training material, creating a mismatch between what the model learned and what it produces.

---

## Solution Overview

We've created a **4-step pipeline** to fix this:

1. **Remaster Training Data** - Remove clipped peaks, normalize levels, control dynamics
2. **Improve Segmentation** - Create consistent, clean training segments
3. **Post-Process Productions** - Apply loudness normalization, spectral matching, brightness enhancement
4. **Automated QC** - Validate all audio meets quality standards

---

## Step 1: Remaster Training Data

### Purpose
Remove 0 dBFS clipping, normalize loudness, reduce excessive brightness, and create clean training data with proper headroom.

### Script
```bash
python remaster_training_data.py
```

### What It Does

1. **Removes Clipped Peaks**
   - Detects samples at/near 0 dBFS (≥0.99)
   - Reconstructs clipped regions using cubic interpolation
   - Prevents model from learning clipping as a feature

2. **Applies De-Essing**
   - Reduces excessive sibilance in 4-10 kHz range
   - Gentle 2.5 dB reduction (dynamic threshold)
   - Smooths harsh high frequencies

3. **Normalizes to Target RMS**
   - Sets all segments to −21 dBFS RMS (matches original training level)
   - Ensures consistent loudness across segments
   - Model learns consistent speech level

4. **Controls Crest Factor**
   - Target: 8 dB crest (realistic for natural speech)
   - Applies gentle 2:1 compression above threshold
   - Soft knee, smooth release (50ms)
   - Preserves dynamics while controlling peaks

5. **True-Peak Limiting**
   - Final limiter at −3 dBFS (leaves headroom)
   - Look-ahead processing (1ms)
   - Smooth release (50ms)
   - No clipping in output

### Output
- `training_data/segments_remastered/` - Cleaned segments ready for training
- `training_data/remastered_*.wav` - Individual remastered audio files

### Expected Results
```
Original:  RMS=-21.0dB, Peak=0.0dB,   Crest=11.0dB
Remastered: RMS=-21.0dB, Peak=-3.0dB, Crest=8.0dB
```

---

## Step 2: Improve Segmentation

### Purpose
Create high-quality training segments with consistent loudness, proper trimming, and no duplicates.

### Script
```bash
python improved_segmentation.py
```

### What It Does

1. **Smart Silence Detection**
   - Configurable threshold (default 35 dB)
   - Trims silence from segment boundaries
   - Preserves natural speech timing

2. **Duration Management**
   - Min: 2 seconds, Max: 10 seconds
   - Target: 5 seconds (optimal for embeddings)
   - Splits long segments with overlap

3. **Quality Filtering**
   - RMS > −40 dBFS (not too quiet)
   - Peak < −0.5 dBFS (no clipping)
   - Spectral centroid: 800-4000 Hz (natural voice range)
   - Pitch detection (ensures voice content)
   - Flatness < 0.5 (not pure noise)

4. **DC Offset Removal**
   - Removes any DC bias
   - Centers waveform around zero

5. **Consistent Normalization**
   - Normalizes each segment to −21 dBFS RMS
   - Soft limiting at 0.95 peak
   - Prevents any clipping

6. **Deduplication**
   - Uses MFCC similarity (13 coefficients)
   - Removes segments with >95% similarity
   - Ensures training diversity

7. **Diversity Analysis**
   - Reports duration statistics
   - Spectral centroid distribution
   - Pitch range coverage

### Output
- `training_data/segments_improved/` - Optimized training segments

### Configuration
Edit `SegmentConfig` in the script to adjust:
```python
SegmentConfig(
    min_duration=2.0,          # Minimum segment length
    max_duration=10.0,         # Maximum segment length
    target_duration=5.0,       # Ideal segment length
    silence_threshold_db=35,   # Silence detection threshold
    target_rms_db=-21.0,      # Target loudness
    sample_rate=22050         # Sample rate
)
```

---

## Step 3: Post-Process Production Outputs

### Purpose
Apply professional mastering to generated voice samples: loudness normalization, spectral matching, and brightness enhancement.

### Script
```bash
python production_post_process.py
```

### What It Does

1. **LUFS Loudness Normalization**
   - Targets −16 LUFS (modern streaming/broadcast standard)
   - Uses ITU-R BS.1770 approximation (K-weighting)
   - Absolute gate (−70 LUFS) and relative gate (−10 dB)
   - Consistent loudness for delivery

2. **Spectral Matching**
   - Calculates 32-band mel-scaled spectral envelope
   - Compares production to training reference
   - Applies correction EQ (70% strength by default)
   - Transfers tonal characteristics while staying clean
   - Uses mel filter bank for perceptual accuracy

3. **Targeted Brightness Enhancement**
   - Measures spectral centroid
   - Boosts if below target (default 2800 Hz)
   - High-shelf filter at 2000 Hz
   - Caps boost at +6 dB
   - Maintains natural sound

4. **True-Peak Limiting**
   - Final limiter at −1 dBFS
   - 1ms look-ahead
   - 50ms smooth release
   - Prevents inter-sample peaks
   - Safe for codec processing

### Output
- `production_clone_output_processed/` - Mastered production files

### Configuration
```python
ProductionPostProcessor(
    target_lufs=-16.0,        # Modern delivery standard
    target_peak_db=-1.0,      # True-peak headroom
    sr=22050
)

# In process_directory call:
spectral_match_strength=0.7    # 70% spectral matching
target_centroid_hz=2800.0      # Target brightness
```

### Usage Examples

#### Process Production Directory
```bash
python production_post_process.py
```

#### Process Single File
```python
processor = ProductionPostProcessor()
processor.process_file(
    Path("input.wav"),
    Path("output.wav"),
    reference_path=Path("training_reference.wav"),
    spectral_match_strength=0.7,
    target_centroid_hz=2800.0
)
```

---

## Step 4: Automated Quality Control

### Purpose
Validate audio quality metrics and catch outliers before release.

### Script
```bash
python audio_quality_checker.py
```

### What It Does

1. **Measures Key Metrics**
   - RMS level (dBFS)
   - Peak level (dBFS)
   - Crest factor (dB)
   - Spectral centroid (Hz) - brightness
   - Spectral bandwidth (Hz)
   - Spectral rolloff (Hz)
   - Spectral flatness
   - Zero-crossing rate
   - Noise floor (dBFS)
   - DC offset detection

2. **Quality Validation**
   - Checks against configurable thresholds
   - Flags: clipping, too quiet/loud, over-compressed, too dynamic
   - Flags: too dark/bright, noisy, DC offset
   - Pass/fail status per file

3. **Reporting**
   - Per-file status (PASS or list of issues)
   - Summary statistics (average metrics)
   - Issue breakdown (counts per issue type)
   - List of problematic files

4. **Comparison Mode**
   - Compare two directories side-by-side
   - Shows metric differences
   - Validates improvements

### Usage Examples

#### Analyze Single Directory
```bash
python audio_quality_checker.py training_data/segments_remastered
```

#### Compare Two Directories
```bash
python audio_quality_checker.py --compare training_data/segments_final_merged training_data/segments_remastered
```

#### Compare Training vs Production
```bash
python audio_quality_checker.py --compare training_data/segments_remastered production_clone_output_processed
```

#### Export Metrics to JSON
```bash
python audio_quality_checker.py production_clone_output --export metrics.json
```

#### Run Default Analysis (All Common Directories)
```bash
python audio_quality_checker.py
```

### Quality Thresholds

Default thresholds (edit `QualityThresholds` in script):
```python
QualityThresholds(
    # Loudness
    min_rms_db=-30.0,
    max_rms_db=-6.0,
    target_rms_db=-21.0,

    # Peak levels
    max_peak_db=-0.5,         # Must leave headroom
    clip_threshold=-0.1,      # Flag near-clipping

    # Dynamic range
    min_crest_db=4.0,         # Too compressed
    max_crest_db=15.0,        # Too dynamic/raw
    target_crest_db=8.0,

    # Spectral
    min_centroid_hz=800.0,    # Too dark
    max_centroid_hz=4500.0,   # Too bright
    target_centroid_hz=2500.0,

    # Noise
    max_zcr_rate=0.2,
    max_spectral_flatness=0.5
)
```

---

## Complete Workflow

### Initial Setup (One Time)

1. **Remaster Existing Training Data**
   ```bash
   python remaster_training_data.py
   ```
   - Input: `training_data/segments_final_merged/`
   - Output: `training_data/segments_remastered/`

2. **Validate Remastered Data**
   ```bash
   python audio_quality_checker.py --compare training_data/segments_final_merged training_data/segments_remastered
   ```
   - Confirm improvements in all metrics

3. **Update Training Configuration**

   Edit voice cloning scripts ([optimal_voice_clone.py](optimal_voice_clone.py:65), [production_voice_clone.py](production_voice_clone.py:71), etc.) to use remastered data:

   ```python
   # Change this line:
   data_path = Path("training_data/segments_final_merged")

   # To this:
   data_path = Path("training_data/segments_remastered")
   ```

---

### Adding New Training Data

1. **Place New Audio/Video Files**
   ```
   training_data/
   ├── new_video_1.mp4
   ├── new_video_2.mp4
   └── new_audio.wav
   ```

2. **Create Clean Segments**
   ```bash
   python improved_segmentation.py
   ```
   - Automatically finds all audio/video in `training_data/`
   - Excludes already-processed files
   - Output: `training_data/segments_improved/`

3. **Validate New Segments**
   ```bash
   python audio_quality_checker.py training_data/segments_improved
   ```
   - Review quality report
   - Check diversity metrics

4. **Merge with Existing Data**
   ```bash
   # On Windows:
   xcopy /Y training_data\segments_improved\*.wav training_data\segments_remastered\

   # On Linux/Mac:
   cp training_data/segments_improved/*.wav training_data/segments_remastered/
   ```

5. **Re-analyze Combined Dataset**
   ```bash
   python audio_quality_checker.py training_data/segments_remastered
   ```

---

### Production Generation Workflow

1. **Generate Voice Samples**
   ```bash
   python optimal_voice_clone.py
   # or
   python production_voice_clone.py
   ```
   - Uses remastered training data
   - Output: `production_clone_output/` or `optimal_clone_output/`

2. **Post-Process Generated Audio**
   ```bash
   python production_post_process.py
   ```
   - Applies loudness normalization
   - Spectral matching to training data
   - Brightness enhancement
   - True-peak limiting
   - Output: `production_clone_output_processed/`

3. **Quality Control Check**
   ```bash
   python audio_quality_checker.py production_clone_output_processed
   ```
   - Validates all metrics
   - Flags any outliers

4. **Compare Training vs Production**
   ```bash
   python audio_quality_checker.py --compare training_data/segments_remastered production_clone_output_processed
   ```
   - Verify metrics are closely matched
   - Target: <10% difference in brightness, RMS, crest

5. **Release Approved Files**
   - Only release files that pass QC
   - Archive metrics for future reference

---

## Integration with Existing Scripts

### Update Voice Cloning Scripts

#### Method 1: Change Data Path (Recommended)

Edit these files:
- [optimal_voice_clone.py](optimal_voice_clone.py:65)
- [production_voice_clone.py](production_voice_clone.py:71)
- [natural_voice_clone.py](natural_voice_clone.py:65)
- [ultra_bright_clone.py](ultra_bright_clone.py:50)

Change:
```python
data_path = Path("training_data/segments_final_merged")
```
To:
```python
data_path = Path("training_data/segments_remastered")
```

#### Method 2: Add Post-Processing to Clone Scripts

Add to the end of cloning functions:
```python
from production_post_process import ProductionPostProcessor

# After generating audio...
processor = ProductionPostProcessor(target_lufs=-16.0, target_peak_db=-1.0)

# Load reference
reference_audio, _ = librosa.load(
    "training_data/segments_remastered/segment_0000.wav",
    sr=22050, mono=True
)

# Post-process
audio_processed, metrics = processor.process_audio(
    generated_audio,
    reference_audio=reference_audio,
    spectral_match_strength=0.7,
    target_centroid_hz=2800.0
)

# Save processed audio instead of raw
sf.write(output_path, audio_processed, 22050, subtype='PCM_16')
```

---

## Expected Improvements

### Training Data
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| RMS | −21.0 dBFS | −21.0 dBFS | −21.0 dBFS ✓ |
| Peak | **0.0 dBFS** | **−3.0 dBFS** | −3.0 dBFS ✓ |
| Crest | **11.0 dB** | **8.0 dB** | 8.0 dB ✓ |
| Brightness | ~3000 Hz | ~2800 Hz | 2500-3000 Hz ✓ |
| ZCR | ~0.15 | ~0.10 | <0.2 ✓ |

### Production Output
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| RMS | −22.1 dBFS | **−16.0 LUFS** | −16.0 LUFS ✓ |
| Peak | −6.0 to −2.8 dBFS | **−1.0 dBFS** | −1.0 dBFS ✓ |
| Crest | ~7.0 dB | **8.0 dB** | 8.0 dB ✓ |
| Brightness | ~2000 Hz | **~2800 Hz** | 2500-3000 Hz ✓ |

### Alignment
- **Training ↔ Production Brightness**: Within 10% (2800 Hz ± 280 Hz)
- **Consistent Dynamics**: Crest factor matched within 1 dB
- **Professional Loudness**: Productions meet modern delivery standards
- **Clean Audio**: No clipping, controlled noise floor

---

## Automated CI/CD Integration

### Pre-Commit Hook (Quality Gate)
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check any modified WAV files
for file in $(git diff --cached --name-only | grep '\.wav$'); do
    python audio_quality_checker.py "$file" --export /tmp/qc_report.json

    if [ $? -ne 0 ]; then
        echo "ERROR: Audio quality check failed for $file"
        exit 1
    fi
done
```

### Makefile Example
```makefile
.PHONY: remaster segment postprocess qc all

remaster:
	python remaster_training_data.py

segment:
	python improved_segmentation.py

postprocess:
	python production_post_process.py

qc:
	python audio_quality_checker.py

all: remaster segment postprocess qc
	@echo "Complete audio pipeline executed successfully"
```

---

## Troubleshooting

### Issue: Training data still has clipped peaks after remastering

**Solution**: Lower the `target_peak_db` in [remaster_training_data.py](remaster_training_data.py:52):
```python
TrainingDataRemaster(
    target_peak_db=-6.0,  # More aggressive headroom
    ...
)
```

### Issue: Productions still too dark after post-processing

**Solution**: Increase spectral matching strength or brightness target:
```python
processor.process_directory(
    ...,
    spectral_match_strength=0.9,   # Increase from 0.7
    target_centroid_hz=3200.0      # Increase from 2800
)
```

### Issue: Too many segments rejected during segmentation

**Solution**: Relax quality thresholds in [improved_segmentation.py](improved_segmentation.py:24):
```python
SegmentConfig(
    silence_threshold_db=30,       # Lower (more permissive)
    min_rms_db=-45.0,             # Lower (accept quieter)
    min_spectral_centroid=600.0    # Lower (accept darker)
)
```

### Issue: QC flags too many false positives

**Solution**: Adjust thresholds in [audio_quality_checker.py](audio_quality_checker.py:22):
```python
QualityThresholds(
    max_crest_db=18.0,      # Increase if raw training data
    max_centroid_hz=5000.0,  # Increase for bright voices
    ...
)
```

---

## Performance Tips

### Batch Processing
Process multiple files in parallel:
```python
from multiprocessing import Pool

def process_wrapper(args):
    input_file, output_file = args
    # Processing logic here

if __name__ == "__main__":
    tasks = [(in_file, out_file) for in_file, out_file in file_pairs]

    with Pool(processes=4) as pool:
        pool.map(process_wrapper, tasks)
```

### Memory Optimization
For large datasets, process in chunks:
```python
def process_large_audio(audio_file, chunk_duration=30.0):
    # Load in chunks
    for chunk in librosa.stream(audio_file,
                                block_length=1024,
                                frame_length=2048,
                                hop_length=512):
        # Process chunk
        yield process_chunk(chunk)
```

---

## Maintenance

### Regular Quality Audits
```bash
# Weekly: Check all production outputs
python audio_quality_checker.py production_clone_output_processed --export weekly_report_$(date +%Y%m%d).json

# Monthly: Re-analyze training data diversity
python improved_segmentation.py  # Just for analysis
```

### Threshold Tuning
Periodically review QC thresholds based on:
- User feedback on audio quality
- Distribution of metric values
- Changes in source material

### Dataset Expansion
- Aim for 20+ minutes of diverse training data
- Include varied speaking styles, emotions, pacing
- Maintain balance of phoneme coverage

---

## Additional Resources

### Key Files
- [remaster_training_data.py](remaster_training_data.py) - Training data remastering
- [improved_segmentation.py](improved_segmentation.py) - Smart segmentation
- [production_post_process.py](production_post_process.py) - Production mastering
- [audio_quality_checker.py](audio_quality_checker.py) - Quality control

### Related Scripts
- [optimal_voice_clone.py](optimal_voice_clone.py) - Best quality cloning (73% brightness match)
- [production_voice_clone.py](production_voice_clone.py) - Balanced production cloning
- [improved_voice_clone.py](improved_voice_clone.py) - Base cloning infrastructure

### Technical References
- ITU-R BS.1770-4: Loudness measurement standard
- EBU R128: Loudness normalization for broadcast
- AES17: Digital audio measurement methods
- Librosa documentation: https://librosa.org/doc/latest/

---

## Summary

This pipeline solves the training-production mismatch by:

1. ✅ **Removing clipping** from training data (0 dBFS → −3 dBFS peaks)
2. ✅ **Normalizing dynamics** (crest factor 11 → 8 dB)
3. ✅ **Consistent segmentation** with quality filtering and deduplication
4. ✅ **Professional production mastering** (−16 LUFS, spectral matching)
5. ✅ **Automated quality control** to catch outliers

**Result**: Training data and production outputs are aligned in loudness, brightness, and dynamics, creating consistent, professional-quality voice generation.

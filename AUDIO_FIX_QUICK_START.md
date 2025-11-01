# Audio Quality Fix - Quick Start Guide

## Problem
Training audio has **0 dBFS clipping** and **high brightness (~3 kHz)**. Production outputs are **darker (~2 kHz)** and **quieter**, creating a mismatch.

## Solution
4-step pipeline to align training and production audio characteristics.

---

## Installation

No additional dependencies needed beyond existing requirements:
```bash
pip install librosa soundfile numpy scipy
```

---

## Quick Start (Complete Pipeline)

Run everything in one command:
```bash
python run_audio_pipeline.py --all
```

This will:
1. ✅ Remaster training data (remove clipping, normalize)
2. ✅ Improve segmentation (clean, dedupe, consistent loudness)
3. ✅ Post-process production outputs (LUFS, spectral matching)
4. ✅ Run quality control (validate metrics)

---

## Step-by-Step Usage

### Step 1: Fix Training Data (Remove Clipping)
```bash
python remaster_training_data.py
```
- **Input**: `training_data/segments_final_merged/`
- **Output**: `training_data/segments_remastered/`
- **Fixes**: Clipping (0→−3dB peak), crest factor (11→8dB), brightness control

### Step 2: Create Clean Segments
```bash
python improved_segmentation.py
```
- **Input**: `training_data/*.wav`, `training_data/*.mp4`
- **Output**: `training_data/segments_improved/`
- **Features**: Consistent RMS, silence trimming, deduplication, quality filtering

### Step 3: Master Production Audio
```bash
python production_post_process.py
```
- **Input**: `production_clone_output/`
- **Output**: `production_clone_output_processed/`
- **Applies**: −16 LUFS normalization, spectral matching, brightness enhancement, true-peak limiting

### Step 4: Quality Control
```bash
# Check single directory
python audio_quality_checker.py training_data/segments_remastered

# Compare two directories
python audio_quality_checker.py --compare training_data/segments_final_merged training_data/segments_remastered

# Compare training vs production
python audio_quality_checker.py --compare training_data/segments_remastered production_clone_output_processed
```

---

## Update Voice Cloning Scripts

Edit [optimal_voice_clone.py](optimal_voice_clone.py:65), [production_voice_clone.py](production_voice_clone.py:71), etc.:

**Change this:**
```python
data_path = Path("training_data/segments_final_merged")
```

**To this:**
```python
data_path = Path("training_data/segments_remastered")
```

Then regenerate voice samples:
```bash
python optimal_voice_clone.py
```

---

## Expected Results

### Before
| Audio Type | RMS | Peak | Crest | Brightness |
|------------|-----|------|-------|------------|
| Training | −21dB | **0dB** | **11dB** | **3000Hz** |
| Production | −22dB | −6dB | 7dB | **2000Hz** |
| **Mismatch** | ✓ OK | **CLIPPING** | **Too dynamic** | **33% darker** |

### After
| Audio Type | RMS | Peak | Crest | Brightness |
|------------|-----|------|-------|------------|
| Training | −21dB | **−3dB** | **8dB** | **2800Hz** |
| Production | **−16 LUFS** | **−1dB** | **8dB** | **2800Hz** |
| **Match** | ✓ OK | ✓ Headroom | ✓ Aligned | ✓ Aligned |

---

## File Reference

### New Scripts (Solutions)
- **[remaster_training_data.py](remaster_training_data.py)** - Fix training audio clipping/dynamics
- **[improved_segmentation.py](improved_segmentation.py)** - Smart segmentation with quality control
- **[production_post_process.py](production_post_process.py)** - Master production outputs
- **[audio_quality_checker.py](audio_quality_checker.py)** - Automated quality validation
- **[run_audio_pipeline.py](run_audio_pipeline.py)** - Complete pipeline automation

### Existing Scripts (Update These)
- **[optimal_voice_clone.py](optimal_voice_clone.py)** - Best quality cloning (update data path)
- **[production_voice_clone.py](production_voice_clone.py)** - Balanced cloning (update data path)
- **[improved_voice_clone.py](improved_voice_clone.py)** - Base cloning module

### Documentation
- **[AUDIO_FIX_WORKFLOW.md](AUDIO_FIX_WORKFLOW.md)** - Complete detailed workflow guide
- **[AUDIO_FIX_QUICK_START.md](AUDIO_FIX_QUICK_START.md)** - This file

---

## Workflow Summary

```
1. Run Pipeline
   └─> python run_audio_pipeline.py --all

2. Update Cloning Scripts
   └─> Change data_path to "segments_remastered"

3. Generate New Voices
   └─> python optimal_voice_clone.py

4. Validate Quality
   └─> python audio_quality_checker.py production_clone_output_processed

5. Compare Results
   └─> python audio_quality_checker.py --compare <before> <after>
```

---

## Troubleshooting

### "No files found in training_data/"
- Place audio/video files in `training_data/` directory
- Run `python improved_segmentation.py` to create segments

### "Production directory not found"
- Generate voice samples first: `python optimal_voice_clone.py`
- Then run post-processing: `python production_post_process.py`

### "Quality check failed: too bright/dark"
- Adjust thresholds in `audio_quality_checker.py` (line 22, `QualityThresholds`)
- Or adjust target values in remastering/post-processing scripts

### "ImportError: No module named..."
- Install dependencies: `pip install librosa soundfile numpy scipy`

---

## Key Metrics Targets

| Metric | Training Target | Production Target | Tolerance |
|--------|----------------|-------------------|-----------|
| **RMS** | −21 dBFS | −16 LUFS | ±3 dB |
| **Peak** | −3 dBFS | −1 dBFS | ±2 dB |
| **Crest** | 8 dB | 8 dB | ±2 dB |
| **Brightness** | 2800 Hz | 2800 Hz | ±300 Hz |

---

## Command Cheat Sheet

```bash
# Complete pipeline (recommended first run)
python run_audio_pipeline.py --all

# Individual steps
python run_audio_pipeline.py --remaster      # Step 1
python run_audio_pipeline.py --segment       # Step 2
python run_audio_pipeline.py --postprocess   # Step 3
python run_audio_pipeline.py --qc            # Step 4

# Quality control
python audio_quality_checker.py <directory>
python audio_quality_checker.py --compare <dir1> <dir2>
python audio_quality_checker.py <dir> --export metrics.json

# Generate voices (after fixing training data)
python optimal_voice_clone.py           # Best quality
python production_voice_clone.py        # Balanced
```

---

## Next Steps

1. **Run the pipeline**:
   ```bash
   python run_audio_pipeline.py --all
   ```

2. **Review quality reports** in terminal output

3. **Update voice cloning scripts** to use remastered data

4. **Regenerate voice samples** and compare quality

5. **Read full documentation**: [AUDIO_FIX_WORKFLOW.md](AUDIO_FIX_WORKFLOW.md)

---

## Support

For detailed documentation, troubleshooting, and advanced configuration:
- See **[AUDIO_FIX_WORKFLOW.md](AUDIO_FIX_WORKFLOW.md)**

For technical details on each component:
- [remaster_training_data.py](remaster_training_data.py) - Line 1-400+
- [production_post_process.py](production_post_process.py) - Line 1-550+
- [improved_segmentation.py](improved_segmentation.py) - Line 1-550+
- [audio_quality_checker.py](audio_quality_checker.py) - Line 1-600+

---

**Ready to start?**
```bash
python run_audio_pipeline.py --all
```

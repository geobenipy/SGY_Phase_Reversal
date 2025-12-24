# Phase Rotation Detector for Seismic Interpretation

Professional tool for detecting phase-rotated reflectors in stacked marine seismic data. Identifies anomalous phase behavior relative to a reference horizon (e.g., seafloor) to support hydrocarbon indicator analysis and seismic interpretation.

## Overview

This tool analyzes the instantaneous phase of seismic reflectors and compares them to a reference horizon. Phase rotations can indicate:

- **Hydrocarbon presence** (polarity reversal effects)
- **Lithology changes** (impedance contrasts)
- **Fluid contacts** (gas-water, oil-water boundaries)
- **Processing artifacts** requiring attention

The output is a SEG-Y file containing the phase rotation attribute that can be loaded as an overlay in any seismic interpretation software (Petrel, Kingdom, OpendTect, etc.).

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- matplotlib
- scipy
- segyio
- tqdm
- seaborn

### 2. Setup

Place the following files in your working directory:
- `config.py` - Configuration file
- `run_phase_rotation_detector.py` - Main script
- `phase_rotation_detector.py` - Core processing module
- `phase_rotation_qc.py` - QC and visualization module

### 3. Configure

Edit `config.py` and set your paths:

```python
INPUT_DIR = Path(r"C:/your/seismic/data")
OUTPUT_DIR = Path(r"C:/your/output/folder")
```

Configure analysis parameters:
```python
REFERENCE_TYPE = 'seafloor'  # or 'manual', 'auto'
PHASE_ROTATION_THRESHOLD = 45.0  # degrees
NORMALIZE_OUTPUT = True
```

### 4. Run

```bash
python run_phase_rotation_detector.py
```

## How It Works

### 1. Reference Horizon Selection

The tool first establishes a reference phase from a selected horizon:

**Seafloor Mode** (default):
- Automatically detects seafloor using amplitude threshold
- Extracts phase from seafloor reflector
- Most reliable for marine data

**Manual Mode**:
- Uses user-specified time window
- Useful when specific horizon is known

**Auto Mode**:
- Detects strongest continuous reflector
- Good for land data or when seafloor unclear

### 2. Instantaneous Phase Analysis

For each sample in the seismic data:
- Computes instantaneous phase using Hilbert transform
- Compares to reference phase
- Calculates phase rotation (difference)

### 3. Phase Rotation Attribute

The output attribute shows:
- **Near 0°**: Normal phase, consistent with reference
- **±45-90°**: Moderate phase rotation, investigate
- **±90-180°**: Strong phase rotation, likely anomaly

### 4. Output Generation

Creates SEG-Y file with:
- Same geometry as input
- Original headers preserved
- Phase rotation as amplitude values
- Compatible with all interpretation software

## Configuration Options

### Reference Horizon Settings

```python
# Use seafloor as reference
REFERENCE_TYPE = 'seafloor'
SEAFLOOR_SEARCH_START = 0      # Start time (ms)
SEAFLOOR_SEARCH_END = 500      # End time (ms)
SEAFLOOR_THRESHOLD = 0.6       # Amplitude threshold

# Or use manual time window
REFERENCE_TYPE = 'manual'
MANUAL_REFERENCE_START = 500   # Window start (ms)
MANUAL_REFERENCE_END = 600     # Window end (ms)
```

### Phase Analysis Settings

```python
ANALYSIS_WINDOW = 32           # Window size for phase estimation
PHASE_ROTATION_THRESHOLD = 45  # Anomaly threshold (degrees)
SMOOTH_PHASES = True           # Reduce noise
SMOOTHING_WINDOW = 5           # Lateral smoothing (traces)
```

### Output Options

```python
OUTPUT_ATTRIBUTE = 'phase_rotation'  # Signed rotation
NORMALIZE_OUTPUT = True              # Scale to [-1, 1]
APPLY_GAIN = True                    # Enhance subtle features
GAIN_FACTOR = 2.0                    # Gain multiplier
```

### Advanced Settings

```python
# Frequency filtering
FREQUENCY_BAND = (10, 80)     # Bandpass 10-80 Hz, or None

# Time gating
TIME_GATE = (500, 2000)       # Analyze 500-2000ms only, or None

# Processing optimization
TRACE_DECIMATION = 1          # Use every Nth trace (1=all)
MAX_WORKERS = None            # Parallel workers (None=all CPUs)
```

## Output Files

### Directory Structure

```
Output/
├── line_001/
│   ├── line_001_phase_rotation.sgy    # Main output - load in interpreter
│   └── qc/
│       └── line_001_analysis.png      # Detailed QC plot
│
├── line_002/
│   ├── line_002_phase_rotation.sgy
│   └── qc/
│       └── line_002_analysis.png
│
├── phase_rotation_qc.png              # Comparison plot
├── phase_rotation_statistics.csv      # Numerical statistics
├── seafloor_picks.csv                 # Reference horizon picks
└── phase_rotation_detection.log       # Processing log
```

### SEG-Y Output Format

The phase rotation SEG-Y file:
- Has identical geometry to input
- Preserves all trace headers
- Amplitude values = phase rotation (degrees or normalized)
- Can be loaded directly in interpretation software

### Using in Interpretation Software

**Petrel**:
1. Import phase rotation SEG-Y as "Seismic Attribute"
2. Display as overlay with transparency
3. Use color bar: -180° (blue) to +180° (red)

**Kingdom**:
1. Load as additional seismic volume
2. Create composite display with base data
3. Threshold at ±45° to highlight anomalies

**OpendTect**:
1. Import as 3D volume
2. Use "Blend" display mode
3. Apply "Phase Rotation" color table

## Quality Control Plots

### Detailed Analysis Plot (per file)

6-panel plot showing:
1. **Original Seismic**: Input data with reference horizon
2. **Phase Rotation Attribute**: Detected phase anomalies
3. **Phase Histogram**: Distribution of rotation values
4. **Amplitude-Phase Crossplot**: Relationship analysis
5. **Vertical Profiles**: Sample traces with phase overlay
6. **Statistics Panel**: Numerical summary

### Comparison Plot

Side-by-side comparison of multiple lines for regional analysis.

## Interpretation Guidelines

### Normal Phase Behavior

- Phase consistent with reference (0 ± 20°)
- Parallel to stratigraphy
- Continuous across section

### Phase Rotation Anomalies

**Hydrocarbon Indicators**:
- Phase rotation 90-180°
- Localized to specific horizons
- Bright spot association
- Conformable to structure

**Lithology Changes**:
- Gradual phase changes 20-45°
- Follow stratigraphic boundaries
- Predictable patterns

**Processing Artifacts**:
- Random phase variations
- No geological correlation
- Associated with data quality issues

### Best Practices

1. **Calibrate with Wells**: Compare phase anomalies to well data
2. **Use Multiple Attributes**: Combine with amplitude, AVO, coherence
3. **Regional Context**: Compare adjacent lines for continuity
4. **Frequency Analysis**: Different frequencies see different depths
5. **QC Thoroughly**: Review all QC plots before interpretation

## Example Workflow

### Hydrocarbon Prospect Evaluation

```python
# config.py settings
REFERENCE_TYPE = 'seafloor'
PHASE_ROTATION_THRESHOLD = 45.0
FREQUENCY_BAND = (15, 60)  # Target reservoir frequency
APPLY_GAIN = True
GAIN_FACTOR = 2.0
```

Steps:
1. Process all seismic lines
2. Load phase rotation attributes in interpreter
3. Identify zones with >90° rotation
4. Correlate with amplitude anomalies
5. Map lateral extent
6. Integrate with well data

### Processing Artifact Detection

```python
# config.py settings
REFERENCE_TYPE = 'auto'
SMOOTH_PHASES = False  # Don't smooth to see artifacts
FREQUENCY_BAND = None  # Full bandwidth
```

Steps:
1. Process problem areas
2. Look for random phase variations
3. Correlate with acquisition geometry
4. Identify reprocessing needs

## Statistics Output

### CSV Format

```csv
filename,n_traces,n_samples,sample_rate,mean_phase,std_phase,min_phase,max_phase,anomaly_percentage,avg_seafloor_time
line_001.sgy,480,2000,2.0,2.34,18.67,-156.2,178.9,8.45,234.5
```

Fields:
- `mean_phase`: Average phase rotation
- `std_phase`: Phase variability
- `anomaly_percentage`: % samples exceeding threshold
- `avg_seafloor_time`: Reference horizon depth

### Seafloor Picks CSV

```csv
filename,trace,sample,time_ms
line_001.sgy,0,120,240.0
line_001.sgy,1,121,242.0
```

## Troubleshooting

### Issue: Noisy phase attribute

**Solution**: Increase smoothing
```python
SMOOTH_PHASES = True
SMOOTHING_WINDOW = 9  # Increase from 5
```

### Issue: Weak phase rotations not visible

**Solution**: Apply gain
```python
APPLY_GAIN = True
GAIN_FACTOR = 3.0  # Increase gain
```

### Issue: Seafloor detection fails

**Solution**: Adjust search parameters
```python
SEAFLOOR_SEARCH_END = 800  # Extend search window
SEAFLOOR_THRESHOLD = 0.4   # Lower threshold
```

Or switch to manual mode:
```python
REFERENCE_TYPE = 'manual'
MANUAL_REFERENCE_START = 200
MANUAL_REFERENCE_END = 300
```

### Issue: Processing too slow

**Solution**: Optimize settings
```python
TRACE_DECIMATION = 2  # Use every 2nd trace
CREATE_DETAILED_PLOTS = False  # Skip detailed plots
```

## Technical Background

### Instantaneous Phase

Computed using Hilbert transform:

```
analytic_signal = trace + i * hilbert(trace)
instantaneous_phase = atan2(imag(analytic), real(analytic))
```

### Phase Rotation

Phase difference wrapped to [-180°, +180°]:

```
rotation = (measured_phase - reference_phase)
rotation = atan2(sin(rotation), cos(rotation))  # Wrap
```

### Normalization

Phase rotation normalized to [-1, 1] for visualization:

```
normalized = rotation / max(|rotation|)
```

## Performance

Typical processing times:
- Small file (100 traces, 2000 samples): 2-5 seconds
- Medium file (500 traces, 3000 samples): 10-20 seconds
- Large file (1000 traces, 5000 samples): 30-60 seconds

Scales linearly with:
- Number of traces
- Number of samples
- Smoothing window size

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scipy >= 1.6.0
- segyio >= 1.9.0
- tqdm >= 4.60.0
- seaborn >= 0.11.0

## References

Theoretical background:
- Taner, M.T., Koehler, F., and Sheriff, R.E., 1979, Complex seismic trace analysis: Geophysics, 44, 1041-1063.
- Barnes, A.E., 2007, A tutorial on complex seismic trace analysis: Geophysics, 72, W33-W43.

## License

MIT License - Free for commercial and academic use

## Support

For issues or questions:
- Check the log file: `phase_rotation_detection.log`
- Review QC plots for data quality issues
- Verify configuration parameters
- Ensure input data is properly formatted SEG-Y
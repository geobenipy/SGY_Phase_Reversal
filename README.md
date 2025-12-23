# SEG-Y Phase Reversal Processor

Professional tool for processing stacked marine reflection seismic data with phase reversal, overlay plotting, and comprehensive QC.

## Quick Start

### 1. Installation

```bash
pip install numpy matplotlib scipy segyio tqdm seaborn
```

### 2. Setup

Download all Python files:
- `config.py` - Configuration file
- `run_processor.py` - Main script
- `segy_processor.py` - Core processing
- `segy_qc.py` - Quality control module

### 3. Configure

Edit `config.py` and set your paths:

```python
INPUT_DIR = Path(r"C:/your/input/folder")
OUTPUT_DIR = Path(r"C:/your/output/folder")
```

### 4. Run

```bash
python run_processor.py
```

That's it!

## What It Does

The processor will:
1. Read all .sgy/.segy files from INPUT_DIR
2. Apply 180° phase reversal
3. Normalize amplitudes
4. Create individual processed files
5. Create combined SEG-Y output
6. Generate overlay plot (PNG)
7. Create QC reports (optional)
8. Export metadata (CSV/JSON)

## Configuration Options

All settings in `config.py`:

### Basic Settings
```python
PHASE_REVERSAL = True          # Apply phase reversal
NORMALIZE_AMPLITUDES = True    # Normalize to [-1, 1]
MAX_WORKERS = None             # Number of CPUs (None = all)
```

### Output Control
```python
CREATE_INDIVIDUAL_SEGY = True  # Individual processed files
CREATE_COMBINED_SEGY = True    # Combined output file
CREATE_OVERLAY_PLOT = True     # PNG overlay plot
CREATE_QC_REPORTS = True       # Quality control reports
```

### Plot Settings
```python
PLOT_TYPE = 'wiggle'           # 'wiggle', 'density', 'both'
FIGURE_SIZE = (16, 10)         # Width, height in inches
DPI = 300                      # Resolution
```

## File Structure

```
your_project/
├── config.py              # Configuration
├── run_processor.py       # Main script
├── segy_processor.py      # Core module
├── segy_qc.py            # QC module
├── data/
│   ├── input/            # Your SEG-Y files here
│   └── output/           # Results will appear here
```

## Output Files

After running, you'll find in OUTPUT_DIR:

- `processed_*.sgy` - Individual processed files
- `combined_phase_reversed.sgy` - All data combined
- `overlay_plot.png` - Visual overlay of all files
- `qc_report_*.png` - QC reports (if enabled)
- `comparison_*.png` - Before/after plots (if enabled)
- `metadata.csv` - File metadata table
- `metadata.json` - Detailed metadata
- `segy_processing.log` - Processing log

## Example Output

```
============================================================
    SEG-Y Phase Reversal Processing
============================================================

Configuration:
  Input Directory:     C:/seismic_data/input
  Output Directory:    C:/seismic_data/output
  
Processing Options:
  Phase Reversal:      True
  Normalize:           True
  Max Workers:         All CPUs

============================================================

Processing SEG-Y files: 100%|████████| 12/12 [00:45<00:00]

============================================================
PROCESSING COMPLETE
============================================================
Total files processed:  12
Total duration:         45.23 seconds
Average per file:       3.77 seconds

Results saved to:       C:/seismic_data/output

Generated files:
  ✓ 12 individual processed SEG-Y files
  ✓ combined_phase_reversed.sgy
  ✓ overlay_plot.png
  ✓ 12 QC reports
  ✓ metadata.csv
  ✓ metadata.json
============================================================
```

## Advanced Usage

### Custom Processing

You can modify `run_processor.py` to add custom processing:

```python
# After line with processor.process_single_file():
data = custom_filter(data)  # Add your processing
data = custom_gain(data)    # Apply custom gain
```

### Selective Processing

Process only specific files by pattern in `config.py`:

```python
FILENAME_PATTERN = '*line_01*'  # Only files matching pattern
```

### Performance Tuning

For very large datasets:

```python
MAX_WORKERS = 4              # Limit parallel processes
CREATE_QC_REPORTS = False    # Skip QC to save time
```

## Troubleshooting

### Issue: Out of memory

**Solution:** Process fewer files at once or reduce workers:
```python
MAX_WORKERS = 2
```

### Issue: Files not found

**Solution:** Check your paths use raw strings:
```python
INPUT_DIR = Path(r"C:/path/to/files")  # Note the 'r'
```

### Issue: Slow processing

**Solution:** Increase workers and disable QC:
```python
MAX_WORKERS = None  # Use all CPUs
CREATE_QC_REPORTS = False
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scipy >= 1.6.0
- segyio >= 1.9.0
- tqdm >= 4.60.0
- seaborn >= 0.11.0

## How Phase Reversal Works

Phase reversal multiplies all amplitude values by -1, creating a 180° phase shift. This is useful for:

- Comparing different processing sequences
- Polarity analysis
- Identifying phase-dependent features
- Quality control of acquisition geometry

The mathematical operation is simple:
```
reversed_data = -original_data
```

But the implications for seismic interpretation can be significant.

## Support

For issues or questions, check:
- The log file in OUTPUT_DIR: `segy_processing.log`
- Console output for error messages
- Make sure all paths exist and are accessible

## License

MIT License - Free to use and modify
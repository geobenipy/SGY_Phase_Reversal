"""
config.py - Configuration File
===============================
Adjust all settings here before running the processor.
"""

from pathlib import Path

# =============================================================================
# PATHS - Set your input and output directories here
# =============================================================================

INPUT_DIR = Path(r"D:\Haimerl\PhD\Vista\AL641\Export\ReProc\Finished")  # Your SEG-Y input folder
OUTPUT_DIR = Path(r"E:\Miscellaneous\Skripte&Tutorials\Phase_Reversal\Output")  # Output folder for results

# Alternative: Use relative paths
# INPUT_DIR = Path("./data/input")
# OUTPUT_DIR = Path("./data/output")


# =============================================================================
# PROCESSING SETTINGS
# =============================================================================

# Phase reversal (180° phase shift)
PHASE_REVERSAL = True

# Normalize amplitudes to [-1, 1] range
NORMALIZE_AMPLITUDES = True

# Number of parallel workers (None = use all CPU cores)
MAX_WORKERS = 1  # or set to specific number like 4, 8, etc.


# =============================================================================
# OUTPUT OPTIONS
# =============================================================================

# Create individual processed SEG-Y files
CREATE_INDIVIDUAL_SEGY = True

# Create combined SEG-Y file from all inputs
CREATE_COMBINED_SEGY = True

# Create overlay plot (PNG)
CREATE_OVERLAY_PLOT = True

# Create QC reports for each file
CREATE_QC_REPORTS = True

# Create before/after comparison plots
CREATE_COMPARISON_PLOTS = True


# =============================================================================
# PLOTTING SETTINGS
# =============================================================================

# Plot type: 'wiggle', 'density', or 'both'
PLOT_TYPE = 'both'

# Figure size in inches (width, height)
FIGURE_SIZE = (16, 10)

# Resolution in DPI
DPI = 300

# Colormap for seismic data
COLORMAP = 'seismic'


# =============================================================================
# FILE FILTERS
# =============================================================================

# File extensions to process (case-insensitive)
FILE_EXTENSIONS = ['.sgy', '.segy']

# Optional: Filename pattern filter (None = process all)
# Example: FILENAME_PATTERN = '*line*' to only process files with 'line' in name
FILENAME_PATTERN = None


# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Log to file
LOG_TO_FILE = True

# Log to console
LOG_TO_CONSOLE = True

# Log filename
LOG_FILENAME = r"E:\Miscellaneous\Skripte&Tutorials\Phase_Reversal\Output\segy_processing.log"


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Clip percentile for plotting (to avoid extreme values)
PLOT_CLIP_PERCENTILE = 99

# Frequency range for spectral analysis (Hz)
FREQUENCY_RANGE = (0, 250)

# Number of traces to use for QC analysis (None = all)
QC_TRACE_LIMIT = None

# Export metadata to CSV
EXPORT_METADATA_CSV = True

# Export metadata to JSON
EXPORT_METADATA_JSON = True


# =============================================================================
# OUTPUT FILENAMES (Optional customization)
# =============================================================================

OVERLAY_PLOT_NAME = r"E:\Miscellaneous\Skripte&Tutorials\Phase_Reversal\Output\overlay_plot.png"
COMBINED_SEGY_NAME = r"E:\Miscellaneous\Skripte&Tutorials\Phase_Reversal\Output\combined_phase_reversed.sgy"
METADATA_CSV_NAME = r"E:\Miscellaneous\Skripte&Tutorials\Phase_Reversal\Output\metadata.csv"
METADATA_JSON_NAME = r"E:\Miscellaneous\Skripte&Tutorials\Phase_Reversal\Output\metadata.json"
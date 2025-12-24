"""
config.py - Configuration File for Phase Rotation Detector
===========================================================
Configure all settings for detecting phase-rotated reflectors.
"""

from pathlib import Path

# =============================================================================
# PATHS - Set your input and output directories
# =============================================================================

INPUT_DIR = Path(r"D:\Haimerl\PhD\Vista\AL641\Export\ReProc\Finished")
OUTPUT_DIR = Path(r"E:/Miscellaneous/Skripte&Tutorials/Phase_Reversal/Output_complextrace")


# =============================================================================
# REFERENCE HORIZON SETTINGS
# =============================================================================

# Reference horizon type: 'seafloor', 'manual', or 'auto'
# - 'seafloor': Automatically detect seafloor as reference
# - 'manual': Use manually specified time window
# - 'auto': Automatically detect strongest continuous reflector
REFERENCE_TYPE = 'seafloor'

# Manual reference time window (ms) - only used if REFERENCE_TYPE = 'manual'
MANUAL_REFERENCE_START = 500
MANUAL_REFERENCE_END = 600

# Seafloor detection parameters
SEAFLOOR_SEARCH_START = 0      # Start time (ms) for seafloor search
SEAFLOOR_SEARCH_END = 500      # End time (ms) for seafloor search
SEAFLOOR_THRESHOLD = 0.6       # Amplitude threshold (0-1) for seafloor detection


# =============================================================================
# PHASE ANALYSIS SETTINGS
# =============================================================================

# Analysis window size (samples) for local phase estimation
ANALYSIS_WINDOW = 32

# Threshold for phase rotation detection (degrees)
# Values above this threshold indicate significant phase rotation
PHASE_ROTATION_THRESHOLD = 45.0

# Instantaneous phase computation method: 'hilbert' or 'complex_trace'
PHASE_METHOD = 'complex_trace'

# Smooth phase estimates over traces (reduces noise)
SMOOTH_PHASES = True
SMOOTHING_WINDOW = 5  # Number of traces for smoothing


# =============================================================================
# ATTRIBUTE COMPUTATION
# =============================================================================

# Output attribute type: 'indicator', 'full_phase', or 'both'
# - 'indicator': Sparse attribute showing phase rotation ONLY at reflectors (RECOMMENDED)
# - 'full_phase': Dense attribute showing phase everywhere
# - 'both': Generate both types of output
OUTPUT_ATTRIBUTE = 'both'

# Reflector detection threshold (percentile of envelope)
# Higher = only strongest reflectors, Lower = more reflectors detected
REFLECTOR_THRESHOLD_PERCENTILE = 70.0

# Taper width around reflectors (samples)
# Expands the indicator slightly around detected reflectors
INDICATOR_TAPER_WIDTH = 3

# Normalize output to [-1, 1] range for better visualization
NORMALIZE_OUTPUT = True

# Apply gain to enhance subtle phase rotations
APPLY_GAIN = True
GAIN_FACTOR = 2.0


# =============================================================================
# PROCESSING OPTIONS
# =============================================================================

# Number of parallel workers (None = use all CPU cores)
MAX_WORKERS = None

# Create QC plots
CREATE_QC_PLOTS = True

# Create detailed analysis plots per file
CREATE_DETAILED_PLOTS = True

# Export statistics to CSV
EXPORT_STATISTICS = True


# =============================================================================
# OUTPUT OPTIONS
# =============================================================================

# Create individual SEG-Y output for each input file
CREATE_INDIVIDUAL_OUTPUTS = True

# SEG-Y output format
# Use same format as input (preserves all headers)
PRESERVE_INPUT_HEADERS = True

# Output filename suffix
OUTPUT_SUFFIX = '_phase_rotation'


# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Colormap for phase rotation visualization
# Recommended: 'seismic', 'RdBu', 'twilight', 'hsv'
PHASE_COLORMAP = 'RdBu'

# Figure size for plots (width, height in inches)
FIGURE_SIZE = (16, 10)

# Plot resolution (DPI)
DPI = 300

# Show seafloor pick on plots
SHOW_SEAFLOOR_PICK = True


# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Minimum coherency threshold for valid phase estimates
# Low coherency indicates noise - phase estimates will be attenuated
MIN_COHERENCY = 0.3

# Frequency band for analysis (Hz) - None for full bandwidth
# Example: (10, 80) for 10-80 Hz bandpass
FREQUENCY_BAND = None

# Time gate for analysis (ms) - None for full time range
# Example: (500, 2000) to analyze only between 500-2000 ms
TIME_GATE = None

# Trace decimation for faster processing (1 = no decimation)
# Example: 2 = use every 2nd trace, 5 = use every 5th trace
TRACE_DECIMATION = 1


# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'WARNING'

# Log to file
LOG_TO_FILE = True

# Log to console
LOG_TO_CONSOLE = True

# Log filename
LOG_FILENAME = 'phase_rotation_detection.log'


# =============================================================================
# FILE FILTERS
# =============================================================================

# File extensions to process
FILE_EXTENSIONS = ['.sgy', '.segy']

# Filename pattern filter (None = process all)
FILENAME_PATTERN = None


# =============================================================================
# OUTPUT FILENAMES
# =============================================================================

QC_PLOT_NAME = 'phase_rotation_qc.png'
STATISTICS_CSV_NAME = 'phase_rotation_statistics.csv'
SEAFLOOR_PICKS_NAME = 'seafloor_picks.csv'
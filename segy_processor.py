"""
phase_rotation_detector.py - Core Phase Rotation Detection Module
==================================================================

Detects phase-rotated reflectors relative to a reference horizon
for hydrocarbon indicator analysis and seismic interpretation.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import uniform_filter1d
import segyio
import warnings

warnings.filterwarnings('ignore')


class PhaseRotationDetector:
    """
    Detects phase rotations in seismic data relative to reference horizon.
    
    Analyzes instantaneous phase to identify reflectors with anomalous
    phase behavior, useful for hydrocarbon detection and interpretation.
    """
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        log_level: int = logging.INFO
    ):
        """
        Initialize phase rotation detector.
        
        Args:
            input_dir: Directory containing input SEG-Y files
            output_dir: Directory for output files
            log_level: Logging level
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def read_segy_file(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """
        Read SEG-Y file and extract seismic data.
        
        Args:
            filepath: Path to SEG-Y file
            
        Returns:
            Tuple of (seismic_data, metadata)
        """
        try:
            with segyio.open(str(filepath), 'r', ignore_geometry=True) as segy:
                # Read all traces
                data = segyio.collect(segy.trace[:])
                
                # Extract metadata
                metadata = {
                    'sample_rate': segyio.tools.dt(segy) / 1000.0,
                    'n_traces': segy.tracecount,
                    'n_samples': segy.samples.size,
                    'samples': np.array(segy.samples),
                    'text_header': segy.text[0],
                    'binary_header': segy.bin
                }
                
                self.logger.info(
                    f"Read {filepath.name}: {metadata['n_traces']} traces, "
                    f"{metadata['n_samples']} samples, dt={metadata['sample_rate']}ms"
                )
                
                return data, metadata
                
        except Exception as e:
            self.logger.error(f"Failed to read {filepath}: {str(e)}")
            raise IOError(f"Cannot read SEG-Y file: {filepath}") from e
    
    def detect_seafloor(
        self,
        data: np.ndarray,
        sample_rate: float,
        search_start: float,
        search_end: float,
        threshold: float = 0.6
    ) -> np.ndarray:
        """
        Detect seafloor horizon using amplitude threshold method.
        
        Args:
            data: Seismic data (n_traces x n_samples)
            sample_rate: Sample rate in milliseconds
            search_start: Start time for search (ms)
            search_end: End time for search (ms)
            threshold: Amplitude threshold (0-1)
            
        Returns:
            Array of seafloor times for each trace (samples)
        """
        n_traces, n_samples = data.shape
        seafloor_picks = np.zeros(n_traces, dtype=int)
        
        # Convert times to samples
        start_sample = int(search_start / sample_rate)
        end_sample = min(int(search_end / sample_rate), n_samples)
        
        self.logger.info(
            f"Detecting seafloor between {search_start}-{search_end}ms "
            f"(samples {start_sample}-{end_sample})"
        )
        
        for trace_idx in range(n_traces):
            trace = data[trace_idx, start_sample:end_sample]
            
            # Find first sample exceeding threshold
            abs_trace = np.abs(trace)
            max_amp = abs_trace.max()
            
            if max_amp > 0:
                threshold_amp = max_amp * threshold
                candidates = np.where(abs_trace > threshold_amp)[0]
                
                if len(candidates) > 0:
                    seafloor_picks[trace_idx] = start_sample + candidates[0]
                else:
                    seafloor_picks[trace_idx] = start_sample
            else:
                seafloor_picks[trace_idx] = start_sample
        
        # Smooth picks to remove outliers
        seafloor_picks = self._smooth_horizon(seafloor_picks, window=11)
        
        avg_time = np.mean(seafloor_picks) * sample_rate
        self.logger.info(f"Average seafloor time: {avg_time:.1f}ms")
        
        return seafloor_picks
    
    def _smooth_horizon(self, horizon: np.ndarray, window: int = 11) -> np.ndarray:
        """Smooth horizon picks using median filter."""
        from scipy.ndimage import median_filter
        return median_filter(horizon.astype(float), size=window).astype(int)
    
    def extract_reference_phase(
        self,
        data: np.ndarray,
        horizon_samples: np.ndarray,
        window: int = 16
    ) -> np.ndarray:
        """
        Extract reference phase from horizon.
        
        Args:
            data: Seismic data
            horizon_samples: Horizon pick in samples for each trace
            window: Window size around horizon
            
        Returns:
            Reference phase for each trace (radians)
        """
        n_traces = data.shape[0]
        reference_phases = np.zeros(n_traces)
        
        for trace_idx in range(n_traces):
            pick = horizon_samples[trace_idx]
            start = max(0, pick - window // 2)
            end = min(data.shape[1], pick + window // 2)
            
            # Extract window around pick
            window_data = data[trace_idx, start:end]
            
            # Compute instantaneous phase
            analytic = hilbert(window_data)
            inst_phase = np.angle(analytic)
            
            # Use phase at peak amplitude
            peak_idx = np.argmax(np.abs(window_data))
            reference_phases[trace_idx] = inst_phase[peak_idx]
        
        return reference_phases
    
    def compute_instantaneous_phase(self, data: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phase using Hilbert transform.
        
        Args:
            data: Seismic data (n_traces x n_samples)
            
        Returns:
            Instantaneous phase in radians (-π to π)
        """
        self.logger.debug("Computing instantaneous phase")
        
        n_traces, n_samples = data.shape
        phase = np.zeros_like(data)
        
        # Process each trace
        for i in range(n_traces):
            analytic_signal = hilbert(data[i])
            phase[i] = np.angle(analytic_signal)
        
        return phase
    
    def compute_phase_rotation(
        self,
        inst_phase: np.ndarray,
        reference_phase: np.ndarray,
        smooth: bool = True,
        smooth_window: int = 5
    ) -> np.ndarray:
        """
        Compute phase rotation relative to reference.
        
        Args:
            inst_phase: Instantaneous phase (n_traces x n_samples)
            reference_phase: Reference phase per trace (n_traces)
            smooth: Apply smoothing to reduce noise
            smooth_window: Smoothing window size
            
        Returns:
            Phase rotation in degrees (-180 to 180)
        """
        n_traces, n_samples = inst_phase.shape
        phase_rotation = np.zeros_like(inst_phase)
        
        # Compute phase difference for each trace
        for trace_idx in range(n_traces):
            ref_phase = reference_phase[trace_idx]
            phase_diff = inst_phase[trace_idx] - ref_phase
            
            # Wrap to [-π, π]
            phase_diff = np.angle(np.exp(1j * phase_diff))
            
            # Convert to degrees
            phase_rotation[trace_idx] = np.degrees(phase_diff)
        
        # Smooth laterally if requested
        if smooth and smooth_window > 1:
            self.logger.debug(f"Smoothing phase rotation (window={smooth_window})")
            for sample_idx in range(n_samples):
                phase_rotation[:, sample_idx] = uniform_filter1d(
                    phase_rotation[:, sample_idx],
                    size=smooth_window,
                    mode='nearest'
                )
        
        return phase_rotation
    
    def apply_frequency_filter(
        self,
        data: np.ndarray,
        sample_rate: float,
        freq_band: Tuple[float, float]
    ) -> np.ndarray:
        """
        Apply bandpass filter to data.
        
        Args:
            data: Input seismic data
            sample_rate: Sample rate in milliseconds
            freq_band: (low_freq, high_freq) in Hz
            
        Returns:
            Filtered data
        """
        nyquist = 1000.0 / (2.0 * sample_rate)
        low = freq_band[0] / nyquist
        high = freq_band[1] / nyquist
        
        self.logger.info(f"Applying bandpass filter: {freq_band[0]}-{freq_band[1]} Hz")
        
        b, a = butter(4, [low, high], btype='band')
        
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        
        return filtered
    
    def compute_envelope(self, data: np.ndarray) -> np.ndarray:
        """
        Compute envelope (instantaneous amplitude) of seismic data.
        
        Args:
            data: Input seismic data
            
        Returns:
            Envelope (instantaneous amplitude)
        """
        envelope = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            analytic = hilbert(data[i])
            envelope[i] = np.abs(analytic)
        
        return envelope
    
    def detect_reflectors(
        self,
        data: np.ndarray,
        envelope: np.ndarray,
        threshold_percentile: float = 70.0
    ) -> np.ndarray:
        """
        Detect reflector locations using envelope and gradient.
        
        Args:
            data: Seismic data
            envelope: Envelope attribute
            threshold_percentile: Percentile threshold for reflector detection
            
        Returns:
            Binary mask where reflectors are detected
        """
        # Calculate envelope threshold
        threshold = np.percentile(envelope, threshold_percentile)
        
        # Detect high amplitude events
        reflector_mask = envelope > threshold
        
        # Also consider zero-crossings with high gradient
        gradient = np.gradient(data, axis=1)
        high_gradient = np.abs(gradient) > np.percentile(np.abs(gradient), 80)
        
        # Combine criteria
        reflectors = reflector_mask & high_gradient
        
        return reflectors.astype(float)
    
    def create_phase_rotation_indicator(
        self,
        phase_rotation: np.ndarray,
        reflector_mask: np.ndarray,
        envelope: np.ndarray,
        threshold: float = 45.0,
        taper_width: int = 3
    ) -> np.ndarray:
        """
        Create phase rotation indicator that shows values only at reflectors.
        
        This creates a more interpretable output that highlights phase anomalies
        at specific reflector locations rather than showing phase everywhere.
        
        Args:
            phase_rotation: Phase rotation attribute
            reflector_mask: Binary mask of reflector locations
            envelope: Envelope for weighting
            threshold: Minimum phase rotation to show (degrees)
            taper_width: Width of taper around reflectors (samples)
            
        Returns:
            Phase rotation indicator (sparse attribute)
        """
        from scipy.ndimage import maximum_filter
        
        # Initialize output
        indicator = np.zeros_like(phase_rotation)
        
        # Expand reflector mask slightly to capture nearby samples
        if taper_width > 0:
            expanded_mask = maximum_filter(
                reflector_mask, 
                size=(1, taper_width)
            ) > 0
        else:
            expanded_mask = reflector_mask > 0
        
        # Apply phase rotation only where we have reflectors
        indicator = phase_rotation * expanded_mask
        
        # Apply threshold - only show significant rotations
        indicator[np.abs(indicator) < threshold] = 0
        
        # Weight by envelope strength (stronger reflectors = more confident)
        envelope_norm = envelope / (np.percentile(envelope, 99) + 1e-10)
        indicator = indicator * envelope_norm
        
        self.logger.info(
            f"Created phase rotation indicator with {np.sum(indicator != 0)} "
            f"non-zero samples ({100*np.sum(indicator != 0)/indicator.size:.2f}%)"
        )
        
        return indicator
    
    def normalize_attribute(
        self,
        attribute: np.ndarray,
        method: str = 'percentile',
        clip_percentile: float = 95
    ) -> np.ndarray:
        """
        Normalize attribute to [-1, 1] range.
        
        Args:
            attribute: Input attribute
            method: 'percentile' or 'absolute'
            clip_percentile: Percentile for clipping
            
        Returns:
            Normalized attribute
        """
        if method == 'percentile':
            vmax = np.percentile(np.abs(attribute), clip_percentile)
        else:
            vmax = np.abs(attribute).max()
        
        if vmax > 0:
            return np.clip(attribute / vmax, -1, 1)
        return attribute
    
    def write_segy_file(
        self,
        data: np.ndarray,
        output_path: Path,
        reference_file: Path,
        preserve_headers: bool = True
    ) -> None:
        """
        Write phase rotation attribute to SEG-Y file.
        
        Args:
            data: Phase rotation attribute
            output_path: Output file path
            reference_file: Reference file for header copying
            preserve_headers: Preserve all headers from reference
        """
        try:
            spec = segyio.spec()
            spec.format = 1
            spec.samples = range(data.shape[1])
            spec.tracecount = data.shape[0]
            
            with segyio.create(str(output_path), spec) as dst:
                if preserve_headers and reference_file.exists():
                    with segyio.open(str(reference_file), 'r', 
                                   ignore_geometry=True) as src:
                        # Copy headers
                        dst.text[0] = src.text[0]
                        dst.bin = src.bin
                        
                        # Copy trace headers
                        for i in range(min(spec.tracecount, src.tracecount)):
                            dst.header[i] = src.header[i]
                
                # Write phase rotation data
                for i in range(data.shape[0]):
                    dst.trace[i] = data[i].astype(np.float32)
            
            self.logger.info(f"Phase rotation attribute written to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write SEG-Y file: {str(e)}")
            raise IOError(f"Cannot write SEG-Y file: {output_path}") from e
    
    def process_file(
        self,
        filepath: Path,
        config: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray]:
        """
        Process single file to detect phase rotations.
        
        Args:
            filepath: Input file path
            config: Configuration dictionary
            
        Returns:
            Tuple of (original_data, phase_rotation, phase_indicator, metadata, seafloor_picks)
        """
        # Read data
        data, metadata = self.read_segy_file(filepath)
        
        # Apply trace decimation if specified
        if config.get('TRACE_DECIMATION', 1) > 1:
            decim = config['TRACE_DECIMATION']
            data = data[::decim, :]
            self.logger.info(f"Decimated to {data.shape[0]} traces")
        
        # Apply frequency filter if specified
        if config.get('FREQUENCY_BAND') is not None:
            data = self.apply_frequency_filter(
                data,
                metadata['sample_rate'],
                config['FREQUENCY_BAND']
            )
        
        # Compute envelope for reflector detection
        envelope = self.compute_envelope(data)
        
        # Detect reference horizon
        reference_type = config.get('REFERENCE_TYPE', 'seafloor')
        
        if reference_type == 'seafloor':
            seafloor_picks = self.detect_seafloor(
                data,
                metadata['sample_rate'],
                config.get('SEAFLOOR_SEARCH_START', 0),
                config.get('SEAFLOOR_SEARCH_END', 500),
                config.get('SEAFLOOR_THRESHOLD', 0.6)
            )
            horizon_samples = seafloor_picks
            
        elif reference_type == 'manual':
            # Use manual time window
            start_sample = int(config['MANUAL_REFERENCE_START'] / metadata['sample_rate'])
            horizon_samples = np.full(data.shape[0], start_sample)
            seafloor_picks = horizon_samples
            
        else:  # auto
            # Auto-detect strongest reflector
            avg_envelope = np.mean(envelope, axis=0)
            peak_sample = np.argmax(avg_envelope)
            horizon_samples = np.full(data.shape[0], peak_sample)
            seafloor_picks = horizon_samples
        
        # Extract reference phase from horizon
        reference_phase = self.extract_reference_phase(
            data,
            horizon_samples,
            window=config.get('ANALYSIS_WINDOW', 32)
        )
        
        # Compute instantaneous phase
        inst_phase = self.compute_instantaneous_phase(data)
        
        # Compute phase rotation
        phase_rotation = self.compute_phase_rotation(
            inst_phase,
            reference_phase,
            smooth=config.get('SMOOTH_PHASES', True),
            smooth_window=config.get('SMOOTHING_WINDOW', 5)
        )
        
        # Detect reflectors
        reflector_mask = self.detect_reflectors(
            data, 
            envelope,
            threshold_percentile=config.get('REFLECTOR_THRESHOLD_PERCENTILE', 70.0)
        )
        
        # Create phase rotation indicator (sparse, only at reflectors)
        phase_indicator = self.create_phase_rotation_indicator(
            phase_rotation,
            reflector_mask,
            envelope,
            threshold=config.get('PHASE_ROTATION_THRESHOLD', 45.0),
            taper_width=config.get('INDICATOR_TAPER_WIDTH', 3)
        )
        
        # Apply gain if specified
        if config.get('APPLY_GAIN', False):
            gain = config.get('GAIN_FACTOR', 2.0)
            phase_indicator *= gain
            phase_rotation *= gain  # Also apply to full phase for consistency
            self.logger.debug(f"Applied gain factor: {gain}")
        
        # Normalize if specified
        if config.get('NORMALIZE_OUTPUT', True):
            max_val = np.abs(phase_indicator).max()
            if max_val > 0:
                phase_indicator = phase_indicator / max_val
            
            max_val_full = np.abs(phase_rotation).max()
            if max_val_full > 0:
                phase_rotation = phase_rotation / max_val_full
        
        # Store seafloor in metadata
        metadata['seafloor_picks'] = seafloor_picks
        metadata['reference_type'] = reference_type
        metadata['reflector_count'] = int(np.sum(reflector_mask > 0))
        metadata['phase_indicator_sparsity'] = float(
            100 * (1 - np.sum(phase_indicator != 0) / phase_indicator.size)
        )
        
        self.logger.info(
            f"Phase indicator sparsity: {metadata['phase_indicator_sparsity']:.1f}% "
            f"(showing values at {100 - metadata['phase_indicator_sparsity']:.1f}% of samples)"
        )
        
        return data, phase_rotation, phase_indicator, metadata, seafloor_picks
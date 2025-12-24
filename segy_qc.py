"""
phase_rotation_qc.py - Quality Control and Visualization
=========================================================

QC tools and visualization for phase rotation detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional
import logging
import seaborn as sns

sns.set_style("whitegrid")


class PhaseRotationQC:
    """Quality control and visualization for phase rotation detection."""
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        """Initialize QC module."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def create_detailed_plot(
        self,
        original_data: np.ndarray,
        phase_rotation: np.ndarray,
        metadata: Dict,
        seafloor_picks: np.ndarray,
        filename: str,
        output_name: str = 'phase_rotation_analysis.png'
    ) -> None:
        """
        Create comprehensive analysis plot.
        
        Args:
            original_data: Original seismic data
            phase_rotation: Phase rotation attribute
            metadata: Metadata dictionary
            seafloor_picks: Seafloor picks in samples
            filename: Original filename
            output_name: Output plot filename
        """
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        sample_rate = metadata['sample_rate']
        n_traces, n_samples = original_data.shape
        time_axis = np.arange(n_samples) * sample_rate
        
        # Panel 1: Original seismic data
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_seismic_section(
            ax1, original_data, sample_rate,
            'Original Seismic Data', seafloor_picks
        )
        
        # Panel 2: Phase rotation attribute
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_phase_rotation(
            ax2, phase_rotation, sample_rate,
            'Phase Rotation Attribute', seafloor_picks
        )
        
        # Panel 3: Phase rotation histogram
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_phase_histogram(ax3, phase_rotation)
        
        # Panel 4: Amplitude vs Phase crossplot
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_amplitude_phase_crossplot(ax4, original_data, phase_rotation)
        
        # Panel 5: Vertical profile comparison
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_vertical_profiles(
            ax5, original_data, phase_rotation, sample_rate
        )
        
        # Panel 6: Statistics panel
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_statistics_panel(
            ax6, original_data, phase_rotation, seafloor_picks, sample_rate
        )
        
        fig.suptitle(f'Phase Rotation Analysis: {filename}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Analysis plot saved to {output_path}")
    
    def _plot_seismic_section(
        self, ax, data, sample_rate, title, seafloor_picks=None
    ):
        """Plot seismic section with optional horizon overlay."""
        n_traces, n_samples = data.shape
        extent = [0, n_traces, n_samples * sample_rate, 0]
        
        vmax = np.percentile(np.abs(data), 99)
        im = ax.imshow(
            data.T, aspect='auto', cmap='seismic',
            extent=extent, vmin=-vmax, vmax=vmax,
            interpolation='bilinear'
        )
        
        # Overlay seafloor picks
        if seafloor_picks is not None:
            seafloor_times = seafloor_picks * sample_rate
            ax.plot(range(len(seafloor_times)), seafloor_times,
                   'y-', linewidth=2, label='Reference Horizon')
            ax.legend(loc='upper right')
        
        ax.set_xlabel('Trace Number', fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Amplitude')
    
    def _plot_phase_rotation(
        self, ax, phase_rotation, sample_rate, title, seafloor_picks=None
    ):
        """Plot phase rotation attribute."""
        n_traces, n_samples = phase_rotation.shape
        extent = [0, n_traces, n_samples * sample_rate, 0]
        
        im = ax.imshow(
            phase_rotation.T, aspect='auto', cmap='RdBu',
            extent=extent, vmin=-180, vmax=180,
            interpolation='bilinear'
        )
        
        # Overlay seafloor picks
        if seafloor_picks is not None:
            seafloor_times = seafloor_picks * sample_rate
            ax.plot(range(len(seafloor_times)), seafloor_times,
                   'k-', linewidth=2, label='Reference Horizon')
            ax.legend(loc='upper right')
        
        ax.set_xlabel('Trace Number', fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, label='Phase Rotation (degrees)')
        cbar.set_ticks([-180, -90, 0, 90, 180])
    
    def _plot_phase_histogram(self, ax, phase_rotation):
        """Plot histogram of phase rotation values."""
        data = phase_rotation.flatten()
        
        ax.hist(data, bins=100, color='steelblue',
               edgecolor='black', alpha=0.7, range=(-180, 180))
        
        ax.axvline(0, color='red', linestyle='--', 
                  linewidth=2, label='Zero Phase')
        ax.axvline(90, color='orange', linestyle='--',
                  linewidth=1.5, label='90° Rotation')
        ax.axvline(-90, color='orange', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel('Phase Rotation (degrees)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Phase Rotation Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_phase = np.mean(data)
        std_phase = np.std(data)
        ax.text(0.02, 0.98, 
               f'Mean: {mean_phase:.1f}°\nStd: {std_phase:.1f}°',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_amplitude_phase_crossplot(self, ax, seismic, phase_rotation):
        """Create amplitude vs phase rotation crossplot."""
        # Sample data for performance
        sample_size = min(10000, seismic.size)
        indices = np.random.choice(seismic.size, sample_size, replace=False)
        
        amp_samples = seismic.flatten()[indices]
        phase_samples = phase_rotation.flatten()[indices]
        
        # Create 2D histogram
        h = ax.hist2d(amp_samples, phase_samples, bins=50,
                     cmap='YlOrRd', cmin=1)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Amplitude', fontweight='bold')
        ax.set_ylabel('Phase Rotation (degrees)', fontweight='bold')
        ax.set_title('Amplitude vs Phase Crossplot', fontweight='bold')
        plt.colorbar(h[3], ax=ax, label='Density')
        ax.grid(True, alpha=0.3)
    
    def _plot_vertical_profiles(self, ax, seismic, phase_rotation, sample_rate):
        """Plot vertical profiles at selected trace locations."""
        n_traces = seismic.shape[0]
        time_axis = np.arange(seismic.shape[1]) * sample_rate
        
        # Select 3 representative traces
        trace_indices = [n_traces // 4, n_traces // 2, 3 * n_traces // 4]
        colors = ['blue', 'green', 'red']
        labels = ['Trace 1/4', 'Trace 1/2', 'Trace 3/4']
        
        for idx, color, label in zip(trace_indices, colors, labels):
            # Normalize amplitude for comparison
            amp = seismic[idx]
            amp_norm = amp / (np.abs(amp).max() + 1e-10)
            
            phase = phase_rotation[idx] / 180.0  # Normalize to [-1, 1]
            
            ax.plot(amp_norm, time_axis, color=color, 
                   linewidth=1.5, label=f'{label} - Amplitude', alpha=0.7)
            ax.plot(phase, time_axis, color=color,
                   linewidth=1.5, linestyle='--', 
                   label=f'{label} - Phase', alpha=0.7)
        
        ax.set_xlabel('Normalized Value', fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('Vertical Profiles', fontweight='bold')
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    def _plot_statistics_panel(
        self, ax, seismic, phase_rotation, seafloor_picks, sample_rate
    ):
        """Create statistics summary panel."""
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate statistics
        avg_seafloor = np.mean(seafloor_picks) * sample_rate
        
        phase_data = phase_rotation.flatten()
        mean_phase = np.mean(phase_data)
        std_phase = np.std(phase_data)
        min_phase = np.min(phase_data)
        max_phase = np.max(phase_data)
        
        # Count anomalies (>45 degrees)
        anomalies = np.abs(phase_data) > 45
        anomaly_percent = 100 * np.sum(anomalies) / len(phase_data)
        
        table_data = [
            ['Parameter', 'Value'],
            ['', ''],
            ['Reference Horizon', ''],
            ['Avg Seafloor Time', f'{avg_seafloor:.1f} ms'],
            ['', ''],
            ['Phase Rotation Statistics', ''],
            ['Mean Phase', f'{mean_phase:.2f}°'],
            ['Std Deviation', f'{std_phase:.2f}°'],
            ['Min Phase', f'{min_phase:.2f}°'],
            ['Max Phase', f'{max_phase:.2f}°'],
            ['', ''],
            ['Anomaly Detection', ''],
            ['Samples >45°', f'{anomaly_percent:.2f}%'],
        ]
        
        table = ax.table(cellText=table_data, cellLoc='left',
                        loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style section headers
        for row in [2, 5, 11]:
            for col in range(2):
                table[(row, col)].set_facecolor('#D9E2F3')
                table[(row, col)].set_text_props(weight='bold')
        
        ax.set_title('Processing Statistics', fontweight='bold', pad=20)
    
    def create_comparison_grid(
        self,
        results_list: List[tuple],
        output_name: str = 'phase_rotation_comparison.png'
    ) -> None:
        """
        Create comparison grid for multiple files.
        
        Args:
            results_list: List of (data, phase_rotation, metadata, filename)
            output_name: Output filename
        """
        n_files = len(results_list)
        fig, axes = plt.subplots(n_files, 2, figsize=(16, 4 * n_files))
        
        if n_files == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (data, phase_rot, metadata, filename) in enumerate(results_list):
            sample_rate = metadata['sample_rate']
            
            # Original data
            ax1 = axes[idx, 0]
            self._plot_seismic_section(
                ax1, data, sample_rate,
                f'{Path(filename).stem} - Original', None
            )
            
            # Phase rotation
            ax2 = axes[idx, 1]
            self._plot_phase_rotation(
                ax2, phase_rot, sample_rate,
                f'{Path(filename).stem} - Phase Rotation', None
            )
        
        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison grid saved to {output_path}")


class StatisticsExporter:
    """Export phase rotation statistics and picks."""
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        """Initialize exporter."""
        import csv
        self.csv = csv
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
    
    def export_statistics(
        self,
        stats_list: List[Dict],
        filename: str = 'phase_rotation_statistics.csv'
    ) -> None:
        """Export statistics to CSV."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = [
                'filename', 'n_traces', 'n_samples', 'sample_rate',
                'mean_phase', 'std_phase', 'min_phase', 'max_phase',
                'anomaly_percentage', 'avg_seafloor_time'
            ]
            writer = self.csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_list)
        
        self.logger.info(f"Statistics exported to {output_path}")
    
    def export_seafloor_picks(
        self,
        picks_dict: Dict[str, np.ndarray],
        sample_rate: float,
        filename: str = 'seafloor_picks.csv'
    ) -> None:
        """Export seafloor picks to CSV."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='') as f:
            writer = self.csv.writer(f)
            writer.writerow(['filename', 'trace', 'sample', 'time_ms'])
            
            for fname, picks in picks_dict.items():
                for trace_idx, pick in enumerate(picks):
                    time_ms = pick * sample_rate
                    writer.writerow([fname, trace_idx, pick, f'{time_ms:.2f}'])
        
        self.logger.info(f"Seafloor picks exported to {output_path}")
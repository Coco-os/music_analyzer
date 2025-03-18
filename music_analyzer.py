import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import sys
import scipy.stats
import os
import json
from scipy.signal import find_peaks
from typing import Tuple, List, Dict, Set, Optional, Any, Union
import argparse
import logging
import configparser
from pathlib import Path
import yaml
import datetime


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('music_analyzer')


class MusicAnalyzer:
    """
    Analyzer for musical characteristics in audio files.
    Handles loading configuration, analyzing audio, and generating visualizations.
    """
    
    # Default configuration - will be overridden by external config
    DEFAULT_CONFIG = {
        'analysis': {
            'hop_length': 512,
            'microtonal_threshold': 0.40,
            'magnitude_threshold_percentile': 75,
            'min_melisma_length': 5,
            'melisma_pitch_threshold': 2,
            'onset_distance': 0.1,
            'call_response_regularity_threshold': 0.2,
            'complex_rhythm_threshold': 0.3,
        },
        'visualization': {
            'dpi': 300,
            'main_figsize': (14, 10),
            'spec_figsize': (12, 6),
            'cmap_spectrogram': 'magma',
            'cmap_chroma': 'viridis',
            'cmap_microtonal': 'coolwarm',
        },
        'scales': {
            'Major': [0, 2, 4, 5, 7, 9, 11],
            'Minor': [0, 2, 3, 5, 7, 8, 10],
            'Pentatonic Major': [0, 2, 4, 7, 9],
            'Pentatonic Minor': [0, 3, 5, 7, 10],
            'Berber (North African)': [0, 3, 5, 6, 7, 10],
            'Blues': [0, 3, 5, 6, 7, 10],
            'Harmonic Minor': [0, 2, 3, 5, 7, 8, 11],
            'Melodic Minor': [0, 2, 3, 5, 7, 9, 11],
            'Dorian': [0, 2, 3, 5, 7, 9, 10],
            'Phrygian': [0, 1, 3, 5, 7, 8, 10],
            'Lydian': [0, 2, 4, 6, 7, 9, 11],
            'Mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'Locrian': [0, 1, 3, 5, 6, 8, 10],
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config_path: Path to configuration file (JSON, YAML, or INI)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
        
        # Convert scale lists to sets for faster lookups
        self.scale_patterns = {
            name: set(pattern) 
            for name, pattern in self.config['scales'].items()
        }
        
        # Extract commonly used config values
        self.hop_length = self.config['analysis']['hop_length']
        
        logger.info("Music analyzer initialized")
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from external file.
        
        Args:
            config_path: Path to configuration file
        """
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using defaults")
            return
        
        logger.info(f"Loading configuration from {config_path}")
        
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            elif file_ext in ('.yml', '.yaml'):
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            elif file_ext in ('.ini', '.cfg'):
                config_parser = configparser.ConfigParser()
                config_parser.read(config_path)
                
                # Convert ConfigParser object to nested dict
                user_config = {}
                for section in config_parser.sections():
                    user_config[section] = dict(config_parser[section])
                    
                    # Convert string values to appropriate types
                    for k, v in user_config[section].items():
                        # Try to convert to numeric if possible
                        try:
                            if '.' in v:
                                user_config[section][k] = float(v)
                            else:
                                user_config[section][k] = int(v)
                        except ValueError:
                            # Keep as string if not numeric
                            pass
                
                # Handle scales section specially (convert to lists)
                if 'scales' in user_config:
                    for scale, pattern in user_config['scales'].items():
                        if isinstance(pattern, str):
                            user_config['scales'][scale] = [int(x) for x in pattern.split(',')]
            else:
                logger.error(f"Unsupported config file format: {file_ext}")
                return
            
            # Merge with default config (only override specified values)
            self._merge_config(user_config)
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.
            
        Args:
            audio_path: Path to the audio file
                
        Returns:
            Tuple of (audio time series, sample rate)
        """
        logger.info(f"Loading audio file: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)
        logger.info(f"Loaded audio: {len(y)/sr:.2f}s, {sr} Hz")
        return y, sr

    
    def _merge_config(self, user_config: Dict) -> None:
        """
        Recursively merge user config with default config.
        
        Args:
            user_config: User configuration dictionary
        """
        for key, value in user_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    # Recursively merge nested dictionaries
                    self._merge_config(value)
                else:
                    # Override value
                    self.config[key] = value
            else:
                # Add new key
                self.config[key] = value
    
    def save_default_config(self, output_path: str) -> None:
        """
        Save default configuration to a file.
        
        Args:
            output_path: Path to save the configuration file
        """
        file_ext = os.path.splitext(output_path)[1].lower()
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            if file_ext == '.json':
                with open(output_path, 'w') as f:
                    json.dump(self.DEFAULT_CONFIG, f, indent=2)
            elif file_ext in ('.yml', '.yaml'):
                with open(output_path, 'w') as f:
                    yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False)
            elif file_ext in ('.ini', '.cfg'):
                config = configparser.ConfigParser()
                
                for section, values in self.DEFAULT_CONFIG.items():
                    config[section] = {}
                    for k, v in values.items():
                        if isinstance(v, list):
                            # Convert lists to comma-separated strings
                            config[section][k] = ','.join(map(str, v))
                        else:
                            config[section][k] = str(v)
                
                with open(output_path, 'w') as f:
                    config.write(f)
            else:
                logger.error(f"Unsupported config file format: {file_ext}")
                return
                
            logger.info(f"Default configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving default config: {e}")
    
    def detect_microtonality(self, y: np.ndarray, sr: int):
        """
        Detect microtonal elements, considering glissando and vibrato.

        Args:
            y: Audio time series
            sr: Sampling rate

        Returns:
            Dict with microtonal pitches, times, deviations, vibrato depth, and glissando rate.
        """
        logger.info("Detecting microtonality...")

        # Extract pitch information
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length)

        # Filter out zeros and get valid pitches
        nonzero_mask = pitches > 0
        if not np.any(nonzero_mask):
            logger.warning("No valid pitches detected")
            return {
                "microtonal_pitches": np.array([]),
                "times": np.array([]),
                "deviations": np.array([]),
                "glissando_count": 0,
                "vibrato_count": 0,
                "intentional_microtonality_score": 0.0
            }

        valid_pitches = pitches[nonzero_mask]
        midi_pitches = librosa.hz_to_midi(valid_pitches)

        # Get corresponding times
        times = librosa.times_like(pitches, sr=sr, hop_length=self.hop_length)
        times = np.tile(times, (pitches.shape[0], 1))
        times = times[nonzero_mask]

        # Filter by magnitude
        valid_magnitudes = magnitudes[nonzero_mask]
        magnitude_threshold = np.percentile(valid_magnitudes, self.config['analysis']['magnitude_threshold_percentile'])
        strong_mask = valid_magnitudes > magnitude_threshold

        if not np.any(strong_mask):
            logger.warning("No strong pitches detected after magnitude filtering")
            return {
                "microtonal_pitches": np.array([]),
                "times": np.array([]),
                "deviations": np.array([]),
                "glissando_count": 0,
                "vibrato_count": 0,
                "intentional_microtonality_score": 0.0
            }

        strong_pitches = midi_pitches[strong_mask]
        strong_times = times[strong_mask]

        # Calculate deviation from nearest semitone
        deviations = np.abs(strong_pitches - np.round(strong_pitches))
        microtonal_mask = deviations > self.config['analysis']['microtonal_threshold']

        # Calculate pitch differences for glissando detection
        # Note: pitch_diff is one element shorter than strong_pitches
        pitch_diff = np.abs(np.diff(strong_pitches))
        
        # Create masks of the same length
        # For glissando_mask, we need to pad it to match the length of microtonal_mask
        glissando_mask_padded = np.zeros_like(microtonal_mask, dtype=bool)
        if len(pitch_diff) > 0:  # Make sure pitch_diff is not empty
            glissando_mask = pitch_diff > 0.3  # Adjust threshold
            # Pad the glissando_mask to match the length of microtonal_mask
            # We can either pad at the beginning or end - here we pad at the end
            glissando_mask_padded[:-1] = glissando_mask
        
        # For vibrato detection
        vibrato_mask = (deviations > 0.1) & (deviations < 0.4)

        # Calculate intentional microtonality score
        # Use the padded glissando mask to ensure shapes match
        if np.sum(microtonal_mask) > 0:
            intentional_microtonality = np.sum(microtonal_mask & (glissando_mask_padded | vibrato_mask)) / np.sum(microtonal_mask)
        else:
            intentional_microtonality = 0.0

        logger.info(f"Detected {np.sum(microtonal_mask)} microtonal notes")
        logger.info(f"Glissando detected in {np.sum(glissando_mask_padded)} cases")
        logger.info(f"Vibrato detected in {np.sum(vibrato_mask)} cases")
        logger.info(f"Intentional microtonality score: {intentional_microtonality:.2f}")

        return {
            "microtonal_pitches": strong_pitches[microtonal_mask] if np.any(microtonal_mask) else np.array([]),
            "times": strong_times[microtonal_mask] if np.any(microtonal_mask) else np.array([]),
            "deviations": deviations[microtonal_mask] if np.any(microtonal_mask) else np.array([]),
            "glissando_count": np.sum(glissando_mask_padded),
            "vibrato_count": np.sum(vibrato_mask),
            "intentional_microtonality_score": intentional_microtonality
        }

    def detect_scale(self, y: np.ndarray, sr: int) -> Tuple[str, int, np.ndarray]:
        """
        Detect the musical scale and tonic note.
        
        Args:
            y: Audio time series
            sr: Sampling rate
            
        Returns:
            Tuple of (scale name, tonic note, chroma vector)
        """
        logger.info("Detecting musical scale...")
        
        # Extract chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        mean_chroma = np.mean(chroma, axis=1)
        
        # Find most prominent notes
        detected_notes = np.argsort(mean_chroma)[-7:]  # Get top 7 notes for better matching
        detected_set = set(detected_notes)
        
        # Find most common note as potential tonic
        if len(detected_notes) > 0:
            tonic = scipy.stats.mode(detected_notes, keepdims=True)[0][0]
        else:
            logger.warning("No notes detected for scale analysis")
            return "Unknown", 0, mean_chroma
        
        # Match against known scales
        best_match_name = "Unknown"
        best_match_score = 0
        best_match_percentage = 0
        
        for scale_name, scale_pattern in self.scale_patterns.items():
            # Create the scale pattern starting from our detected tonic
            shifted_scale = {(note + tonic) % 12 for note in scale_pattern}
            
            # Calculate the match score
            intersection = detected_set.intersection(shifted_scale)
            match_score = len(intersection)
            
            # Calculate coverage as percentage
            scale_coverage = len(intersection) / len(scale_pattern)
            detected_coverage = len(intersection) / len(detected_set) if detected_set else 0
            combined_score = (scale_coverage + detected_coverage) / 2
            
            if combined_score > best_match_percentage or (
                combined_score == best_match_percentage and match_score > best_match_score
            ):
                best_match_score = match_score
                best_match_percentage = combined_score
                best_match_name = scale_name
        
        confidence = int(best_match_percentage * 100)
        logger.info(f"Detected scale: {best_match_name} (tonic: {tonic}, confidence: {confidence}%)")
        
        return best_match_name, tonic, mean_chroma
    def detect_melisma(self, y: np.ndarray, sr: int) -> Tuple[int, Optional[np.ndarray]]:
        """
        Detect melismatic singing (ornamental vocal runs).
        
        Args:
            y: Audio time series
            sr: Sampling rate
            
        Returns:
            Tuple of (count of melismatic phrases, pitch contour or None)
        """
        logger.info("Detecting melismatic patterns...")
        
        try:
            # Check if input is valid
            if y is None or len(y) == 0:
                logger.warning("Empty audio input for melisma detection")
                return 0, None
                
            # Add a safety check for NaN or inf values in the input
            if np.any(~np.isfinite(y)):
                logger.warning("Input contains NaN or infinite values")
                # Clean the input
                y = np.nan_to_num(y)
            
            # Extract pitch using PYIN (more accurate for voice)
            # Wrap this in a try-except as it's the likely source of the segfault
            try:
                fmin = librosa.note_to_hz("C2")
                fmax = librosa.note_to_hz("C7")
                
                # Log the parameters for debugging
                logger.debug(f"PYIN parameters: fmin={fmin}, fmax={fmax}, sr={sr}, hop_length={self.hop_length}")
                logger.debug(f"Audio shape: {y.shape}, dtype: {y.dtype}, range: [{np.min(y)}, {np.max(y)}]")
                
                # Use a smaller chunk of audio if it's very long (to avoid memory issues)
                max_duration = 60  # seconds
                max_samples = sr * max_duration
                if len(y) > max_samples:
                    logger.info(f"Audio too long ({len(y)/sr:.1f}s), using first {max_duration}s for melisma detection")
                    y_analysis = y[:max_samples]
                else:
                    y_analysis = y
                    
                pitches, voiced_flag, _ = librosa.pyin(
                    y_analysis, 
                    fmin=fmin,
                    fmax=fmax,
                    sr=sr,
                    hop_length=self.hop_length
                )
            except Exception as e:
                logger.error(f"PYIN algorithm failed: {str(e)}")
                return 0, None
            
            if pitches is None or len(pitches) == 0 or not np.any(~np.isnan(pitches)):
                logger.warning("No valid pitches detected for melisma analysis")
                return 0, None
            
            # Convert to MIDI notes and remove NaN values
            valid_indices = ~np.isnan(pitches)
            if not np.any(valid_indices):
                logger.warning("All pitches are NaN")
                return 0, None
                
            valid_pitches = pitches[valid_indices]
            
            # Safety check for valid pitch values
            if np.any(valid_pitches <= 0):
                logger.warning("Some pitch values are zero or negative, filtering them out")
                positive_mask = valid_pitches > 0
                valid_pitches = valid_pitches[positive_mask]
                
            if len(valid_pitches) == 0:
                logger.warning("No valid positive pitches for melisma detection")
                return 0, None
                
            midi_pitches = librosa.hz_to_midi(valid_pitches)
            
            if len(midi_pitches) < 2:
                logger.warning("Too few valid pitches for melisma detection")
                return 0, None
            
            # Calculate pitch changes
            pitch_diff = np.diff(midi_pitches)
            
            # Mark significant pitch changes
            threshold = self.config['analysis']['melisma_pitch_threshold']
            melismatic_changes = np.abs(pitch_diff) > threshold
            
            # Detect sustained melismatic passages
            min_length = self.config['analysis']['min_melisma_length']
            
            # Safety check for min_length
            if min_length <= 0:
                logger.warning("Invalid min_melisma_length, using default value of 3")
                min_length = 3
                
            window = np.ones(min_length, dtype=int)
            
            # Ensure arrays are compatible for convolution
            if len(melismatic_changes) < min_length:
                logger.warning("Not enough pitch changes for melisma detection")
                return 0, None
                
            sustained_melismas = np.convolve(melismatic_changes.astype(int), window, "same") >= min_length // 2
            
            melisma_count = int(np.sum(sustained_melismas) / min_length)
            logger.info(f"Detected {melisma_count} melismatic phrases")
            
            return melisma_count, midi_pitches
            
        except Exception as e:
            logger.error(f"Error in detect_melisma: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0, None

    def detect_call_response(self, y: np.ndarray, sr: int) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Detect call and response patterns in the music.
        
        Args:
            y: Audio time series
            sr: Sampling rate
            
        Returns:
            Tuple of (detection result, onset times, onset strengths)
        """
        logger.info("Detecting call-response patterns...")
        
        # Get onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_times = librosa.times_like(onset_env, sr=sr, hop_length=self.hop_length)
        
        # Detect peaks in onset strength
        distance_frames = int(self.config['analysis']['onset_distance'] * sr / self.hop_length)
        peaks, properties = find_peaks(
            onset_env,
            height=np.mean(onset_env) * 1.2,  # Only consider significant onsets
            distance=max(1, distance_frames)  # Convert seconds to frames
        )
        
        if len(peaks) < 4:  # Need at least a few phrases to detect patterns
            logger.warning("Too few onset peaks detected for call-response analysis")
            return False, onset_times, onset_env
        
        # Calculate intervals between phrases
        phrase_intervals = np.diff(peaks)
        
        if len(phrase_intervals) == 0:
            return False, onset_times, onset_env
        
        mean_interval = np.mean(phrase_intervals)
        std_interval = np.std(phrase_intervals)
        
        # If intervals are very regular, it might indicate call-response
        regularity = std_interval / mean_interval if mean_interval > 0 else float('inf')
        threshold = self.config['analysis']['call_response_regularity_threshold']
        is_call_response = regularity < threshold
        
        logger.info(f"Call-response detected: {is_call_response} (regularity: {regularity:.3f})")
        
        return is_call_response, onset_times, onset_env
    
    def detect_complex_rhythm(self, y: np.ndarray, sr: int) -> Tuple[float, bool, np.ndarray, np.ndarray]:
        """
        Detect rhythmic complexity and tempo.
        
        Args:
            y: Audio time series
            sr: Sampling rate
            
        Returns:
            Tuple of (tempo, is_complex, beat_times, beat_intervals)
        """
        logger.info("Analyzing rhythmic patterns...")
        
        # Detect beats and tempo
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, 
            sr=sr,
            hop_length=self.hop_length
        )
        
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        
        if len(beat_frames) < 2:
            logger.warning("Too few beats detected for rhythm analysis")
            return tempo, False, np.array([]), np.array([])
        
        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
        
        # Calculate beat intervals and their regularity
        beat_intervals = np.diff(beat_times)
        
        if len(beat_intervals) == 0:
            return tempo, False, beat_times, np.array([])
        
        mean_interval = np.mean(beat_intervals)
        std_interval = np.std(beat_intervals)
        
        # Normalized measure of irregularity
        irregularity = std_interval / mean_interval if mean_interval > 0 else 0
        threshold = self.config['analysis']['complex_rhythm_threshold']
        is_complex = irregularity > threshold
        
        logger.info(f"Tempo: {tempo:.2f} BPM, Complex rhythm: {is_complex} (irregularity: {irregularity:.3f})")
        
        return tempo, is_complex, beat_times, beat_intervals
    
    def create_spectrogram_visualization(self, y: np.ndarray, sr: int, file_stem: str, output_dir: str) -> None:
        """
        Create detailed spectrogram visualizations.
        
        Args:
            y: Audio time series
            sr: Sampling rate
            file_stem: Base filename without extension
            output_dir: Directory to save output files
        """
        logger.info("Creating spectrogram visualizations...")
        
        # Create figure with multiple spectrograms
        plt.figure(figsize=self.config['visualization']['spec_figsize'])
        
        gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])
        
        # Plot 1: Linear-frequency spectrogram
        ax1 = plt.subplot(gs[0, 0])
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, hop_length=self.hop_length)),
            ref=np.max
        )
        img = librosa.display.specshow(
            D, 
            y_axis='linear', 
            x_axis='time',
            sr=sr,
            hop_length=self.hop_length,
            cmap=self.config['visualization']['cmap_spectrogram'],
            ax=ax1
        )
        ax1.set_title('Linear-Frequency Power Spectrogram')
        ax1.label_outer()
        
        # Plot 2: Log-frequency spectrogram
        ax2 = plt.subplot(gs[1, 0], sharex=ax1)
        librosa.display.specshow(
            D, 
            y_axis='log', 
            x_axis='time',
            sr=sr,
            hop_length=self.hop_length,
            cmap=self.config['visualization']['cmap_spectrogram'],
            ax=ax2
        )
        ax2.set_title('Log-Frequency Power Spectrogram')
        
        # Add colorbar TODO Fix colorbar position. Whole plot is a mess.
        plt.colorbar(img, ax=[ax1, ax2], format="%+2.0f dB")
        
        # Plot 3: Chromagram
        ax3 = plt.subplot(gs[0, 1])
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        librosa.display.specshow(
            chroma, 
            y_axis='chroma', 
            x_axis='time',
            sr=sr, 
            hop_length=self.hop_length,
            cmap=self.config['visualization']['cmap_chroma'],
            ax=ax3
        )
        ax3.set_title('Chromagram')
        
        # Plot 4: Spectral contrast
        ax4 = plt.subplot(gs[1, 1], sharex=ax3)
        contrast = librosa.feature.spectral_contrast(
            y=y, 
            sr=sr,
            hop_length=self.hop_length
        )
        librosa.display.specshow(
            contrast, 
            x_axis='time',
            sr=sr, 
            hop_length=self.hop_length,
            cmap=self.config['visualization']['cmap_spectrogram'],
            ax=ax4
        )
        ax4.set_title('Spectral Contrast')
        
        plt.tight_layout()
        spectrogram_path = os.path.join(output_dir, f"spectrogram_{file_stem}.png")
        plt.savefig(spectrogram_path, dpi=self.config['visualization']['dpi'])
        plt.close()
        
        # Create additional advanced visualizations
        plt.figure(figsize=(10, 8))
        
        # Plot 1: MFCC (Mel-frequency cepstral coefficients)
        plt.subplot(2, 1, 1)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        librosa.display.specshow(
            mfccs, 
            x_axis='time',
            sr=sr, 
            hop_length=self.hop_length
        )
        plt.colorbar(format='%+2.0f')
        plt.title('MFCCs')
        
        # Plot 2: Mel Spectrogram
        plt.subplot(2, 1, 2)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=self.hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(
            mel_spec_db,
            y_axis='mel', 
            x_axis='time',
            sr=sr, 
            hop_length=self.hop_length,
            cmap=self.config['visualization']['cmap_spectrogram']
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        plt.tight_layout()
        advanced_spec_path = os.path.join(output_dir, f"advanced_features_{file_stem}.png")
        plt.savefig(advanced_spec_path, dpi=self.config['visualization']['dpi'])
        plt.close()
        
        logger.info(f"Spectrogram visualizations saved to {output_dir}")
    
    def create_feature_visualization(self, analysis_results, file_stem, output_dir):
        """
        Create visualization of extracted features.
        
        Args:
            analysis_results: Dictionary of analysis results
            file_stem: Base filename without extension
            output_dir: Directory to save visualizations
        """
        logger.info("Creating feature visualization...")
        
        try:
            # Get the sampling rate from analysis_results
            sr = analysis_results.get('sample_rate')
            if sr is None:
                logger.error("Sampling rate not found in analysis results")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Plot pitch contour if available
            if 'pitch_contour' in analysis_results and analysis_results['pitch_contour'] is not None:
                pitch_contour = analysis_results['pitch_contour']
                plt.subplot(3, 1, 1)
                plt.title('Pitch Contour')
                plt.plot(np.arange(len(pitch_contour)) * self.hop_length / sr, pitch_contour)
                plt.ylabel('MIDI Note')
                
            # Plot rhythm features
            plt.subplot(3, 1, 2)
            plt.title('Rhythm Features')
            x_pos = np.arange(3)
            features = [
                analysis_results.get('tempo', 0),
                analysis_results.get('rhythm_complexity', 0),
                analysis_results.get('beat_strength', 0)
            ]
            plt.bar(x_pos, features)
            plt.xticks(x_pos, ['Tempo', 'Complexity', 'Beat Strength'])
            
            # Plot tonal features
            plt.subplot(3, 1, 3)
            plt.title('Tonal Features')
            x_pos = np.arange(4)
            features = [
                analysis_results.get('key_clarity', 0),
                analysis_results.get('mode_strength', 0),
                analysis_results.get('harmonic_complexity', 0),
                analysis_results.get('microtonal_count', 0)
            ]
            plt.bar(x_pos, features)
            plt.xticks(x_pos, ['Key Clarity', 'Mode', 'Harmonic Complexity', 'Microtonal'])
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{file_stem}_features.png")
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Feature visualization saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Error creating feature visualization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def analyze_audio(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of an audio file.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save output files and visualizations
            
        Returns:
            Dictionary of analysis results
        """
        logger.info(f"Analyzing audio file: {audio_path}")
        
        # Get file name without extension for outputs
        file_stem = os.path.splitext(os.path.basename(audio_path))[0]
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {}
            
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Create a default output directory and use it
            output_dir = os.path.join("Resultados", file_stem)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        # Load audio file
        try:
            y, sr = self._load_audio(audio_path)
            logger.info(f"Loaded audio: {len(y)/sr:.2f}s, {sr} Hz")
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return {}
        
        # Perform all analyses
        micro_results = self.detect_microtonality(y, sr)
        scale_name, tonic, chroma = self.detect_scale(y, sr)
        melisma_count, pitch_contour = self.detect_melisma(y, sr)
        call_response, onset_times, onset_env = self.detect_call_response(y, sr)
        tempo, complex_rhythm, beat_times, beat_intervals = self.detect_complex_rhythm(y, sr)
        
        # Compile results
        analysis_results = {
            'filename': os.path.basename(audio_path),
            'duration': len(y) / sr,
            'sample_rate': sr,
            'microtonal_pitches': micro_results["microtonal_pitches"],
            'microtonal_times': micro_results["times"], 
            'deviations': micro_results["deviations"],
            'microtonality_score': len(micro_results["microtonal_pitches"]) / max(1, len(y) / sr * 10),  # Normalized by duration
            'intentional_microtonality_score': micro_results["intentional_microtonality_score"],
            'glissando_count': micro_results["glissando_count"],
            'vibrato_count': micro_results["vibrato_count"],
            'scale': scale_name,
            'tonic': tonic,
            'tonic_note': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][tonic],
            'chroma': chroma,
            'melisma_count': melisma_count,
            'melisma_score': min(1.0, melisma_count / 10),  # Normalized to 0-1
            'pitch_contour': pitch_contour,
            'call_response': call_response,
            'tempo': tempo,
            'complex_rhythm': complex_rhythm,
            'beat_times': beat_times,
            'onset_times': onset_times,
            'onset_env': onset_env,
            'rhythm_complexity_score': 0.2 + (0.8 * int(complex_rhythm)),  # Base score + bonus if complex
        }
        
        # Calculate overall cultural characteristics score
        cultural_features = [
            ('Microtonality', analysis_results['microtonality_score']),
            ('Melismatic Singing', analysis_results['melisma_score']),
            ('Call and Response', float(call_response)),
            ('Rhythmic Complexity', analysis_results['rhythm_complexity_score'])
        ]
        analysis_results['cultural_features'] = cultural_features
        
        # Generate visualizations
        # Always generate visualizations since we have a valid output_dir now
        self.create_spectrogram_visualization(y, sr, file_stem, output_dir)
        self.create_feature_visualization(analysis_results, file_stem, output_dir)
        self.generate_report(analysis_results, file_stem, output_dir)

        logger.info(f"Analysis complete for {os.path.basename(audio_path)}")
        return analysis_results

    def generate_report(self, analysis_results: Dict[str, Any], file_stem: str, output_dir: str) -> None:
        """
        Generate a comprehensive text report of the analysis results.
        
        Args:
            analysis_results: Dictionary of analysis results
            file_stem: Base filename without extension
            output_dir: Directory to save output files
        """
        logger.info("Generating analysis report...")
        
        report_path = os.path.join(output_dir, f"report_{file_stem}.txt")
        
        try:
            with open(report_path, 'w') as f:
                f.write("=== MUSIC ANALYSIS REPORT ===\n\n")
                f.write(f"File: {analysis_results.get('filename', 'Unknown')}\n")
                f.write(f"Duration: {analysis_results.get('duration', 0):.2f} seconds\n")
                f.write(f"Sample Rate: {analysis_results.get('sample_rate', 0)} Hz\n\n")
                
                # Scale information
                f.write("--- SCALE ANALYSIS ---\n")
                f.write(f"Detected Scale: {analysis_results.get('scale', 'Unknown')}\n")
                f.write(f"Tonic Note: {analysis_results.get('tonic_note', 'Unknown')} " + 
                        f"(MIDI: {analysis_results.get('tonic', 0)})\n\n")
                
                # Microtonal analysis  TODO Completar el output del archivo
                f.write("--- MICROTONALITY ---\n")
                microtonal_count = len(analysis_results.get('microtonal_pitches', []))
                f.write(f"Microtonal Notes Detected: {microtonal_count}\n")
                f.write(f"Microtonality Score: {analysis_results.get('microtonality_score', 0):.3f}\n")
                f.write(f"Intentional Microtonality Score: {analysis_results.get('intentional_microtonality_score', 0):.3f}\n")
                f.write(f"Glissando Count: {analysis_results.get('glissando_count', 0)}\n")
                f.write(f"Vibrato Count: {analysis_results.get('vibrato_count', 0)}\n\n")


                
                # Melisma analysis
                f.write("--- VOCAL CHARACTERISTICS ---\n")
                f.write(f"Melismatic Phrases: {analysis_results.get('melisma_count', 0)}\n")
                f.write(f"Melisma Score: {analysis_results.get('melisma_score', 0):.3f}\n\n")
                
                # Rhythm analysis
                f.write("--- RHYTHM ANALYSIS ---\n")
                f.write(f"Tempo: {analysis_results.get('tempo', 0):.2f} BPM\n")
                f.write(f"Complex Rhythm: {'Yes' if analysis_results.get('complex_rhythm', False) else 'No'}\n")
                f.write(f"Call and Response Pattern: {'Detected' if analysis_results.get('call_response', False) else 'Not Detected'}\n")
                f.write(f"Rhythm Complexity Score: {analysis_results.get('rhythm_complexity_score', 0):.3f}\n\n")
                
                # Cultural characteristics summary
                f.write("--- CULTURAL CHARACTERISTICS SUMMARY ---\n")
                for feature, score in analysis_results.get('cultural_features', []):
                    f.write(f"{feature}: {score:.3f}\n")
                
                f.write("\n=== END OF REPORT ===\n")
                
            logger.info(f"Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def save_analysis_results(self, analysis_results: Dict[str, Any], output_path: str) -> None:
        """
        Save analysis results to a JSON or YAML file.
        
        Args:
            analysis_results: Dictionary of analysis results
            output_path: Path to save the results file
        """
        logger.info(f"Saving analysis results to {output_path}")
        
        # Filter non-serializable elements
        serializable_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif key not in ['microtonal_pitches', 'microtonal_times', 'deviations', 
                           'chroma', 'pitch_contour', 'beat_times', 'onset_times', 'onset_env']:
                serializable_results[key] = value
        
        file_ext = os.path.splitext(output_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(output_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            elif file_ext in ('.yml', '.yaml'):
                with open(output_path, 'w') as f:
                    yaml.dump(serializable_results, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported output format: {file_ext}")
                return
                
            logger.info(f"Analysis results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")


def main():
    """
    Main entry point for the CLI interface.
    """
    parser = argparse.ArgumentParser(
        description="Analyze musical characteristics in audio files with focus on cultural elements"
    )
    
    parser.add_argument(
        "audio_path", 
        type=str,
        help="Path to audio file for analysis"
    )
    
    parser.add_argument(
        "-o", 
        "--output", 
        type=str,
        help="Directory to save output files",
        default=None
    )
    
    parser.add_argument(
        "-c", 
        "--config", 
        type=str,
        help="Path to configuration file",
        default=None
    )
    
    parser.add_argument(
        "--save-config", 
        type=str,
        help="Save default configuration to specified path",
        default=None
    )
    
    parser.add_argument(
        "--save-results", 
        type=str,
        help="Save analysis results to JSON or YAML file",
        default=None
    )
    
    parser.add_argument(
        "--no-visualizations", 
        action="store_true",
        help="Skip generating visualizations"
    )
    
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create analyzer
    analyzer = MusicAnalyzer(config_path=args.config)
    
    # Save default config if requested
    if args.save_config:
        analyzer.save_default_config(args.save_config)
        if not args.audio_path:
            return
    
    # Analyze audio
    if os.path.exists(args.audio_path):
        results = analyzer.analyze_audio(
            args.audio_path, 
            output_dir=args.output
        )
        
        # Save results if requested
        if args.save_results and results:
            analyzer.save_analysis_results(results, args.save_results)
    else:
        logger.error(f"Audio file not found: {args.audio_path}")


if __name__ == "__main__":
    main()
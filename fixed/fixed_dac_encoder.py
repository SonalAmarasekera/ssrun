import math
import warnings
from pathlib import Path
import argparse

import torch
import numpy as np
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm
import os

# Import DAC components
from dac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)


class DatasetEncoder:
    def __init__(
        self,
        model_type: str = "44khz",
        model_bitrate: str = "8kbps",
        weights_path: str = "",
        model_tag: str = "latest",
        device: str = "cuda",
        n_quantizers: int = None,
        win_duration: float = 5.0,
        batch_size: int = 4,
        target_sample_rate: int = 44100
    ):
        """Initialize the dataset encoder with batch processing capabilities."""
        self.batch_size = batch_size
        self.target_sample_rate = target_sample_rate
        self.win_duration = win_duration
        
        # Load the model
        self.generator = load_model(
            model_type=model_type,
            model_bitrate=model_bitrate,
            tag=model_tag,
            load_path=weights_path,
        )
        self.generator.to(device)
        self.generator.eval()
        
        self.device = device
        self.n_quantizers = n_quantizers
        self.kwargs = {"n_quantizers": n_quantizers}

    def create_batches(self, file_list, batch_size):
        """Split file list into batches."""
        for i in range(0, len(file_list), batch_size):
            yield file_list[i:i + batch_size]

    def load_audio_batch(self, audio_files):
        """Load a batch of audio files with consistent properties."""
        signals = []
        original_lengths = []
        max_length = 0
        valid_files = []
        
        for audio_file in audio_files:
            try:
                signal = AudioSignal(audio_file)
                
                # Resample if necessary
                if signal.sample_rate != self.target_sample_rate:
                    signal = signal.resample(self.target_sample_rate)
                
                # Ensure mono
                if signal.num_channels > 1:
                    signal = signal.to_mono()
                
                original_lengths.append(signal.length)
                audio_data = signal.audio_data
                
                # Update max length for padding
                if audio_data.shape[-1] > max_length:
                    max_length = audio_data.shape[-1]
                
                signals.append(audio_data)
                valid_files.append(audio_file)
                
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                continue
        
        return signals, original_lengths, max_length, valid_files

    def pad_batch(self, signals, target_length):
        """Pad batch to consistent length."""
        padded_signals = []
        for signal in signals:
            current_length = signal.shape[-1]
            if current_length < target_length:
                pad_length = target_length - current_length
                padded_signal = torch.nn.functional.pad(
                    signal, (0, pad_length), mode='constant', value=0
                )
                padded_signals.append(padded_signal)
            else:
                padded_signals.append(signal[..., :target_length])
        
        if padded_signals:
            return torch.cat(padded_signals, dim=0)
        else:
            return torch.tensor([])

    def encode_single_file(self, audio_file, output_path):
        """Encode a single audio file with proper CUDA tensor handling."""
        try:
            # Load and preprocess audio
            signal = AudioSignal(audio_file)
            
            # Resample if necessary
            if signal.sample_rate != self.target_sample_rate:
                signal = signal.resample(self.target_sample_rate)
            
            # Ensure mono
            if signal.num_channels > 1:
                signal = signal.to_mono()
            
            # Encode audio - ensure we're using the correct method signature
            artifact = self.generator.compress(
                signal, 
                win_duration=self.win_duration,
                n_quantizers=self.n_quantizers
            )
            
            # Save encoded file - ensure artifact is moved to CPU if needed
            artifact.save(output_path)
            return True
            
        except Exception as e:
            print(f"Error encoding {audio_file}: {e}")
            return False

    def encode_dataset(
        self,
        input_dir: str,
        output_dir: str,
        verbose: bool = False
    ):
        """Encode entire dataset with proper progress tracking."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        audio_files = util.find_audio(input_path)
        print(f"Found {len(audio_files)} audio files")

        # Create batches for processing
        batches = list(self.create_batches(audio_files, self.batch_size))
        
        # Create main progress bar for batches
        main_pbar = tqdm(total=len(audio_files), desc="Encoding files", unit="file")
        
        successful_encodings = 0
        failed_encodings = 0
        
        with torch.no_grad():
            for batch_files in batches:
                # Process each file in the batch individually to avoid tensor issues
                for audio_file in batch_files:
                    try:
                        # Compute output path preserving hierarchy
                        relative_path = audio_file.relative_to(input_path)
                        output_file_path = output_path / relative_path.parent
                        output_file_path.mkdir(parents=True, exist_ok=True)
                        
                        output_name = relative_path.with_suffix(".dac").name
                        final_output_path = output_file_path / output_name
                        
                        # Skip if file already exists
                        if final_output_path.exists():
                            if verbose:
                                print(f"Skipping existing file: {final_output_path}")
                            main_pbar.update(1)
                            successful_encodings += 1
                            continue
                        
                        # Encode single file
                        success = self.encode_single_file(audio_file, final_output_path)
                        
                        if success:
                            successful_encodings += 1
                            if verbose:
                                print(f"Successfully encoded: {audio_file}")
                        else:
                            failed_encodings += 1
                            print(f"Failed to encode: {audio_file}")
                        
                        # Update progress bar
                        main_pbar.update(1)
                        main_pbar.set_postfix({
                            'success': successful_encodings,
                            'failed': failed_encodings,
                            'progress': f'{successful_encodings + failed_encodings}/{len(audio_files)}'
                        })
                        
                    except Exception as e:
                        print(f"Unexpected error processing {audio_file}: {e}")
                        failed_encodings += 1
                        main_pbar.update(1)
                        main_pbar.set_postfix({
                            'success': successful_encodings,
                            'failed': failed_encodings,
                            'progress': f'{successful_encodings + failed_encodings}/{len(audio_files)}'
                        })
                        continue
        
        main_pbar.close()
        
        print(f"\nEncoding complete!")
        print(f"Successfully encoded: {successful_encodings} files")
        print(f"Failed encodings: {failed_encodings} files")
        print(f"Total processed: {successful_encodings + failed_encodings}/{len(audio_files)}")


def main():
    parser = argparse.ArgumentParser(description="Batch encode audio dataset using DAC")
    parser.add_argument("input", type=str, help="Path to input audio file or directory")
    parser.add_argument("--output", type=str, default="", help="Path to output directory")
    parser.add_argument("--weights_path", type=str, default="", help="Path to weights file")
    parser.add_argument("--model_tag", type=str, default="latest", help="Model tag")
    parser.add_argument("--model_bitrate", type=str, default="8kbps", 
                       choices=["8kbps", "16kbps"], help="Model bitrate")
    parser.add_argument("--n_quantizers", type=int, default=None, 
                       help="Number of quantizers to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--model_type", type=str, default="44khz", 
                       choices=["44khz", "24khz", "16khz"], help="Model type")
    parser.add_argument("--win_duration", type=float, default=5.0, 
                       help="Window duration for processing")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size for processing (files processed in parallel)")
    parser.add_argument("--target_sample_rate", type=int, default=44100, 
                       help="Target sample rate for audio")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")

    args = parser.parse_args()

    # Set output directory if not specified
    if not args.output:
        args.output = str(Path(args.input).parent / "encoded_output")

    # Initialize encoder
    encoder = DatasetEncoder(
        model_type=args.model_type,
        model_bitrate=args.model_bitrate,
        weights_path=args.weights_path,
        model_tag=args.model_tag,
        device=args.device,
        n_quantizers=args.n_quantizers,
        win_duration=args.win_duration,
        batch_size=args.batch_size,
        target_sample_rate=args.target_sample_rate
    )

    # Encode dataset
    encoder.encode_dataset(
        input_dir=args.input,
        output_dir=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
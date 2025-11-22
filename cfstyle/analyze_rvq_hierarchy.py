import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from audiotools import AudioSignal
import gc

# Import from your local dac package structure
from dac.utils import load_model
from dac.model import DACFile

def plot_spectrogram(signal, title, ax, sample_rate):
    """Plots log-spectrogram of an AudioSignal."""
    # signal is AudioSignal on CPU
    # Convert to mono for visualization if needed
    if signal.num_channels > 1:
        signal = signal.to_mono()
    
    audio_data = signal.audio_data.squeeze().numpy()
    
    ax.specgram(audio_data, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='inferno')
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")

def analyze(args):
    device = torch.device(args.device)
    
    # 1. Load Model
    print(f"[INFO] Loading DAC model: {args.model_type} ({args.model_bitrate})")
    model = load_model(
        model_type=args.model_type,
        model_bitrate=args.model_bitrate,
        tag=args.model_tag,
        load_path=args.weights_path,
    )
    model.to(device)
    model.eval()

    # 2. Load Input (Wav or Dac)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    codes = None
    
    if input_path.suffix == ".dac":
        print(f"[INFO] Loading codes from .dac file: {input_path}")
        dac_file = DACFile.load(input_path)
        codes = dac_file.codes.to(device) # [B, N_codebooks, T]
        
    else:
        # Assume Audio (wav, mp3, etc)
        print(f"[INFO] Encoding audio file: {input_path}")
        signal = AudioSignal(input_path)
        
        # Resample to model rate
        signal.resample(model.sample_rate)
        signal = signal.to(device)
        
        # Encode
        with torch.no_grad():
            # Compress handles padding/chunking better
            dac_file = model.compress(signal, win_duration=args.win_duration)
            codes = dac_file.codes.to(device)
            
        # Free input signal from GPU
        del signal
        torch.cuda.empty_cache()

    n_codebooks = codes.shape[1]
    print(f"[INFO] Total Codebooks detected: {n_codebooks}")

    # 3. Define Groups
    k1 = int(n_codebooks * 0.25) # First 25% (Structure)
    k2 = int(n_codebooks * 0.60) # Next 35% (Texture)
    k1 = max(1, k1)
    k2 = max(k1 + 1, k2)
    
    print(f"[INFO] Analysis Groups :: Coarse: 0-{k1} | Mid: {k1}-{k2} | Fine: {k2}-{n_codebooks}")

    # Helper to get latent difference
    def get_latent_difference_signal(c_full, start_idx, end_idx):
        """
        Decodes the latent contribution of codebooks[start:end].
        Optimized to move results to CPU immediately to avoid OOM.
        """
        chunk_len = dac_file.chunk_length
        recons = []
        
        # We iterate chunks to avoid allocating huge Z tensors for long files
        # c_full is on GPU.
        
        with torch.no_grad():
            for i in range(0, c_full.shape[-1], chunk_len):
                # 1. Get codes for this chunk
                c_chunk = c_full[..., i : i + chunk_len]
                
                # 2. Get Z for 0..End
                c_end = c_chunk[:, :end_idx, :]
                z_end = model.quantizer.from_codes(c_end)[0]
                
                if start_idx > 0:
                    # Get Z for 0..Start
                    c_start = c_chunk[:, :start_idx, :]
                    z_start = model.quantizer.from_codes(c_start)[0]
                    
                    # Subtract in Latent Space (Latent Difference)
                    z_diff = z_end - z_start
                else:
                    z_diff = z_end
                
                # 3. Decode the difference latent
                # This is the memory intensive part (ConvTranspose upsampling)
                r = model.decode(z_diff)
                
                # 4. Move to CPU immediately
                recons.append(r.cpu())
                
                # Cleanup intermediates for this chunk
                del c_chunk, c_end, z_end, z_diff, r
                if start_idx > 0:
                    del c_start, z_start
        
        # Force clean before stitching
        torch.cuda.empty_cache()
            
        # Stitch on CPU
        recons = torch.cat(recons, dim=-1)
        
        # Create AudioSignal on CPU
        # We use model.sample_rate from the model on GPU, but signal is CPU
        signal = AudioSignal(recons, model.sample_rate)
        return signal

    # Process Groups one by one, clearing cache in between
    
    print("[INFO] Decoding Coarse Group...")
    sig_coarse = get_latent_difference_signal(codes, 0, k1)
    torch.cuda.empty_cache()
    
    print("[INFO] Decoding Mid Group...")
    sig_mid = get_latent_difference_signal(codes, k1, k2)
    torch.cuda.empty_cache()
    
    print("[INFO] Decoding Fine Group...")
    sig_fine = get_latent_difference_signal(codes, k2, n_codebooks)
    torch.cuda.empty_cache()
    
    print("[INFO] Decoding Full Audio...")
    sig_full = get_latent_difference_signal(codes, 0, n_codebooks)
    torch.cuda.empty_cache()

    # 4. Plotting (All signals are now on CPU)
    print("[INFO] Generating Spectrograms...")
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    plot_spectrogram(sig_coarse, f"Coarse Layers (0-{k1}): Structure/Bass", ax[0], model.sample_rate)
    plot_spectrogram(sig_mid, f"Mid Layers ({k1}-{k2}): Texture/Vocals", ax[1], model.sample_rate)
    plot_spectrogram(sig_fine, f"Fine Layers ({k2}-{n_codebooks}): High Freq/Noise", ax[2], model.sample_rate)
    plot_spectrogram(sig_full, f"Full Reconstruction (0-{n_codebooks})", ax[3], model.sample_rate)
    
    plt.tight_layout()
    output_filename = "dac_analysis_16khz.png"
    plt.savefig(output_filename)
    print(f"[SUCCESS] Analysis saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DAC RVQ Hierarchy")
    parser.add_argument("--input", type=str, required=True, help="Path to .wav or .dac file")
    
    # Model args
    parser.add_argument("--model_type", type=str, default="16khz", help="16khz, 24khz, 44khz")
    parser.add_argument("--model_bitrate", type=str, default="8kbps", help="8kbps, 16kbps")
    parser.add_argument("--model_tag", type=str, default="latest")
    parser.add_argument("--weights_path", type=str, default="", help="Path to local weights")
    
    # Run args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--win_duration", type=float, default=5.0, help="Window duration for chunking")
    
    args = parser.parse_args()
    analyze(args)
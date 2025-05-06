import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from dataset_decoder import SAMPLE_RATE, N_MELS, HOP_LENGTH

def save_audio(waveform, path, sample_rate=SAMPLE_RATE):
    """Save audio waveform to file"""
    # Ensure waveform is in range [-1, 1]
    waveform = np.clip(waveform, -1, 1)
    sf.write(path, waveform, sample_rate)

def visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, latent_mel=None, save_dir='visuals/decoder'):
    """
    Visualize model outputs and expressive parameters
    
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        mel: Original mel spectrogram
        predicted_mel: Reconstructed mel spectrogram
        wave: Predicted waveform
        target_audio: Target audio waveform
        latent_mel: Latent mel representation from model
        save_dir: Directory to save visualizations
    """
    # Determine number of subplots based on whether latent_mel is provided
    n_plots = 4
    
    # Create figure with subplots
    fig, ax = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), 
                          gridspec_kw={'height_ratios': [1, 1, 1, 1.5] })
    
    # Plot original mel
    if mel.dim() == 3 and mel.size(1) == N_MELS:
        # [B, n_mels, T] format
        mel_plot = mel[0].detach().cpu().numpy()
    else:
        # [B, T, n_mels] format
        mel_plot = mel[0].transpose(0, 1).detach().cpu().numpy()
    
    ax[0].imshow(mel_plot, aspect='auto', origin='lower')
    ax[0].set_title('Original Mel Spectrogram')
    ax[0].set_ylabel('Mel Bin')
    
    # Plot predicted mel
    ax[1].imshow(predicted_mel[0].detach().cpu().numpy(), aspect='auto', origin='lower')
    ax[1].set_title('Reconstructed Mel Spectrogram')
    ax[1].set_ylabel('Mel Bin')

    # Plot original mel
    if latent_mel.dim() == 3 and latent_mel.size(1) == N_MELS:
        # [B, n_mels, T] format
        latent_mel_plot = latent_mel[0].detach().cpu().numpy()
    else:
        # [B, T, n_mels] format
        latent_mel_plot = latent_mel[0].transpose(0, 1).detach().cpu().numpy()
    
    ax[2].imshow(latent_mel_plot, aspect='auto', origin='lower')
    ax[2].set_title('latent Mel Spectrogram')
    ax[2].set_ylabel('Mel Bin')
    
    # Plot waveforms
    wave_predicted = wave[0].detach().cpu().numpy()
    wave_target = target_audio[0].detach().cpu().numpy()

    # Plot waveforms
    min_len = min(wave_predicted.shape[0], wave_target.shape[0])
    
    # Trim both arrays to the minimum length
    wave_predicted_aligned = wave_predicted[:min_len]
    wave_target_aligned = wave_target[:min_len]
    time = np.arange(min_len) / SAMPLE_RATE

    waveform_idx = 3 if latent_mel is not None else 2
    ax[waveform_idx].plot(time, wave_predicted_aligned, label='Predicted', color='blue', alpha=0.7)
    ax[waveform_idx].plot(time, wave_target_aligned, label='Target', color='green', alpha=0.5)
    ax[waveform_idx].set_title('Waveform Comparison')
    ax[waveform_idx].set_xlabel('Time (s)')
    ax[waveform_idx].set_ylabel('Amplitude')
    ax[waveform_idx].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()
    
    # Save audio samples
    save_audio(wave[0].detach().cpu().numpy(), f'audio_samples/epoch_{epoch}_batch_{batch_idx}_pred.wav')
    save_audio(target_audio[0].detach().cpu().numpy(), f'audio_samples/epoch_{epoch}_batch_{batch_idx}_target.wav')

def visualize_expressive_params_with_waveform(epoch, batch_idx, wave, params, save_dir='visuals/decoder/params'):
    """
    Create a dedicated visualization for vocal filter parameters overlaid on the waveform
    
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        wave: Predicted waveform
        params: Dictionary of vocal filter parameters
        save_dir: Directory to save visualizations
    """
    if params is None:
        return
    
    # Create figure with a single plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Convert waveform to numpy array
    wave_np = wave[0].detach().cpu().numpy()
    
    # Create time axis for waveform
    sample_time = np.arange(len(wave_np)) / SAMPLE_RATE
    
    # Plot waveform
    ax1.plot(sample_time, wave_np, color='blue', alpha=0.4, label='Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a second y-axis for parameters
    ax2 = ax1.twinx()
    ax2.set_ylabel('Parameter Value', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Create frame-level time axis for parameters
    n_frames = params['harmonic_articulation'].shape[1]
    frame_time = np.arange(n_frames) * HOP_LENGTH / SAMPLE_RATE
    
    # Parameters to plot with colors and line styles - updated for vocal filter parameters
    param_config = {
        'harmonic_articulation': {'color': 'red', 'linestyle': '-', 'linewidth': 2},
        'harmonic_presence_amount': {'color': 'orange', 'linestyle': '-', 'linewidth': 2},
        'harmonic_exciter_amount': {'color': 'green', 'linestyle': '--', 'linewidth': 2},
        'harmonic_breathiness': {'color': 'purple', 'linestyle': '-.', 'linewidth': 2},
        'noise_articulation': {'color': 'brown', 'linestyle': ':', 'linewidth': 2},
        'noise_presence_amount': {'color': 'magenta', 'linestyle': '-', 'linewidth': 2.5},
        'noise_exciter_amount': {'color': 'cyan', 'linestyle': '--', 'linewidth': 2},
        'noise_breathiness': {'color': 'black', 'linestyle': '-.', 'linewidth': 2}
    }
    
    # Plot each parameter
    for param_name, style in param_config.items():
        if param_name in params:
            # Extract parameter values for the first batch example
            param_values = torch.sigmoid(params[param_name][0]).detach().cpu().numpy()
            
            # Reshape if needed
            if len(param_values.shape) > 1:
                param_values = param_values.flatten()
            
            # Use only up to n_frames points
            display_frames = min(len(frame_time), len(param_values))
            ax2.plot(
                frame_time[:display_frames], 
                param_values[:display_frames], 
                label=param_name,
                **style
            )
    
    # Add legend for all lines
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='medium')
    
    ax1.set_title(f'Vocal Filter Parameters vs Waveform (Epoch {epoch}, Batch {batch_idx})')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/params_epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

def visualize_formant_and_range_params(epoch, batch_idx, wave, params, save_dir='visuals/decoder/formants'):
    """
    Create a dedicated visualization for formant offsets and vocal range parameters
    
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        wave: Predicted waveform
        params: Dictionary of vocal filter parameters
        save_dir: Directory to save visualizations
    """
    if params is None:
        return
    
    # Check if formant and range parameters exist
    formant_params = [p for p in ['formant1_offset', 'formant2_offset', 'formant3_offset', 'formant4_offset',
                                 'noise_formant1_offset', 'noise_formant2_offset', 'noise_formant3_offset', 'noise_formant4_offset'] 
                      if p in params]
    range_params = [p for p in ['vocal_range_min', 'vocal_range_max', 'noise_vocal_range_min', 'noise_vocal_range_max'] 
                    if p in params]
    
    if not formant_params and not range_params:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert waveform to numpy array
    wave_np = wave[0].detach().cpu().numpy()
    
    # Create time axis for waveform
    sample_time = np.arange(len(wave_np)) / SAMPLE_RATE
    
    # Create frame-level time axis for parameters
    n_frames = next(iter(params.values())).shape[1]
    frame_time = np.arange(n_frames) * HOP_LENGTH / SAMPLE_RATE
    
    # Plot waveform on both subplots as reference
    for ax in axes:
        ax.plot(sample_time, wave_np, color='blue', alpha=0.2, label='Waveform')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
    
    # Configure formant subplot
    ax_formant = axes[0]
    ax_formant.set_title(f'Formant Offsets (Epoch {epoch}, Batch {batch_idx})')
    ax_formant.set_ylabel('Formant Offset (Hz)')
    
    # Configure vocal range subplot
    ax_range = axes[1]
    ax_range.set_title(f'Vocal Range Parameters (Epoch {epoch}, Batch {batch_idx})')
    ax_range.set_ylabel('Frequency (Hz)')
    
    # Colors for formant offsets
    formant_colors = {
        'formant1_offset': 'red',
        'formant2_offset': 'green',
        'formant3_offset': 'blue',
        'formant4_offset': 'purple',
        'noise_formant1_offset': 'darkred',
        'noise_formant2_offset': 'darkgreen',
        'noise_formant3_offset': 'darkblue',
        'noise_formant4_offset': 'darkviolet'
    }
    
    # Colors for vocal ranges
    range_colors = {
        'vocal_range_min': 'orange',
        'vocal_range_max': 'red',
        'noise_vocal_range_min': 'darkorange',
        'noise_vocal_range_max': 'darkred'
    }
    
    # Plot formant offsets
    for param_name in formant_params:
        # For formant offsets, transform the sigmoid output to the appropriate Hz range
        if 'formant1' in param_name:
            # ±100Hz range for F1
            param_values = (torch.sigmoid(params[param_name][0]) * 2 - 1) * 100
        elif 'formant2' in param_name:
            # ±200Hz range for F2
            param_values = (torch.sigmoid(params[param_name][0]) * 2 - 1) * 200
        elif 'formant3' in param_name:
            # ±300Hz range for F3
            param_values = (torch.sigmoid(params[param_name][0]) * 2 - 1) * 300
        elif 'formant4' in param_name:
            # ±400Hz range for F4
            param_values = (torch.sigmoid(params[param_name][0]) * 2 - 1) * 400
            
        param_values = param_values.detach().cpu().numpy()
        
        # Reshape if needed
        if len(param_values.shape) > 1:
            param_values = param_values.flatten()
        
        # Use only up to n_frames points
        display_frames = min(len(frame_time), len(param_values))
        ax_formant.plot(
            frame_time[:display_frames], 
            param_values[:display_frames], 
            label=param_name,
            color=formant_colors.get(param_name, 'gray'),
            linewidth=2
        )
    
    # Plot vocal range parameters
    for param_name in range_params:
        # For vocal range parameters, transform the sigmoid output to the appropriate Hz range
        if 'min' in param_name:
            # 0-300Hz range for min
            param_values = torch.sigmoid(params[param_name][0]) * 300
        else:  # 'max' in param_name
            # 300-1500Hz range for max
            param_values = 300 + torch.sigmoid(params[param_name][0]) * 1200
            
        param_values = param_values.detach().cpu().numpy()
        
        # Reshape if needed
        if len(param_values.shape) > 1:
            param_values = param_values.flatten()
        
        # Use only up to n_frames points
        display_frames = min(len(frame_time), len(param_values))
        ax_range.plot(
            frame_time[:display_frames], 
            param_values[:display_frames], 
            label=param_name,
            color=range_colors.get(param_name, 'gray'),
            linewidth=2
        )
    
    # Add legends
    ax_formant.legend(loc='upper right')
    ax_range.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/formant_range_epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()
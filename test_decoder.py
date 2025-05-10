import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio
import soundfile as sf

# Import updated loss functions
from loss import DecoderLoss, RegisterAwareLoss
from dataset_decoder import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH

from svs import SVS

def extract_audio_from_dataset(batch, device):
    """Extract original audio from dataset"""
    # Simply get the audio from the batch and move it to the device
    return batch['audio'].to(device)

def visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, latent_mel=None, 
                     register_weights=None, formant_frequencies=None, save_dir='visuals/decoder'):
    """
    Visualize model outputs, expressive parameters, and register information
    
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        mel: Original mel spectrogram
        predicted_mel: Reconstructed mel spectrogram
        wave: Predicted waveform
        target_audio: Target audio waveform
        latent_mel: Latent mel representation from model
        register_weights: Predicted register weights [B, T, 3]
        formant_frequencies: Predicted formant frequencies [B, T, num_formants]
        save_dir: Directory to save visualizations
    """
    # Determine number of subplots based on available data
    n_plots = 4
    if register_weights is not None:
        n_plots += 1
    if formant_frequencies is not None:
        n_plots += 1
    
    # Create figure with subplots
    fig, ax = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    
    plot_idx = 0
    
    # Plot original mel
    if mel.dim() == 3 and mel.size(1) == N_MELS:
        # [B, n_mels, T] format
        mel_plot = mel[0].detach().cpu().numpy()
    else:
        # [B, T, n_mels] format
        mel_plot = mel[0].transpose(0, 1).detach().cpu().numpy()
    
    ax[plot_idx].imshow(mel_plot, aspect='auto', origin='lower')
    ax[plot_idx].set_title('Original Mel Spectrogram')
    ax[plot_idx].set_ylabel('Mel Bin')
    plot_idx += 1
    
    # Plot predicted mel
    ax[plot_idx].imshow(predicted_mel[0].detach().cpu().numpy(), aspect='auto', origin='lower')
    ax[plot_idx].set_title('Reconstructed Mel Spectrogram')
    ax[plot_idx].set_ylabel('Mel Bin')
    plot_idx += 1

    # Plot latent mel if available
    if latent_mel is not None:
        if latent_mel.dim() == 3 and latent_mel.size(1) == N_MELS:
            # [B, n_mels, T] format
            latent_mel_plot = latent_mel[0].detach().cpu().numpy()
        else:
            # [B, T, n_mels] format
            latent_mel_plot = latent_mel[0].transpose(0, 1).detach().cpu().numpy()
        
        ax[plot_idx].imshow(latent_mel_plot, aspect='auto', origin='lower')
        ax[plot_idx].set_title('Latent Mel Spectrogram')
        ax[plot_idx].set_ylabel('Mel Bin')
        plot_idx += 1
    
    # Plot register weights if available
    if register_weights is not None:
        register_names = ['Chest', 'Mixed', 'Head']
        register_weights_np = register_weights[0].detach().cpu().numpy()  # [T, 3]
        time_frames = np.arange(register_weights_np.shape[0])
        
        for i, name in enumerate(register_names):
            ax[plot_idx].plot(time_frames, register_weights_np[:, i], 
                            label=f'{name} Register', linewidth=2)
        
        ax[plot_idx].set_title('Vocal Register Weights')
        ax[plot_idx].set_ylabel('Weight')
        ax[plot_idx].set_ylim(0, 1)
        ax[plot_idx].grid(True, alpha=0.3)
        ax[plot_idx].legend()
        plot_idx += 1
    
    # Plot formant frequencies if available
    if formant_frequencies is not None:
        formant_freq_np = formant_frequencies[0].detach().cpu().numpy()  # [T, num_formants]
        time_frames = np.arange(formant_freq_np.shape[0])
        
        for i in range(formant_freq_np.shape[1]):
            ax[plot_idx].plot(time_frames, formant_freq_np[:, i], 
                            label=f'F{i+1}', linewidth=2)
        
        ax[plot_idx].set_title('Formant Frequencies')
        ax[plot_idx].set_ylabel('Frequency (Hz)')
        ax[plot_idx].set_ylim(0, 5000)
        ax[plot_idx].grid(True, alpha=0.3)
        ax[plot_idx].legend()
        plot_idx += 1
    
    # Plot waveforms
    wave_predicted = wave[0].detach().cpu().numpy()
    wave_target = target_audio[0].detach().cpu().numpy()

    # Plot waveforms
    min_len = min(wave_predicted.shape[0], wave_target.shape[0])
    
    # Trim both arrays to the minimum length
    wave_predicted_aligned = wave_predicted[:min_len]
    wave_target_aligned = wave_target[:min_len]
    time = np.arange(min_len) / SAMPLE_RATE

    ax[plot_idx].plot(time, wave_predicted_aligned, label='Predicted', color='blue', alpha=0.7)
    ax[plot_idx].plot(time, wave_target_aligned, label='Target', color='green', alpha=0.5)
    ax[plot_idx].set_title('Waveform Comparison')
    ax[plot_idx].set_xlabel('Time (s)')
    ax[plot_idx].set_ylabel('Amplitude')
    ax[plot_idx].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()
    
    # Save audio samples
    save_audio(wave[0].detach().cpu().numpy(), f'audio_samples/epoch_{epoch}_batch_{batch_idx}_pred.wav')
    save_audio(target_audio[0].detach().cpu().numpy(), f'audio_samples/epoch_{epoch}_batch_{batch_idx}_target.wav')

def visualize_expressive_params_with_waveform(epoch, batch_idx, wave, params, save_dir='visuals/decoder/params'):
    """
    Create a dedicated visualization for expressive parameters overlaid on the waveform
    
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        wave: Predicted waveform
        params: Dictionary of expressive parameters
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
    n_frames = next(iter([v for v in params.values() if isinstance(v, torch.Tensor)])).shape[1]
    frame_time = np.arange(n_frames) * HOP_LENGTH / SAMPLE_RATE
    
    # Parameters to plot with colors and line styles
    param_config = {
        'register_weights': {'color': 'red', 'linestyle': '-', 'linewidth': 2},
        'formant_frequencies': {'color': 'orange', 'linestyle': '-', 'linewidth': 2},
        'formant_bandwidths': {'color': 'purple', 'linestyle': '--', 'linewidth': 2},
        'formant_amplitudes': {'color': 'brown', 'linestyle': '-.', 'linewidth': 2},
    }
    
    # Plot each parameter if available
    for param_name, style in param_config.items():
        if param_name in params and isinstance(params[param_name], torch.Tensor):
            # Extract parameter values for the first batch example
            param_values = params[param_name][0].detach().cpu().numpy()
            
            # Special handling for register weights (multiple lines)
            if param_name == 'register_weights':
                register_names = ['Chest', 'Mixed', 'Head']
                for i, name in enumerate(register_names):
                    ax2.plot(
                        frame_time[:param_values.shape[0]], 
                        param_values[:, i],
                        label=f'{name} Register',
                        color=f'C{i+3}',
                        linestyle='-',
                        linewidth=2
                    )
            # Special handling for formant frequencies (multiple lines)
            elif param_name == 'formant_frequencies':
                for i in range(min(3, param_values.shape[1])):  # Plot first 3 formants
                    ax2.plot(
                        frame_time[:param_values.shape[0]],
                        param_values[:, i],
                        label=f'F{i+1}',
                        color=f'C{i+6}',
                        linestyle='-',
                        linewidth=2
                    )
            # Other parameters (single line)
            else:
                # Reshape if needed
                if len(param_values.shape) > 1:
                    param_values = param_values.mean(axis=1)  # Average across dimension
                
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
    
    ax1.set_title(f'Register Parameters vs Waveform (Epoch {epoch}, Batch {batch_idx})')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/params_epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

def save_audio(waveform, path, sample_rate=SAMPLE_RATE):
    """Save audio waveform to file"""
    # Ensure waveform is in range [-1, 1]
    waveform = np.clip(waveform, -1, 1)
    sf.write(path, waveform, sample_rate)

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size(model):
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def train_epoch(model, dataloader, criterion, register_criterion, optimizer, device, epoch, mel_transform, 
                register_loss_weight=0.1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mel_loss = 0
    total_stft_loss = 0
    total_formant_reg_loss = 0
    total_register_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
        # Move data to device
        mel = batch['mel'].to(device)  # This will be [B, T, n_mels]
        f0 = batch['f0'].to(device)
        phoneme_seq = batch['phone_seq_mel'].to(device)  # Get phoneme sequence
        singer_id = batch['singer_id'].to(device).squeeze(1)  # Remove extra dimension
        language_id = batch['language_id'].to(device).squeeze(1)  # Remove extra dimension
        
        # Extract original audio for comparison
        target_audio = extract_audio_from_dataset(batch, device)
        
        # Forward pass
        optimizer.zero_grad()
        wave, latent_mel, _, model_params = model(f0, phoneme_seq, singer_id, language_id)
        
        # Compute combined loss with formant regularization
        combined_loss, loss_dict, predicted_mel = criterion(wave, target_audio, mel_transform, model_params)
        
        # Compute register-aware loss
        register_loss = register_criterion(model_params['register_weights'], f0)
        
        # Total loss with register loss
        loss = combined_loss + register_loss_weight * register_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_mel_loss += loss_dict['mel_loss'].item()
        total_stft_loss += loss_dict['stft_loss'].item()
        total_formant_reg_loss += loss_dict['formant_reg_loss'].item()
        total_register_loss += register_loss.item()
        '''
        # Create visualizations for the first batch every 50 batches
        if batch_idx % 50 == 0:
            # Regular visualization
            visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, 
                             latent_mel, 
                             model_params['register_weights'],
                             model_params['formant_frequencies'],
                             save_dir='visuals/decoder')
            
            # Parameter-specific visualization
            os.makedirs('visuals/decoder/params', exist_ok=True)
            visualize_expressive_params_with_waveform(
                epoch, batch_idx, wave, model_params, save_dir='visuals/decoder/params'
            )
        '''
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_mel_loss = total_mel_loss / len(dataloader)
    avg_stft_loss = total_stft_loss / len(dataloader)
    avg_formant_reg_loss = total_formant_reg_loss / len(dataloader)
    avg_register_loss = total_register_loss / len(dataloader)
    
    return {
        'total': avg_loss, 
        'mel': avg_mel_loss, 
        'stft': avg_stft_loss,
        'formant_reg': avg_formant_reg_loss,
        'register': avg_register_loss
    }

def evaluate(model, dataloader, criterion, register_criterion, device, epoch, mel_transform, 
            register_loss_weight=0.1, visualize=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    total_stft_loss = 0
    total_formant_reg_loss = 0
    total_register_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            # Move data to device
            mel = batch['mel'].to(device)  # This will be [B, T, n_mels]
            f0 = batch['f0'].to(device)
            phoneme_seq = batch['phone_seq_mel'].to(device)  # Get phoneme sequence
            singer_id = batch['singer_id'].to(device).squeeze(1)  # Remove extra dimension
            language_id = batch['language_id'].to(device).squeeze(1)  # Remove extra dimension
            
            # Extract original audio for comparison
            target_audio = extract_audio_from_dataset(batch, device)
            
            # Forward pass
            wave, latent_mel, _, model_params = model(f0, phoneme_seq, singer_id, language_id)
            
            # Compute combined loss with formant regularization
            combined_loss, loss_dict, predicted_mel = criterion(wave, target_audio, mel_transform, model_params)
            
            # Compute register-aware loss
            register_loss = register_criterion(model_params['register_weights'], f0)
            
            # Total loss with register loss
            loss = combined_loss + register_loss_weight * register_loss
            
            total_loss += loss.item()
            total_mel_loss += loss_dict['mel_loss'].item()
            total_stft_loss += loss_dict['stft_loss'].item()
            total_formant_reg_loss += loss_dict['formant_reg_loss'].item()
            total_register_loss += register_loss.item()
            
            # Visualize only the first batch if requested
            if visualize and batch_idx == 0:
                # Regular visualization
                visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, 
                                 latent_mel, 
                                 model_params['register_weights'],
                                 model_params['formant_frequencies'],
                                 save_dir='visuals/decoder/val')
                
                # Parameter-specific visualization
                os.makedirs('visuals/decoder/val/params', exist_ok=True)
                visualize_expressive_params_with_waveform(
                    epoch, batch_idx, wave, model_params, save_dir='visuals/decoder/val/params'
                )
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_mel_loss = total_mel_loss / len(dataloader)
        avg_stft_loss = total_stft_loss / len(dataloader)
        avg_formant_reg_loss = total_formant_reg_loss / len(dataloader)
        avg_register_loss = total_register_loss / len(dataloader)
        
        return {
            'total': avg_loss, 
            'mel': avg_mel_loss, 
            'stft': avg_stft_loss,
            'formant_reg': avg_formant_reg_loss,
            'register': avg_register_loss
        }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create visualization directories
    os.makedirs('visuals/decoder', exist_ok=True)
    os.makedirs('visuals/decoder/val', exist_ok=True)
    os.makedirs('audio_samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # torch stuff
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Load dataset
    batch_size = 32  # Smaller batch size for complex model
    num_epochs = 500
    visualization_interval = 10  # Visualize every 10 epochs

    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=1,
        train_files=200,
        val_files=10,
        device=device,
        context_window_sec=2,  # 2-second window
        persistent_workers=True
    )
    
    # Get dataset parameters
    num_phonemes = len(train_dataset.phone_map)
    num_singers = len(train_dataset.singer_map)
    num_languages = len(train_dataset.language_map)
    
    # Create model
    model = SVS(
        num_phonemes=num_phonemes,
        num_singers=num_singers,
        num_languages=num_languages,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        sample_rate=SAMPLE_RATE
    ).to(device)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size:.2f} MB")
    
    # Create loss functions
    criterion = DecoderLoss(
        stft_loss_weight=0.6,
        mel_loss_weight=0.3,
        formant_reg_weight=0.1  # Weight for formant regularization
    ).to(device)
    
    # Register-aware loss with weights
    register_criterion = RegisterAwareLoss(
        sharpness_weight=0.05,
        smoothness_weight=0.05
    ).to(device)
    
    # Weight for register loss in total loss
    register_loss_weight = 0.1
    
    # Mel transform for extracting mel spectrogram from predicted audio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=WIN_LENGTH,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=1  # Use amplitude instead of power
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler - reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=50,
        verbose=True
    )
    
    # Training loop    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_losses = train_epoch(
            model, train_loader, criterion, register_criterion, 
            optimizer, device, epoch, mel_transform, register_loss_weight
        )
        
        # Visualize during evaluation at certain intervals
        should_visualize = (epoch % visualization_interval == 0)
        val_losses = evaluate(
            model, val_loader, criterion, register_criterion,
            device, epoch, mel_transform, register_loss_weight, 
            visualize=should_visualize
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_losses['total'])
        
        # Print training information
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_losses['total']:.4f} (Mel: {train_losses['mel']:.4f}, "
              f"STFT: {train_losses['stft']:.4f}, Formant Reg: {train_losses['formant_reg']:.4f}, "
              f"Register: {train_losses['register']:.4f})")
        print(f"  Val Loss: {val_losses['total']:.4f} (Mel: {val_losses['mel']:.4f}, "
              f"STFT: {val_losses['stft']:.4f}, Formant Reg: {val_losses['formant_reg']:.4f}, "
              f"Register: {val_losses['register']:.4f})")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(model.state_dict(), 'best_decoder_model.pth')
            print(f"  Saved best model with val loss: {val_losses['total']:.4f}")
        
        # Also save regular checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
            }, f'checkpoints/decoder_checkpoint_epoch_{epoch}.pth')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio
import soundfile as sf

from data_utils import get_total_files, load_mappings
from dataset import get_dataloader

from loss import HybridLoss
from svs import SVS

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

def extract_audio_from_dataset(batch, device):
    return batch['audio'].to(device)

def visualize_outputs(config, epoch, batch_idx, mel, predicted_mel, wave, target_audio, latent_mel=None, save_dir='visuals'):
    """
    Visualize model outputs and expressive parameters
    
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        mel: Original mel spectrogram
        predicted_mel: Reconstructed mel spectrogram
        wave: Predicted waveform
        target_audio: Target audio waveform
        expressive_params: Dictionary of expressive parameters
        latent_mel: Latent mel representation from model
        save_dir: Directory to save visualizations
    """
    # Determine number of subplots based on whether latent_mel is provided
    n_plots = 4
    
    # Create figure with subplots
    fig, ax = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), 
                          gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
    
    # Plot original mel
    if mel.dim() == 3 and mel.size(1) == config['model']['n_mels']:
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
    if latent_mel.dim() == 3 and latent_mel.size(1) == config['model']['n_mels']:
        # [B, n_mels, T] format
        latent_mel_plot = latent_mel[0].detach().cpu().numpy()
    else:
        # [B, T, n_mels] format
        latent_mel_plot = latent_mel[0].transpose(0, 1).detach().cpu().numpy()
    
    ax[2].imshow(latent_mel_plot, aspect='auto', origin='lower')
    ax[2].set_title('Latent Mel Space')
    ax[2].set_ylabel('Mel Bin')
    
    # Plot waveforms
    wave_predicted = wave[0].detach().cpu().numpy()
    wave_target = target_audio[0].detach().cpu().numpy()

    # Plot waveforms
    min_len = min(wave_predicted.shape[0], wave_target.shape[0])
    
    # Trim both arrays to the minimum length
    wave_predicted_aligned = wave_predicted[:min_len]
    wave_target_aligned = wave_target[:min_len]
    time = np.arange(min_len) / config['model']['sample_rate']

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
    save_audio(wave[0].detach().cpu().numpy(), f'audio_samples/epoch_{epoch}_batch_{batch_idx}_pred.wav', config['model']['sample_rate'])
    save_audio(target_audio[0].detach().cpu().numpy(), f'audio_samples/epoch_{epoch}_batch_{batch_idx}_target.wav', config['model']['sample_rate'])

def save_audio(waveform, path, sample_rate):
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

def train_epoch(config, model, dataloader, criterion, optimizer, device, epoch, mel_transform):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_stft_loss = 0
    total_astft_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    for batch_idx, batch in enumerate(pbar):
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
        wave, latent_mel = model(f0, phoneme_seq, singer_id, language_id)
        
        # Compute combined loss
        loss, stft_loss, astft_loss = criterion(wave, target_audio)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_stft_loss += stft_loss.item()
        total_astft_loss += astft_loss.item()

        # Update progress bar with current batch loss
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'stft_loss': f'{stft_loss.item():.4f}',
            'astft_loss': f'{astft_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_stft_loss = total_stft_loss / len(dataloader)
    aavg_stft_loss = total_astft_loss / len(dataloader)

    return avg_loss, avg_stft_loss, aavg_stft_loss

def evaluate(config, model, dataloader, criterion, device, epoch, mel_transform, visualize=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_stft_loss = 0
    total_astft_loss = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating', leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            mel = batch['mel'].to(device)  # This will be [B, T, n_mels]
            f0 = batch['f0'].to(device)
            phoneme_seq = batch['phone_seq_mel'].to(device)  # Get phoneme sequence
            singer_id = batch['singer_id'].to(device).squeeze(1)  # Remove extra dimension
            language_id = batch['language_id'].to(device).squeeze(1)  # Remove extra dimension
            
            # Extract original audio for comparison
            target_audio = extract_audio_from_dataset(batch, device)
            
            # Forward pass
            wave, latent_mel = model(f0, phoneme_seq, singer_id, language_id)
            
            # Compute combined loss
            loss, stft_loss, astft_loss = criterion(wave, target_audio)
            
            total_loss += loss.item()
            total_stft_loss += stft_loss.item()
            total_astft_loss += astft_loss.item()
            
            # Update progress bar with current batch loss
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'stft_loss': f'{stft_loss.item():.4f}',
                'astft_loss': f'{astft_loss.item():.4f}'
            })

            predicted_mel = torch.log(mel_transform(wave))
            # Visualize only the first batch if requested
            if visualize and batch_idx == 0:
                # Regular visualization with parameters and latent_mel
                visualize_outputs(config, epoch, batch_idx, mel, predicted_mel, wave, target_audio, 
                                 latent_mel, save_dir='visuals')
        
        avg_loss = total_loss / len(dataloader)
        avg_stft_loss = total_stft_loss / len(dataloader)
        aavg_stft_loss = total_astft_loss / len(dataloader)

        return avg_loss, avg_stft_loss, aavg_stft_loss

def train_stage(config, device, stage, num_epochs, train_loader, val_loader, model, criterion, optimizer, scheduler, mel_transform, visualization_interval):
    
    print(f"starting training stage: {stage}")

    # Training loop    
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", leave=False)
    for epoch in epoch_pbar:
        train_loss, train_mel_loss, train_stft_loss = train_epoch( config,
            model, train_loader, criterion, optimizer, device, epoch, mel_transform
        )
        
        # Visualize during evaluation at certain intervals
        should_visualize = (epoch % visualization_interval == 0)
        val_loss, val_mel_loss, val_stft_loss = evaluate( config,
            model, val_loader, criterion, device, epoch, mel_transform, visualize=should_visualize
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Update epoch progress bar with loss information
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  Saved best model with val loss: {val_loss:.4f}")
        
        # Also save regular checkpoints
        if epoch % config['training']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'stage' : stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, f'checkpoints/checkpoint_stage_{stage}_epoch_{epoch}.pth')
    
    print(f"\nStage {stage} Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

def main():
    
    # Create directories
    os.makedirs('visuals', exist_ok=True)
    os.makedirs('audio_samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    dataset_dir = config['data']['dataset_dir']
    map_file = config['data']['map_file']

    mappings = load_mappings(map_file)
    total_files = get_total_files(dataset_dir, mappings['singer_map'], mappings['language_map'])

    print( 'Setting Up Training Stage...' )
    print( f'Total training files: {total_files}' )
    
    # stage training
    current_stage = 0
    stage_files = config['training']['stage_files']
    max_stage = total_files // stage_files
    num_epochs = config['training']['num_epochs']
    visualization_interval = config['log']['visualization_interval']
    
    print( f'Stage training files: {stage_files}' )
    print( f'Epoch per stage: {num_epochs}' )
    print(f"Max training stage: {max_stage}")

    # continue checkpoint
    checkpoint_path = config['training']['checkpoint_path']
    batch_size = config['training']['batch_size']

    # Set up model
    num_phonemes = config['model']['phone_num']
    num_singers = config['model']['singer_num']
    num_languages = config['model']['lang_num']
    
    # Create model
    model = SVS(
        num_phonemes=num_phonemes,
        num_singers=num_singers,
        num_languages=num_languages,
        n_mels=config['model']['n_mels'],
        hop_length=config['model']['hop_length'],
        sample_rate=config['model']['sample_rate'],
        num_harmonics=config['model']['num_harmonics'],
        num_mag_harmonic=config['model']['num_mag_harmonic'],
        num_mag_noise=config['model']['num_mag_noise']
    ).to(device)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel info:")
    print(f"  Num singer: {num_phonemes}")
    print(f"  Num lang: {num_singers}")
    print(f"  Num phone: {num_languages}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size:.2f} MB")
    
    # load check point
    if config['training']['continue_ckpt'] is True:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"\nLoaded checkpoint : {checkpoint_path}")
    
    # Create loss function
    criterion = HybridLoss(
        n_ffts=[1024, 512, 256, 128],
        n_affts=[512+256, 256+128, 128+64]
    ).to(device)
    
    # Mel transform for extracting mel spectrogram from predicted audio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['model']['sample_rate'],
        n_fft=config['model']['win_length'],
        win_length=config['model']['win_length'],
        hop_length=config['model']['hop_length'],
        n_mels=config['model']['n_mels'],
        power=1  # Use amplitude instead of power
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # Learning rate scheduler - reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10000,
    )

    for stage in range(current_stage, max_stage + 1):

        # recreate dataset per stage
        train_loader, val_loader, train_dataset, val_dataset, _ = get_dataloader(
            batch_size=batch_size,
            num_workers=1,
            train_files=stage_files,
            val_files=20,
            device=device,
            context_window_sec=2,  # 2-second window
            persistent_workers=True,
            rebuild_cache=True,
            start_index=stage*stage_files
        )

        train_stage( config=config, device=device, stage=stage, num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                    model=model, criterion=criterion, scheduler=scheduler, mel_transform=mel_transform, visualization_interval=visualization_interval
                    )

if __name__ == "__main__":
    main()
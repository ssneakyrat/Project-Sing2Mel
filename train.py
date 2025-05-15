import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio
import soundfile as sf

from loss import HybridLoss
from dataset_decoder import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH

from svs import SVS
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

def extract_audio_from_dataset(batch, device):
    """Extract original audio from dataset"""
    # Simply get the audio from the batch and move it to the device
    return batch['audio'].to(device)

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
        expressive_params: Dictionary of expressive parameters
        latent_mel: Latent mel representation from model
        save_dir: Directory to save visualizations
    """
    # Determine number of subplots based on whether latent_mel is provided
    n_plots = 4
    
    # Create figure with subplots
    fig, ax = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), 
                          gridspec_kw={'height_ratios': [1, 1, 1.5, 1.5]})
    
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

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, mel_transform):
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

def evaluate(model, dataloader, criterion, device, epoch, mel_transform, visualize=False):
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
                visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, 
                                 latent_mel, save_dir='visuals/decoder/val')
        
        avg_loss = total_loss / len(dataloader)
        avg_stft_loss = total_stft_loss / len(dataloader)
        aavg_stft_loss = total_astft_loss / len(dataloader)

        return avg_loss, avg_stft_loss, aavg_stft_loss

def train_stage(device, stage, num_epochs, train_loader, val_loader, model, criterion, optimizer, scheduler, mel_transform, visualization_interval):
    
    print(f"starting training stage: {stage}")

    # Training loop    
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", leave=False)
    for epoch in epoch_pbar:
        train_loss, train_mel_loss, train_stft_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, mel_transform
        )
        
        # Visualize during evaluation at certain intervals
        should_visualize = (epoch % visualization_interval == 0)
        val_loss, val_mel_loss, val_stft_loss = evaluate(
            model, val_loader, criterion, device, epoch, mel_transform, visualize=should_visualize
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print training information
        '''
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} (Mel: {train_mel_loss:.4f}, STFT: {train_stft_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Mel: {val_mel_loss:.4f}, STFT: {val_stft_loss:.4f})")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        '''
        # Update epoch progress bar with loss information
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_decoder_model.pth')
            print(f"  Saved best model with val loss: {val_loss:.4f}")
        
        # Also save regular checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'stage' : stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, f'checkpoints/decoder_checkpoint_stage_{stage}_epoch_{epoch}.pth')
    
    print(f"\nStage {stage} Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create visualization directories
    os.makedirs('visuals/decoder', exist_ok=True)
    os.makedirs('visuals/decoder/val', exist_ok=True)
    os.makedirs('audio_samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # continue checkpoint
    checkpoint_path = None #'best_decoder_model.pth'        

    batch_size = 32 

    # create dummy dataset to count for phonemes, singer, languages
    train_loader, val_loader, train_dataset, val_dataset, total_files = get_dataloader(
        batch_size=batch_size,
        num_workers=1,
        train_files=4,
        val_files=2,
        device=device,
        context_window_sec=2,  # 2-second window
        persistent_workers=True,
        rebuild_cache=True,
        start_index=0
    )

    # stage training
    current_stage = 0    
    stage_files = 100
    max_stage = total_files // stage_files
    num_epochs = 200 # per stage
    visualization_interval = 5  # Visualize every 5 epochs

    print(f"\nMax training stage: {max_stage}")

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
    
    # load check point
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # Print model info
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size:.2f} MB")
    
    # Create loss function
    criterion = HybridLoss(
        n_ffts=[1024, 512, 256, 128],
        n_affts=[512+256, 256+128, 128+64]
    ).to(device)
    
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

        train_stage(device=device, stage=stage, num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                    model=model, criterion=criterion, scheduler=scheduler, mel_transform=mel_transform, visualization_interval=visualization_interval
                    )

if __name__ == "__main__":
    main()
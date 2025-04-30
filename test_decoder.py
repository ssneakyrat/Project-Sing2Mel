import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio
import soundfile as sf
from dataset_decoder import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH

from mel_decoder import MelDecoder

# Create visuals folder if it doesn't exist
os.makedirs('visuals', exist_ok=True)
os.makedirs('visuals/decoder', exist_ok=True)
os.makedirs('audio_samples', exist_ok=True)

class DecoderLoss(nn.Module):
    """Combined loss function for mel decoder including STFT loss"""
    def __init__(self, stft_loss_weight=0.5, mel_loss_weight=0.5, 
                 stft_n_ffts=[512, 1024, 2048], stft_hop_lengths=[50, 120, 240], 
                 stft_win_lengths=[240, 600, 1200]):
        super().__init__()
        self.stft_loss_weight = stft_loss_weight
        self.mel_loss_weight = mel_loss_weight
        
        # Mel loss
        self.mel_criterion = nn.L1Loss()
        
        # Multi-resolution STFT loss components
        self.stft_n_ffts = stft_n_ffts
        self.stft_hop_lengths = stft_hop_lengths
        self.stft_win_lengths = stft_win_lengths
        
        self.stft_mag_criterion = nn.L1Loss()
        
    def compute_stft_loss(self, predicted_audio, target_audio):
        """Compute multi-resolution STFT loss"""
        stft_loss = 0.0
        
        for n_fft, hop_length, win_length in zip(self.stft_n_ffts, self.stft_hop_lengths, self.stft_win_lengths):
            # Compute STFT for predicted audio
            pred_stft = torch.stft(
                predicted_audio, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length,
                window=torch.hann_window(win_length).to(predicted_audio.device),
                return_complex=True
            )
            
            # Compute STFT for target audio
            target_stft = torch.stft(
                target_audio, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length,
                window=torch.hann_window(win_length).to(target_audio.device),
                return_complex=True
            )
            
            # Compute magnitude spectrogram
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            if pred_mag.shape[2] > target_mag.shape[2]:
                predicted_mel_aligned = pred_mag[:, :, :target_mag.shape[2]]
                target_mel_aligned = target_mag
            elif target_mag.shape[2] > pred_mag.shape[2]:
                target_mel_aligned = target_mag[:, :, :pred_mag.shape[2]]
                predicted_mel_aligned = pred_mag
            else:
                # Shapes are already the same
                predicted_mel_aligned = pred_mag
                target_mel_aligned = target_mag

            # Magnitude loss
            mag_loss = self.stft_mag_criterion(predicted_mel_aligned, target_mel_aligned)
            
            # Spectral convergence loss
            sc_loss = torch.norm(target_mel_aligned - predicted_mel_aligned, p='fro') / torch.norm(target_mel_aligned, p='fro')
            
            stft_loss += mag_loss + sc_loss
        
        # Average over all resolutions
        stft_loss /= len(self.stft_n_ffts)
        return stft_loss
    
    def forward(self, predicted_audio, target_audio, mel_transform):
        """Compute combined loss"""
        # Compute mel spectrogram from predicted and target audio
        # Add small epsilon to avoid log(0)
        predicted_mel = torch.log(mel_transform(predicted_audio) + 1e-7)
        target_mel = torch.log(mel_transform(target_audio) + 1e-7)
        
        if predicted_mel.shape[2] > target_mel.shape[2]:
            predicted_mel_aligned = predicted_mel[:, :, :target_mel.shape[2]]
            target_mel_aligned = target_mel
        elif target_mel.shape[2] > predicted_mel.shape[2]:
            target_mel_aligned = target_mel[:, :, :predicted_mel.shape[2]]
            predicted_mel_aligned = predicted_mel
        else:
            # Shapes are already the same
            predicted_mel_aligned = predicted_mel
            target_mel_aligned = target_mel

        # Mel spectrogram loss
        mel_loss = self.mel_criterion(predicted_mel_aligned, target_mel_aligned)
        
        # Multi-resolution STFT loss
        stft_loss = self.compute_stft_loss(predicted_audio, target_audio)
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss + 
                     self.stft_loss_weight * stft_loss)
        
        return total_loss, mel_loss, stft_loss, predicted_mel

def extract_audio_from_dataset(batch, device):
    """Extract original audio from dataset"""
    # Simply get the audio from the batch and move it to the device
    return batch['audio'].to(device)

def visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, save_dir='visuals/decoder'):
    """Visualize model outputs"""
    # Create figure with 3 subplots
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
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
    
    wave_predicted = wave[0].detach().cpu().numpy()
    wave_target = target_audio[0].detach().cpu().numpy()

    # Plot waveforms
    time = np.arange(wave.shape[1]) / SAMPLE_RATE

    # Find the minimum length       
    min_len = min(wave_predicted.shape[0], wave_target.shape[0])
    
    # Trim both arrays to the minimum length
    wave_predicted_aligned = wave_predicted[:min_len]
    wave_target_aligned = wave_target[:min_len]
    time = time[:min_len]

    ax[2].plot(time, wave_predicted_aligned, label='Predicted')
    ax[2].plot(time, wave_target_aligned, alpha=0.5, label='Target')
    ax[2].set_title('Waveform Comparison')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Amplitude')
    ax[2].legend()
    
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
    total_mel_loss = 0
    total_stft_loss = 0
    
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
        wave = model(mel, f0, phoneme_seq, singer_id, language_id)
        
        # Compute combined loss
        loss, mel_loss, stft_loss, predicted_mel = criterion(wave, target_audio, mel_transform)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_mel_loss += mel_loss.item()
        total_stft_loss += stft_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mel_loss = total_mel_loss / len(dataloader)
    avg_stft_loss = total_stft_loss / len(dataloader)
    
    return avg_loss, avg_mel_loss, avg_stft_loss

def evaluate(model, dataloader, criterion, device, epoch, mel_transform, visualize=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    total_stft_loss = 0
    
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
            wave = model(mel, f0, phoneme_seq, singer_id, language_id)
            
            # Compute combined loss
            loss, mel_loss, stft_loss, predicted_mel = criterion(wave, target_audio, mel_transform)
            
            total_loss += loss.item()
            total_mel_loss += mel_loss.item()
            total_stft_loss += stft_loss.item()
            
            # Visualize only the first batch if requested
            if visualize and batch_idx == 0:
                visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, save_dir='visuals/decoder/val')
        
        avg_loss = total_loss / len(dataloader)
        avg_mel_loss = total_mel_loss / len(dataloader)
        avg_stft_loss = total_stft_loss / len(dataloader)
        
        return avg_loss, avg_mel_loss, avg_stft_loss

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create visualization directories
    os.makedirs('visuals/decoder', exist_ok=True)
    os.makedirs('visuals/decoder/val', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load dataset
    batch_size = 16  # Smaller batch size for complex model
    num_epochs = 500
    visualization_interval = 10  # Visualize every 5 epochs

    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=4,
        train_files=50,
        val_files=10,
        device=device,
        context_window_sec=2  # 2-second window
    )
    
    # Get dataset parameters
    num_phonemes = len(train_dataset.phone_map)
    num_singers = len(train_dataset.singer_map)
    num_languages = len(train_dataset.language_map)
    
    # Create model
    model = MelDecoder(
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
    
    # Create loss function
    criterion = DecoderLoss(
        stft_loss_weight=0.7,
        mel_loss_weight=0.3
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
        patience=50,
        verbose=True
    )
    
    # Training loop    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
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
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} (Mel: {train_mel_loss:.4f}, STFT: {train_stft_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Mel: {val_mel_loss:.4f}, STFT: {val_stft_loss:.4f})")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_decoder_model.pth')
            print(f"  Saved best model with val loss: {val_loss:.4f}")
        
        # Also save regular checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, f'checkpoints/decoder_checkpoint_epoch_{epoch}.pth')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
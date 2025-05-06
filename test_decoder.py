import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio

from loss import DecoderLoss
from dataset_decoder import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH

from utils_log import visualize_outputs, visualize_formant_and_range_params, visualize_expressive_params_with_waveform
from svs import SVS

# Create visuals folder if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('visuals', exist_ok=True)
os.makedirs('visuals/decoder', exist_ok=True)
os.makedirs('audio_samples', exist_ok=True)
os.makedirs('visuals/decoder/params', exist_ok=True)  # New folder for parameter visualization
os.makedirs('visuals/decoder/formants', exist_ok=True)  # New folder for formant and vocal range visualization

def extract_audio_from_dataset(batch, device):
    """Extract original audio from dataset"""
    # Simply get the audio from the batch and move it to the device
    return batch['audio'].to(device)



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
        wave, latent_mel, vocal_params = model(f0, phoneme_seq, singer_id, language_id)
        
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
        
        # Visualize parameters occasionally during training
        if batch_idx == 0 and epoch % 20 == 0:
            visualize_expressive_params_with_waveform(epoch, batch_idx, wave, vocal_params)
            # Add the new formant and range visualization
            visualize_formant_and_range_params(epoch, batch_idx, wave, vocal_params)
    
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
            wave, latent_mel, vocal_params = model(f0, phoneme_seq, singer_id, language_id)
            
            # Compute combined loss
            loss, mel_loss, stft_loss, predicted_mel = criterion(wave, target_audio, mel_transform)
            
            total_loss += loss.item()
            total_mel_loss += mel_loss.item()
            total_stft_loss += stft_loss.item()
            
            # Visualize only the first batch if requested
            if visualize and batch_idx == 0:
                # Regular visualization with parameters and latent_mel
                visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, 
                                 latent_mel, save_dir='visuals/decoder/val')
                                 
                # Visualize vocal parameters
                visualize_expressive_params_with_waveform(epoch, batch_idx, wave, vocal_params, 
                                                       save_dir='visuals/decoder/params')
                
                # Add the new formant and range visualization
                visualize_formant_and_range_params(epoch, batch_idx, wave, vocal_params,
                                                save_dir='visuals/decoder/formants')
        
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
    os.makedirs('visuals/decoder/params', exist_ok=True)
    os.makedirs('visuals/decoder/formants', exist_ok=True)  # New directory for formant visualizations
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load dataset
    batch_size = 64  # Smaller batch size for complex model
    num_epochs = 1000
    visualization_interval = 5  # Visualize every 20 epochs

    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=1,
        train_files=950,
        val_files=50,
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
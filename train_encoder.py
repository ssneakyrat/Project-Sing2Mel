import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio
import soundfile as sf

from loss.encoder_loss import EncoderLoss
from dataset_decoder import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH

from sing2mel import Sing2Mel

def extract_audio_from_dataset(batch, device):
    """Extract original audio from dataset"""
    # Simply get the audio from the batch and move it to the device
    return batch['audio'].to(device)

def visualize_outputs(epoch, batch_idx, predicted_mel, target_mel, save_dir='visuals/encoder'):
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
    n_plots = 2
    
    # Create figure with subplots
    fig, ax = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), 
                          gridspec_kw={'height_ratios': [1, 1] })
    
    # Plot original mel
    if target_mel.dim() == 3 and target_mel.size(1) == N_MELS:
        # [B, n_mels, T] format
        mel_plot = target_mel[0].detach().cpu().numpy()
    else:
        # [B, T, n_mels] format
        mel_plot = target_mel[0].transpose(0, 1).detach().cpu().numpy()
    
    # Plot original mel
    if predicted_mel.dim() == 3 and predicted_mel.size(1) == N_MELS:
        # [B, n_mels, T] format
        predicted_plot = predicted_mel[0].detach().cpu().numpy()
    else:
        # [B, T, n_mels] format
        predicted_plot = predicted_mel[0].transpose(0, 1).detach().cpu().numpy()

    ax[0].imshow(mel_plot, aspect='auto', origin='lower')
    ax[0].set_title('Original Mel Spectrogram')
    ax[0].set_ylabel('Mel Bin')
    
    # Plot predicted mel
    ax[1].imshow(predicted_plot, aspect='auto', origin='lower')
    ax[1].set_title('Predicted Mel Spectrogram')
    ax[1].set_ylabel('Mel Bin')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

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

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mel_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
        # Move data to device
        mel = batch['mel'].to(device)  # This will be [B, T, n_mels]
        f0 = batch['f0'].to(device)
        phoneme_seq = batch['phone_seq_mel'].to(device)  # Get phoneme sequence
        singer_id = batch['singer_id'].to(device).squeeze(1)  # Remove extra dimension
        language_id = batch['language_id'].to(device).squeeze(1)  # Remove extra dimension
        
        # Forward pass
        optimizer.zero_grad()
        predicted_mel = model(f0, phoneme_seq, singer_id, language_id)
        
        # Compute combined loss
        loss, mel_loss = criterion(predicted_mel, mel)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_mel_loss += mel_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mel_loss = total_mel_loss / len(dataloader)
    
    return avg_loss, avg_mel_loss

def evaluate(model, dataloader, criterion, device, epoch, visualize=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            # Move data to device
            mel = batch['mel'].to(device)  # This will be [B, T, n_mels]
            f0 = batch['f0'].to(device)
            phoneme_seq = batch['phone_seq_mel'].to(device)  # Get phoneme sequence
            singer_id = batch['singer_id'].to(device).squeeze(1)  # Remove extra dimension
            language_id = batch['language_id'].to(device).squeeze(1)  # Remove extra dimension
            
            # Forward pass
            predicted_mel = model(f0, phoneme_seq, singer_id, language_id)
            
            # Compute combined loss
            loss, mel_loss = criterion(predicted_mel, mel)
            
            total_loss += loss.item()
            total_mel_loss += mel_loss.item()
            
            # Visualize only the first batch if requested
            if visualize and batch_idx == 0:
                # Regular visualization with parameters and latent_mel
                visualize_outputs(epoch, batch_idx, predicted_mel, mel, save_dir='visuals/encoder/val')
        
        avg_loss = total_loss / len(dataloader)
        avg_mel_loss = total_mel_loss / len(dataloader)
        
        return avg_loss, avg_mel_loss

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create visualization directories
    os.makedirs('visuals/encoder', exist_ok=True)
    os.makedirs('visuals/encoder/val', exist_ok=True)
    os.makedirs('checkpoints/encoder', exist_ok=True)
    
    # Load dataset
    batch_size = 32  # Smaller batch size for complex model
    num_epochs = 2000
    visualization_interval = 10  # Visualize every 5 epochs

    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=1,
        train_files=None,
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
    model = Sing2Mel(
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
    criterion = EncoderLoss(
        mel_loss_weight=1
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
        train_loss, train_mel_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Visualize during evaluation at certain intervals
        should_visualize = (epoch % visualization_interval == 0)
        val_loss, val_mel_loss = evaluate(
            model, val_loader, criterion, device, epoch, visualize=should_visualize
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print training information
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} (Mel: {train_mel_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Mel: {val_mel_loss:.4f})")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/encoder/best_encoder_model.pth')
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
            }, f'checkpoints/encoder/encoder_checkpoint_epoch_{epoch}.pth')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
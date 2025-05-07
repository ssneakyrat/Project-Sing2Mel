import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio
import soundfile as sf

from mel_loss import MelSynthLoss
from dataset_decoder import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH

from mel_synth import MelSynth

# Create visuals folder if it doesn't exist
os.makedirs('visuals', exist_ok=True)
os.makedirs('visuals/mel_predictor', exist_ok=True)

def extract_audio_from_dataset(batch, device):
    """Extract original audio from dataset"""
    # Simply get the audio from the batch and move it to the device
    return batch['audio'].to(device)

def visualize_outputs(epoch, batch_idx, target_mel, predicted_mel, durations=None, attention_weights=None, save_dir='visuals/mel_predictor'):
    """
    Visualize model outputs
    
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        target_mel: Target mel spectrogram
        predicted_mel: Predicted mel spectrogram
        durations: Phoneme durations (optional)
        attention_weights: Attention weights from model (optional)
        save_dir: Directory to save visualizations
    """
    # Determine number of subplots
    n_plots = 2  # Base: mel spectrograms
    if durations is not None:
        n_plots += 1  # Add durations plot
    if attention_weights is not None:
        n_plots += 1  # Add attention plot
    
    # Create figure with subplots
    fig, ax = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    target_mel_vis = target_mel[0].detach().cpu().numpy()
    predicted_mel_vis = predicted_mel[0].detach().cpu().numpy()
    # Plot target mel
    ax[0].imshow(target_mel_vis.T, aspect='auto', origin='lower')
    ax[0].set_title('Target Mel Spectrogram')
    ax[0].set_ylabel('Mel Bin')
    
    # Plot predicted mel
    ax[1].imshow(predicted_mel_vis.T, aspect='auto', origin='lower')
    ax[1].set_title('Predicted Mel Spectrogram')
    ax[1].set_ylabel('Mel Bin')
    
    plot_idx = 2
    
    # Plot durations if provided
    if durations is not None:
        ax[plot_idx].bar(range(len(durations[0].detach().cpu().numpy())), 
                         durations[0].detach().cpu().numpy())
        ax[plot_idx].set_title('Phoneme Durations')
        ax[plot_idx].set_xlabel('Phoneme Index')
        ax[plot_idx].set_ylabel('Duration (frames)')
        plot_idx += 1
    
    # Plot attention weights if provided
    if attention_weights is not None:
        # Use the last layer's attention weights
        attn = attention_weights[-1][0].detach().cpu().numpy()
        # Get the first head from the multi-head attention
        attn_head = attn[0] if len(attn.shape) > 2 else attn
        ax[plot_idx].imshow(attn_head, aspect='auto', origin='lower')
        ax[plot_idx].set_title('Attention Weights')
        ax[plot_idx].set_xlabel('Encoder Steps')
        ax[plot_idx].set_ylabel('Decoder Steps')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}_batch_{batch_idx}.png')
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

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, mel_transform):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    losses_dict_total = {}
    
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
        
        # Compute loss
        loss, losses_dict = criterion(
            predicted_mel, mel
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accumulate loss components
        for k, v in losses_dict.items():
            if k not in losses_dict_total:
                losses_dict_total[k] = 0
            losses_dict_total[k] += v
    
    # Average the losses
    avg_loss = total_loss / len(dataloader)
    for k in losses_dict_total:
        losses_dict_total[k] /= len(dataloader)
    
    return avg_loss, losses_dict_total

def evaluate(model, dataloader, criterion, device, epoch, mel_transform, visualize=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    losses_dict_total = {}
    
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
            predicted_mel = model(f0, phoneme_seq, singer_id, language_id)
            
            # Compute loss
            loss, losses_dict = criterion(
                predicted_mel, mel
            )
            
            total_loss += loss.item()
            
            # Accumulate loss components
            for k, v in losses_dict.items():
                if k not in losses_dict_total:
                    losses_dict_total[k] = 0
                losses_dict_total[k] += v
            
            # Visualize only the first batch if requested
            if visualize and batch_idx == 0:
                visualize_outputs(
                    epoch, batch_idx, mel, predicted_mel,
                    save_dir='visuals/mel_predictor'
                )
        
        # Average the losses
        avg_loss = total_loss / len(dataloader)
        for k in losses_dict_total:
            losses_dict_total[k] /= len(dataloader)
        
        return avg_loss, losses_dict_total

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    batch_size = 32  # Smaller batch size for complex model
    num_epochs = 1000
    visualization_interval = 10  # Visualize every 5 epochs

    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=1,
        train_files=100,
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
    model = MelSynth(
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
    criterion = MelSynthLoss(
        l1_weight=0.5,
        l2_weight=0.5,
        duration_weight=0.1,
        use_adversarial=False,  # Optional: enable for GAN training
        use_ssim=True,          # Use structural similarity index
        ssim_weight=0.1
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
        train_loss, train_losses_dict  = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, mel_transform
        )
        
        # Visualize during evaluation at certain intervals
        should_visualize = (epoch % visualization_interval == 0)
        val_loss, val_losses_dict  = evaluate(
            model, val_loader, criterion, device, epoch, mel_transform, visualize=should_visualize
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print training information
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        for k, v in train_losses_dict.items():
            print(f"    - {k}: {v:.4f}")
        
        print(f"  Val Loss: {val_loss:.4f}")
        for k, v in val_losses_dict.items():
            print(f"    - {k}: {v:.4f}")
        
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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from dataset import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH
from svm import SingingVoiceModel

# Create visuals folder if it doesn't exist
os.makedirs('visuals', exist_ok=True)

class CombinedLoss(nn.Module):
    """Combined loss with adjustable weights for L1 and spectral losses"""
    def __init__(self, l1_weight=0.5, spectral_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        """Compute combined loss with weighted components"""
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # Combined weighted loss
        total_loss = self.l1_weight * l1
        
        # For monitoring individual components
        return total_loss, l1

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

def visualize_mel_comparison(harmonic_mel, melodic_mel, noise_mel, combined_mel, target_mel, epoch, batch_idx=0, save_path='visuals/eval'):
    """Visualize and compare harmonic, melodic, noise, combined and target mel spectrograms"""
    harmonic_mel_np = harmonic_mel[0].detach().cpu().numpy().T
    melodic_mel_np = melodic_mel[0].detach().cpu().numpy().T
    noise_mel_np = noise_mel[0].detach().cpu().numpy().T
    combined_mel_np = combined_mel[0].detach().cpu().numpy().T
    target_mel_np = target_mel[0].detach().cpu().numpy().T
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 25))
    
    # Plot target mel spectrogram
    im1 = axes[0].imshow(target_mel_np, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Target Mel Spectrogram')
    axes[0].set_ylabel('Mel Frequency')
    axes[0].set_xlabel('Time')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot combined mel spectrogram
    im5 = axes[1].imshow(combined_mel_np, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Combined Mel Spectrogram (Predicted)')
    axes[1].set_ylabel('Mel Frequency')
    axes[1].set_xlabel('Time')
    plt.colorbar(im5, ax=axes[1])

    # Plot harmonic mel spectrogram
    im2 = axes[2].imshow(harmonic_mel_np, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('harmonic Mel Spectrogram (from F0)')
    axes[2].set_ylabel('Mel Frequency')
    axes[2].set_xlabel('Time')
    plt.colorbar(im2, ax=axes[2])

    # Plot temporal mel spectrogram
    im3 = axes[3].imshow(melodic_mel_np, aspect='auto', origin='lower', cmap='viridis')
    axes[3].set_title('Temporal Mel Spectrogram (from phonemes)')
    axes[3].set_ylabel('Mel Frequency')
    axes[3].set_xlabel('Time')
    plt.colorbar(im3, ax=axes[3])
    
    # Plot noise mel spectrogram
    im4 = axes[4].imshow(noise_mel_np, aspect='auto', origin='lower', cmap='viridis')
    axes[4].set_title('Noise Mel Spectrogram')
    axes[4].set_ylabel('Mel Frequency')
    axes[4].set_xlabel('Time')
    plt.colorbar(im4, ax=axes[4])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/mel_comparison_epoch_{epoch:03d}_batch_{batch_idx}.png')
    plt.close()

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    # Explicitly set all submodules to training mode
    for module in model.modules():
        module.train()
    
    total_loss = 0
    total_l1_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
        # Move data to device
        phoneme_seq = batch['phone_seq_mel'].to(device)
        f0 = batch['f0'].to(device)
        mel = batch['mel'].to(device)
        singer_id = batch['singer_id'].squeeze(1).to(device)
        language_id = batch['language_id'].squeeze(1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        combined_mel, temporal_mel, harmonic_mel, noise_mel = model(phoneme_seq, f0, singer_id, language_id)
        
        # Use combined_mel as the prediction
        predicted_mel = combined_mel
        
        # Compute loss
        loss, l1_loss = criterion(predicted_mel, mel)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_l1_loss += l1_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    
    return avg_loss, avg_l1_loss

def evaluate(model, dataloader, criterion, device, epoch, visualize=False):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_l1_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            # Move data to device
            phoneme_seq = batch['phone_seq_mel'].to(device)
            f0 = batch['f0'].to(device)
            mel = batch['mel'].to(device)
            singer_id = batch['singer_id'].squeeze(1).to(device)
            language_id = batch['language_id'].squeeze(1).to(device)
            
            # Forward pass
            combined_mel, temporal_mel, harmonic_mel, noise_mel = model(phoneme_seq, f0, singer_id, language_id)
            
            # Use combined_mel as the prediction
            predicted_mel = combined_mel
            
            # Compute loss
            loss, l1_loss = criterion(predicted_mel, mel)
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            
            # Visualize results every certain batch and epochs
            if visualize and batch_idx == 0:
                visualize_mel_comparison(harmonic_mel, temporal_mel, noise_mel, combined_mel, mel, epoch, batch_idx)
    
    avg_loss = total_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    
    return avg_loss, avg_l1_loss

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create visualization directories
    os.makedirs('visuals/eval', exist_ok=True)
    os.makedirs('visuals/', exist_ok=True)
    
    # Load dataset
    batch_size = 32
    num_epochs = 500
    visualization_interval = 50  # Visualize every 5 epochs

    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=4,
        train_files=100,
        val_files=10,
        device=device,
        context_window_sec=2
    )
    
    # Get dataset info
    num_phonemes = len(train_dataset.phone_map)
    num_singers = len(train_dataset.singer_map)
    num_languages = len(train_dataset.language_map)
    
    print(f"Dataset info:")
    print(f"  Phonemes: {num_phonemes}")
    print(f"  Singers: {num_singers}")
    print(f"  Languages: {num_languages}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create model
    model = SingingVoiceModel(num_phonemes, num_singers, num_languages, n_mels=N_MELS).to(device)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size:.2f} MB")
    
    # Loss function and optimizer
    criterion = CombinedLoss(l1_weight=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    # Training loop    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss, train_l1 = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Visualize during evaluation at certain intervals
        should_visualize = (epoch % visualization_interval == 0)
        val_loss, val_l1 = evaluate(model, val_loader, criterion, device, epoch, visualize=should_visualize)
        
        print(f"Epoch {epoch}:")
        print(f"  Train - Total: {train_loss:.4f}, L1: {train_l1:.4f}")
        print(f"  Val   - Total: {val_loss:.4f}, L1: {val_l1:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  Saved best model with val loss: {val_loss:.4f}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio
import soundfile as sf

from loss import DecoderLoss
from dataset_decoder import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH
from discriminator import FastGANDiscriminator
from svs import SVS

def extract_audio_from_dataset(batch, device):
    """Extract original audio from dataset"""
    # Simply get the audio from the batch and move it to the device
    return batch['audio'].to(device)

def visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, 
                      latent_mel=None, adv_losses=None, save_dir='visuals/decoder'):
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
        adv_losses: Dictionary of adversarial losses to plot
        save_dir: Directory to save visualizations
    """
    # Determine number of subplots based on inputs
    n_plots = 5 if adv_losses else 4
    if latent_mel is None:
        n_plots -= 1
    
    # Create figure with subplots
    fig, ax = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), 
                          gridspec_kw={'height_ratios': [1, 1, 1.5, 1.5] if latent_mel is None else [1, 1, 1.5, 1.5, 1]})
    
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

    # Plot latent mel if provided
    plot_offset = 0
    if latent_mel is not None:
        if latent_mel.dim() == 3 and latent_mel.size(1) == N_MELS:
            # [B, n_mels, T] format
            latent_mel_plot = latent_mel[0].detach().cpu().numpy()
        else:
            # [B, T, n_mels] format
            latent_mel_plot = latent_mel[0].transpose(0, 1).detach().cpu().numpy()
        
        ax[2].imshow(latent_mel_plot, aspect='auto', origin='lower')
        ax[2].set_title('Latent Mel Spectrogram')
        ax[2].set_ylabel('Mel Bin')
        plot_offset = 1
    
    # Plot waveforms
    wave_predicted = wave[0].detach().cpu().numpy()
    wave_target = target_audio[0].detach().cpu().numpy()

    # Trim both arrays to the minimum length
    min_len = min(wave_predicted.shape[0], wave_target.shape[0])
    wave_predicted_aligned = wave_predicted[:min_len]
    wave_target_aligned = wave_target[:min_len]
    time = np.arange(min_len) / SAMPLE_RATE

    waveform_idx = 2 + plot_offset
    ax[waveform_idx].plot(time, wave_predicted_aligned, label='Predicted', color='blue', alpha=0.7)
    ax[waveform_idx].plot(time, wave_target_aligned, label='Target', color='green', alpha=0.5)
    ax[waveform_idx].set_title('Waveform Comparison')
    ax[waveform_idx].set_xlabel('Time (s)')
    ax[waveform_idx].set_ylabel('Amplitude')
    ax[waveform_idx].legend(loc='upper right')
    
    # Plot adversarial losses if provided
    if adv_losses:
        loss_idx = 3 + plot_offset
        # Extract loss keys and values
        keys = list(adv_losses.keys())
        values = [adv_losses[k] for k in keys]
        
        # Create bar plot
        ax[loss_idx].bar(keys, values)
        ax[loss_idx].set_title('Adversarial Loss Components')
        ax[loss_idx].set_ylabel('Loss Value')
        for i, v in enumerate(values):
            ax[loss_idx].text(i, v + 0.01, f"{v:.4f}", ha='center')
    
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

def train_epoch(model, discriminator, dataloader, criterion, 
               optimizer_g, optimizer_d, device, epoch, mel_transform,
               d_steps=1, g_steps=1, adv_weight_schedule=None):
    """Train for one epoch with adversarial loss"""
    model.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    total_mel_loss = 0
    total_stft_loss = 0
    total_adv_loss = 0
    total_fm_loss = 0
    
    # Schedule adversarial weight if provided
    if adv_weight_schedule is not None:
        current_adv_weight = adv_weight_schedule(epoch)
        criterion.adv_loss_weight = current_adv_weight
    
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
        
        # Add channel dimension to target audio for discriminator
        target_audio_disc = target_audio.unsqueeze(1)  # [B, 1, T]
        
        # Generate fake audio
        with torch.no_grad():
            wave, latent_mel, _ = model(f0, phoneme_seq, singer_id, language_id)
            wave_disc = wave.unsqueeze(1)  # [B, 1, T]
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        for _ in range(d_steps):
            optimizer_d.zero_grad()
            
            # Generate fake audio
            with torch.no_grad():
                wave, latent_mel, _ = model(f0, phoneme_seq, singer_id, language_id)
                wave_disc = wave.unsqueeze(1)  # [B, 1, T]
            
            # Process real audio through discriminator
            real_outputs, real_features = discriminator(target_audio_disc)
            
            # Process fake audio through discriminator
            fake_outputs, fake_features = discriminator(wave_disc.detach())
            
            # Compute discriminator loss
            d_loss = criterion.adv_criterion.discriminator_loss(real_outputs, fake_outputs)
            
            # Backward pass and optimize
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()
            
            total_d_loss += d_loss.item()
        
        # ---------------------
        # Train Generator
        # ---------------------
        for _ in range(g_steps):
            optimizer_g.zero_grad()
            
            # Generate fake audio
            wave, latent_mel, _ = model(f0, phoneme_seq, singer_id, language_id)
            wave_disc = wave.unsqueeze(1)  # [B, 1, T]
            
            # Process fake audio through discriminator
            fake_outputs, fake_features = discriminator(wave_disc)
            
            # Process real audio through discriminator (for feature matching)
            with torch.no_grad():
                real_outputs, real_features = discriminator(target_audio_disc)
            
            # Compute combined loss
            g_loss, mel_loss, stft_loss, adv_loss, fm_loss, predicted_mel = criterion(
                wave, target_audio, mel_transform,
                disc_outputs=fake_outputs,
                real_disc_features=real_features,
                fake_disc_features=fake_features,
                train_generator=True
            )
            
            # Backward pass and optimize
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()
            
            total_g_loss += g_loss.item()
            total_mel_loss += mel_loss.item()
            total_stft_loss += stft_loss.item()
            total_adv_loss += adv_loss.item()
            total_fm_loss += fm_loss.item()
        
        # Update progress bar with current batch loss
        pbar.set_postfix({
            'g_loss': f'{g_loss.item():.4f}',
            'd_loss': f'{d_loss.item():.4f}',
            'mel_loss': f'{mel_loss.item():.4f}',
            'adv_weight': f'{criterion.adv_loss_weight:.4f}'
        })
        
        # Visualize first batch or occasionally
        if batch_idx == 0 and epoch % 5 == 0:
            adv_losses = {
                'mel_loss': mel_loss.item(),
                'stft_loss': stft_loss.item(),
                'adv_loss': adv_loss.item(),
                'fm_loss': fm_loss.item()
            }
            visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, 
                             latent_mel, adv_losses, save_dir='visuals/decoder')

    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)
    avg_mel_loss = total_mel_loss / len(dataloader)
    avg_stft_loss = total_stft_loss / len(dataloader)
    avg_adv_loss = total_adv_loss / len(dataloader)
    avg_fm_loss = total_fm_loss / len(dataloader)
    
    return avg_g_loss, avg_d_loss, avg_mel_loss, avg_stft_loss, avg_adv_loss, avg_fm_loss

def evaluate(model, discriminator, dataloader, criterion, device, epoch, mel_transform, visualize=False):
    """Evaluate the model with adversarial components"""
    model.eval()
    discriminator.eval()
    
    total_g_loss = 0
    total_mel_loss = 0
    total_stft_loss = 0
    total_adv_loss = 0
    total_fm_loss = 0
    
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
            target_audio_disc = target_audio.unsqueeze(1)  # [B, 1, T]
            
            # Forward pass
            wave, latent_mel, _ = model(f0, phoneme_seq, singer_id, language_id)
            wave_disc = wave.unsqueeze(1)  # [B, 1, T]
            
            # Process through discriminator
            fake_outputs, fake_features = discriminator(wave_disc)
            real_outputs, real_features = discriminator(target_audio_disc)
            
            # Compute combined loss
            g_loss, mel_loss, stft_loss, adv_loss, fm_loss, predicted_mel = criterion(
                wave, target_audio, mel_transform,
                disc_outputs=fake_outputs,
                real_disc_features=real_features,
                fake_disc_features=fake_features,
                train_generator=True
            )
            
            total_g_loss += g_loss.item()
            total_mel_loss += mel_loss.item()
            total_stft_loss += stft_loss.item()
            total_adv_loss += adv_loss.item()
            total_fm_loss += fm_loss.item()
            
            # Update progress bar with current batch loss
            pbar.set_postfix({
                'g_loss': f'{g_loss.item():.4f}',
                'mel_loss': f'{mel_loss.item():.4f}'
            })

            # Visualize only the first batch if requested
            if visualize and batch_idx == 0:
                adv_losses = {
                    'mel_loss': mel_loss.item(),
                    'stft_loss': stft_loss.item(),
                    'adv_loss': adv_loss.item(),
                    'fm_loss': fm_loss.item()
                }
                visualize_outputs(epoch, batch_idx, mel, predicted_mel, wave, target_audio, 
                                 latent_mel, adv_losses, save_dir='visuals/decoder/val')
        
        avg_g_loss = total_g_loss / len(dataloader)
        avg_mel_loss = total_mel_loss / len(dataloader)
        avg_stft_loss = total_stft_loss / len(dataloader)
        avg_adv_loss = total_adv_loss / len(dataloader)
        avg_fm_loss = total_fm_loss / len(dataloader)
        
        return avg_g_loss, avg_mel_loss, avg_stft_loss, avg_adv_loss, avg_fm_loss

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
    batch_size = 32  # Reduced batch size for GAN training
    num_epochs = 500
    visualization_interval = 1  # Visualize every epoch

    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=1,
        train_files=None,
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
    
    # Mel transform for extracting mel spectrogram from predicted audio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=WIN_LENGTH,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=1  # Use amplitude instead of power
    ).to(device)
    
    # Create discriminator
    discriminator = FastGANDiscriminator(
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        mel_transform=mel_transform
    ).to(device)
    
    # Print model info
    total_params_g, trainable_params_g = count_parameters(model)
    model_size_g = get_model_size(model)
    
    total_params_d, trainable_params_d = count_parameters(discriminator)
    model_size_d = get_model_size(discriminator)

    print(f"\nModel info:")
    print(f"  Generator:")
    print(f"    Total parameters: {total_params_g:,}")
    print(f"    Trainable parameters: {trainable_params_g:,}")
    print(f"    Model size: {model_size_g:.2f} MB")
    print(f"  Discriminator:")
    print(f"    Total parameters: {total_params_d:,}")
    print(f"    Trainable parameters: {trainable_params_d:,}")
    print(f"    Model size: {model_size_d:.2f} MB")
    
    # Create loss function
    criterion = DecoderLoss(
        stft_loss_weight=0.5,
        mel_loss_weight=0.3,
        adv_loss_weight=0.0,  # Start with 0 and gradually increase
        feature_matching_weight=10.0
    ).to(device)
    
    # Define adversarial weight schedule - linear warmup
    def adv_weight_schedule(epoch):
        # Start with 0, linearly increase to 0.2 over 50 epochs, then constant
        if epoch < 50:
            return 0.2 * (epoch / 50)
        else:
            return 0.2
    
    # Separate optimizers for generator and discriminator
    optimizer_g = optim.Adam(model.parameters(), lr=0.0001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0004)  # Higher LR for discriminator
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, 
        mode='min', 
        factor=0.5, 
        patience=50
    )
    
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, 
        mode='min', 
        factor=0.5, 
        patience=50
    )
    
    # Training loop    
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", leave=False)
    for epoch in epoch_pbar:
        # Train
        g_loss, d_loss, mel_loss, stft_loss, adv_loss, fm_loss = train_epoch(
            model, discriminator, train_loader, criterion, 
            optimizer_g, optimizer_d, device, epoch, mel_transform,
            d_steps=1, g_steps=1, adv_weight_schedule=adv_weight_schedule
        )
        
        # Visualize during evaluation at certain intervals
        should_visualize = (epoch % visualization_interval == 0)
        val_g_loss, val_mel_loss, val_stft_loss, val_adv_loss, val_fm_loss = evaluate(
            model, discriminator, val_loader, criterion, device, epoch, mel_transform, 
            visualize=should_visualize
        )
        
        # Update learning rates based on validation loss
        scheduler_g.step(val_g_loss)
        scheduler_d.step(d_loss)  # Use discriminator training loss for its scheduler
        
        # Update epoch progress bar with loss information
        epoch_pbar.set_postfix({
            'g_loss': f'{g_loss:.4f}',
            'd_loss': f'{d_loss:.4f}',
            'val_g': f'{val_g_loss:.4f}',
            'adv_w': f'{criterion.adv_loss_weight:.3f}'
        })

        # Save best model
        if val_g_loss < best_val_loss:
            best_val_loss = val_g_loss
            torch.save({
                'generator': model.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch,
                'loss': val_g_loss
            }, 'best_gan_model.pth')
            print(f"  Saved best model with val loss: {val_g_loss:.4f}")
        
        # Also save regular checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'scheduler_d_state_dict': scheduler_d.state_dict(),
                'g_loss': g_loss,
                'd_loss': d_loss,
                'val_g_loss': val_g_loss,
                'best_val_loss': best_val_loss,
            }, f'checkpoints/gan_checkpoint_epoch_{epoch}.pth')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
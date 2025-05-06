import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torchaudio

from loss import DecoderLoss, AdversarialDecoderLoss
from discriminator import MultiScaleDiscriminator
from dataset_decoder import get_dataloader, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH

from utils_log import visualize_outputs, visualize_formant_and_range_params, visualize_expressive_params_with_waveform
from svs import SVS

# Create visuals folder if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('visuals', exist_ok=True)
os.makedirs('visuals/decoder', exist_ok=True)
os.makedirs('audio_samples', exist_ok=True)
os.makedirs('visuals/decoder/params', exist_ok=True)  # Folder for parameter visualization
os.makedirs('visuals/decoder/formants', exist_ok=True)  # Folder for formant visualization
os.makedirs('visuals/decoder/loss', exist_ok=True)  # New folder for loss visualization

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

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    
    fake_samples = fake_samples[:, :real_samples.shape[1]]

    # Random weight for interpolation
    alpha = torch.rand(real_samples.shape[0], 1).to(device)
    
    # Interpolate between real and fake samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    #print(real_samples.shape)
    #print(fake_samples.shape)
    #print(interpolates.shape)

    # Get discriminator output for interpolated samples
    disc_interpolates, _ = discriminator(interpolates)
    
    # Take the gradient of the outputs with respect to the interpolates
    gradients = torch.autograd.grad(
        outputs=disc_interpolates[0].sum(),
        inputs=interpolates,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Compute the gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def train_epoch_with_adversarial(model, discriminator, dataloader, criterion, 
                               optimizer_G, optimizer_D, device, epoch, mel_transform,
                               d_steps_per_g_step=2, gradient_penalty_weight=10.0):
    """Train for one epoch with adversarial loss"""
    model.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    total_mel_loss = 0
    total_stft_loss = 0
    total_adv_loss = 0
    total_fm_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
        # Move data to device
        mel = batch['mel'].to(device)  # This will be [B, T, n_mels]
        f0 = batch['f0'].to(device)
        phoneme_seq = batch['phone_seq_mel'].to(device)  # Get phoneme sequence
        singer_id = batch['singer_id'].to(device).squeeze(1)  # Remove extra dimension
        language_id = batch['language_id'].to(device).squeeze(1)  # Remove extra dimension
        
        # Extract original audio for comparison
        target_audio = extract_audio_from_dataset(batch, device)
        
        # Generate audio
        with torch.autograd.set_detect_anomaly(True):  # Help debug gradient issues
            wave, latent_mel, vocal_params = model(f0, phoneme_seq, singer_id, language_id)
            
            # Run the discriminator on real audio to get features for feature matching
            with torch.no_grad():
                disc_outputs_real, disc_features_real = discriminator(target_audio)
            
            # --------------------
            # Train Discriminator
            # --------------------
            for _ in range(d_steps_per_g_step):
                optimizer_D.zero_grad()
                
                # Get discriminator outputs for real audio again (with gradients this time)
                disc_outputs_real, _ = discriminator(target_audio)
                
                # Get discriminator outputs for fake audio
                disc_outputs_fake, _ = discriminator(wave.detach())  # Detach to avoid training generator
                
                # WGAN discriminator loss
                d_loss = criterion.discriminator_loss(disc_outputs_real, disc_outputs_fake)
                
                # Compute gradient penalty (WGAN-GP)
                grad_penalty = compute_gradient_penalty(discriminator, target_audio, wave.detach(), device)
                d_loss = d_loss + gradient_penalty_weight * grad_penalty
                
                # Backward and optimize
                d_loss.backward()
                optimizer_D.step()
                
                # Track loss
                total_d_loss += d_loss.item()
            
            # --------------------
            # Train Generator (SVS model)
            # --------------------
            optimizer_G.zero_grad()
            
            # Get updated discriminator outputs for fake audio (for generator loss)
            disc_outputs_fake, disc_features_fake = discriminator(wave)
            
            # Compute full generator loss (reconstruction + adversarial)
            g_loss, mel_loss, stft_loss, adv_loss, fm_loss, predicted_mel, is_unstable = criterion.forward_generator(
                wave, target_audio, mel_transform, 
                disc_outputs_fake, disc_features_real, disc_features_fake
            )
            
            # Handle training instability if detected
            if is_unstable:
                print(f"Warning: Training instability detected at batch {batch_idx}. Skipping generator update.")
                # Optional: could implement more sophisticated strategies here
            else:
                # Backward and optimize
                g_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer_G.step()
            
            # Track losses
            total_g_loss += g_loss.item()
            total_mel_loss += mel_loss.item()
            total_stft_loss += stft_loss.item()
            total_adv_loss += adv_loss.item()
            total_fm_loss += fm_loss.item()
        
        # Store discriminator loss in loss history for plotting
        criterion.loss_history['disc_loss'].append(d_loss.item())
        
        # Visualize parameters occasionally during training
        if batch_idx == 0 and epoch % 20 == 0:
            visualize_expressive_params_with_waveform(epoch, batch_idx, wave, vocal_params)
            visualize_formant_and_range_params(epoch, batch_idx, wave, vocal_params)
    
    # Compute average losses
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / (len(dataloader) * d_steps_per_g_step)
    avg_mel_loss = total_mel_loss / len(dataloader)
    avg_stft_loss = total_stft_loss / len(dataloader)
    avg_adv_loss = total_adv_loss / len(dataloader)
    avg_fm_loss = total_fm_loss / len(dataloader)
    
    # Plot losses
    criterion.plot_losses(filename=f'loss_plot')
    
    return avg_g_loss, avg_d_loss, avg_mel_loss, avg_stft_loss, avg_adv_loss, avg_fm_loss

def evaluate_with_adversarial(model, discriminator, dataloader, criterion, device, epoch, mel_transform, visualize=False):
    """Evaluate the model with adversarial components"""
    model.eval()
    discriminator.eval()
    
    total_g_loss = 0
    total_mel_loss = 0
    total_stft_loss = 0
    total_adv_loss = 0
    total_fm_loss = 0
    
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
            
            # Get discriminator outputs
            disc_outputs_real, disc_features_real = discriminator(target_audio)
            disc_outputs_fake, disc_features_fake = discriminator(wave)
            
            # Compute full generator loss
            g_loss, mel_loss, stft_loss, adv_loss, fm_loss, predicted_mel, _ = criterion.forward_generator(
                wave, target_audio, mel_transform, 
                disc_outputs_fake, disc_features_real, disc_features_fake
            )
            
            total_g_loss += g_loss.item()
            total_mel_loss += mel_loss.item()
            total_stft_loss += stft_loss.item()
            total_adv_loss += adv_loss.item()
            total_fm_loss += fm_loss.item()
            
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
    os.makedirs('visuals/decoder/params', exist_ok=True)
    os.makedirs('visuals/decoder/formants', exist_ok=True)
    os.makedirs('visuals/decoder/loss', exist_ok=True)  # New directory for loss visualizations
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load dataset
    batch_size = 16  # Smaller batch size for complex model
    num_epochs = 1000
    visualization_interval = 1  # Visualize every 5 epochs

    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=1,
        train_files=50,
        val_files=10,
        device=device,
        context_window_sec=2,  # 2-second window
        persistent_workers=True
    )
    
    # Get dataset parameters
    num_phonemes = len(train_dataset.phone_map)
    num_singers = len(train_dataset.singer_map)
    num_languages = len(train_dataset.language_map)
    
    # Create SVS model (generator)
    model = SVS(
        num_phonemes=num_phonemes,
        num_singers=num_singers,
        num_languages=num_languages,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        sample_rate=SAMPLE_RATE
    ).to(device)
    
    # Create discriminator
    discriminator = MultiScaleDiscriminator(
        scales=3,
        channels=32,
        use_spectral_norm=True
    ).to(device)
    
    # Print model info
    total_params_gen, trainable_params_gen = count_parameters(model)
    model_size_gen = get_model_size(model)
    total_params_disc, trainable_params_disc = count_parameters(discriminator)
    model_size_disc = get_model_size(discriminator)

    print(f"\nGenerator (SVS) info:")
    print(f"  Total parameters: {total_params_gen:,}")
    print(f"  Trainable parameters: {trainable_params_gen:,}")
    print(f"  Model size: {model_size_gen:.2f} MB")
    
    print(f"\nDiscriminator info:")
    print(f"  Total parameters: {total_params_disc:,}")
    print(f"  Trainable parameters: {trainable_params_disc:,}")
    print(f"  Model size: {model_size_disc:.2f} MB")
    
    # Create adversarial loss function with progressive training
    criterion = AdversarialDecoderLoss(
        stft_loss_weight=0.5,
        mel_loss_weight=0.3,
        adv_loss_weight=0.2,
        feature_matching_weight=0.1,
        progressive_steps=50000  # Gradually increase adversarial weight over steps
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
    
    # Separate optimizers for generator and discriminator
    optimizer_G = optim.Adam(model.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0004)  # Lower LR for discriminator
    
    # Learning rate scheduler for generator - reduce LR when loss plateaus
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, 
        mode='min', 
        factor=0.5, 
        patience=50,
        verbose=True
    )
    
    # Training loop    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train with adversarial loss
        train_g_loss, train_d_loss, train_mel_loss, train_stft_loss, train_adv_loss, train_fm_loss = train_epoch_with_adversarial(
            model=model,
            discriminator=discriminator,
            dataloader=train_loader,
            criterion=criterion,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            device=device,
            epoch=epoch,
            mel_transform=mel_transform,
            d_steps_per_g_step=2  # Train discriminator twice for each generator step
        )
        
        # Visualize during evaluation at certain intervals
        should_visualize = (epoch % visualization_interval == 0)
        val_g_loss, val_mel_loss, val_stft_loss, val_adv_loss, val_fm_loss = evaluate_with_adversarial(
            model=model,
            discriminator=discriminator,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            mel_transform=mel_transform,
            visualize=should_visualize
        )
        
        # Update learning rate based on validation loss
        scheduler_G.step(val_g_loss)
        
        # Print training information
        print(f"Epoch {epoch}:")
        print(f"  Train Generator Loss: {train_g_loss:.4f}")
        print(f"  Train Discriminator Loss: {train_d_loss:.4f}")
        print(f"  Train Component Losses: Mel: {train_mel_loss:.4f}, STFT: {train_stft_loss:.4f}, Adv: {train_adv_loss:.4f}, FM: {train_fm_loss:.4f}")
        print(f"  Val Generator Loss: {val_g_loss:.4f}")
        print(f"  Val Component Losses: Mel: {val_mel_loss:.4f}, STFT: {val_stft_loss:.4f}, Adv: {val_adv_loss:.4f}, FM: {val_fm_loss:.4f}")
        print(f"  Current Gen LR: {optimizer_G.param_groups[0]['lr']:.6f}")
        print(f"  Current Disc LR: {optimizer_D.param_groups[0]['lr']:.6f}")
        print(f"  Current Adv Weight: {criterion.get_adv_weight():.4f}")
        
        # Save best model
        if val_g_loss < best_val_loss:
            best_val_loss = val_g_loss
            # Save both generator and discriminator
            torch.save({
                'generator': model.state_dict(),
                'discriminator': discriminator.state_dict()
            }, 'best_adversarial_decoder_model.pth')
            print(f"  Saved best model with val loss: {val_g_loss:.4f}")
        
        # Also save regular checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'train_g_loss': train_g_loss,
                'train_d_loss': train_d_loss,
                'val_g_loss': val_g_loss,
                'best_val_loss': best_val_loss,
            }, f'checkpoints/adversarial_decoder_checkpoint_epoch_{epoch}.pth')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
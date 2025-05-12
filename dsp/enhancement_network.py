import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv1d(dim, dim, 1)
        self.key = nn.Conv1d(dim, dim, 1)
        self.value = nn.Conv1d(dim, dim, 1)
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight for attention
        
    def forward(self, x):
        # x shape: [B, C, T]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # [B, T, T]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)  # [B, C, T]
        
        # Residual connection with learnable weight
        return x + self.gamma * out


class EnhancementNetwork(nn.Module):
    def __init__(self, fft_size=1024, hop_length=240, hidden_size=128, condition_dim=256, 
                 specialization=None):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.specialization = specialization  # Can be 'low', 'mid', 'high', 'phase', 'magnitude', etc.

        # Conditioning projection for parameter predictor features with normalization
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim),  # Add normalization at input
            nn.Linear(condition_dim, hidden_size),
            nn.LayerNorm(hidden_size),    # Add normalization after projection
            nn.LeakyReLU(0.1)
        )
        
        # Simplified conditioning - single projection to modulation values
        self.condition_modulation = nn.Sequential(
            nn.Linear(hidden_size, self.fft_size//2 + 1),
            nn.Tanh()  # Constrain to [-1, 1]
        )
        
        # Conditioning strength - gradually increase during training
        self.condition_strength = nn.Parameter(torch.tensor(0.01))  # Start very small
        
        # Network to enhance magnitude spectrum with attention mechanism
        self.mag_enhance = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_size, 3, padding=1),
            nn.InstanceNorm1d(hidden_size),  # Normalization for better stability
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),  # Dilated convolution for wider receptive field
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, self.fft_size//2 + 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Phase correction network (predicts phase adjustments)
        self.phase_enhance = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_size//2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size//2, hidden_size//2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size//2, self.fft_size//2 + 1, 3, padding=1),
            nn.Tanh()  # Output range [-1, 1] for phase adjustments
        )
        
        # Cross-attention between magnitude and phase
        self.cross_attention = SelfAttentionBlock(self.fft_size//2 + 1)
        
        # Final projection layer to combine features
        self.final_proj = nn.Sequential(
            nn.Conv1d(self.fft_size//2 + 1, hidden_size, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, self.fft_size//2 + 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize expert with specialization if provided
        if specialization:
            self._initialize_specialization()
        
    def _initialize_specialization(self):
        """Initialize network weights based on specialization"""
        if self.specialization == 'low_freq':
            # Emphasize low frequency processing
            # Initialize filters to focus on lower frequency bands
            with torch.no_grad():
                for layer in self.mag_enhance:
                    if isinstance(layer, nn.Conv1d):
                        # Initialize weights to emphasize low frequencies
                        nn.init.normal_(layer.weight, 0.0, 0.02)
                        # Apply frequency-dependent bias (higher for low freqs)
                        if layer.bias is not None:
                            freq_bias = torch.linspace(0.1, -0.1, layer.bias.shape[0])
                            layer.bias.data.copy_(freq_bias)
                            
        elif self.specialization == 'high_freq':
            # Emphasize high frequency processing
            with torch.no_grad():
                for layer in self.mag_enhance:
                    if isinstance(layer, nn.Conv1d):
                        nn.init.normal_(layer.weight, 0.0, 0.02)
                        if layer.bias is not None:
                            freq_bias = torch.linspace(-0.1, 0.1, layer.bias.shape[0])
                            layer.bias.data.copy_(freq_bias)
        
        elif self.specialization == 'phase':
            # Emphasize phase correction
            with torch.no_grad():
                # Scale up the phase network's impact
                for layer in self.phase_enhance:
                    if isinstance(layer, nn.Conv1d):
                        layer.weight.data *= 1.5
                
                # Scale down the magnitude network's impact
                for layer in self.mag_enhance:
                    if isinstance(layer, nn.Conv1d):
                        layer.weight.data *= 0.7
                        
        elif self.specialization == 'magnitude':
            # Emphasize magnitude enhancement
            with torch.no_grad():
                # Scale up the magnitude network's impact
                for layer in self.mag_enhance:
                    if isinstance(layer, nn.Conv1d):
                        layer.weight.data *= 1.5
                
                # Scale down the phase network's impact
                for layer in self.phase_enhance:
                    if isinstance(layer, nn.Conv1d):
                        layer.weight.data *= 0.7
                        
        elif self.specialization == 'transient':
            # Optimize for transient sounds (attack, percussion)
            # Use higher learning rates for faster response
            with torch.no_grad():
                self.cross_attention.gamma.data.fill_(0.3)  # Increase attention impact
                
                # Initialize with wider kernels for temporal context
                for layer in self.mag_enhance:
                    if isinstance(layer, nn.Conv1d) and layer.dilation[0] > 1:
                        layer.weight.data *= 1.2
        
        elif self.specialization == 'sustained':
            # Optimize for sustained sounds (vocals, strings)
            with torch.no_grad():
                # Adjust attention to focus more on longer contexts
                self.cross_attention.gamma.data.fill_(0.15)  # More conservative attention
        
    def forward(self, x, condition=None):
        """
        Forward pass with optional conservative conditioning from parameter predictor.
        
        Args:
            x: Input complex spectrogram [B, F, T]
            condition: Conditioning features from parameter predictor [B, T, condition_dim]
                       (Hidden features providing phonetic and singer context)
        
        Returns:
            enhanced_complex: Enhanced complex spectrogram [B, F, T]
        """
        x_stft = x
        
        # Get magnitude and phase
        mag = torch.abs(x_stft)  # [B, F, T]
        phase = torch.angle(x_stft)  # [B, F, T]
        
        # Apply frequency band masking based on specialization
        if self.specialization in ['low_freq', 'mid_freq', 'high_freq']:
            F = mag.shape[1]  # Number of frequency bins
            
            if self.specialization == 'low_freq':
                # Focus on lower third of frequencies
                focus_range = (0, F // 3)
            elif self.specialization == 'mid_freq':
                # Focus on middle third of frequencies
                focus_range = (F // 3, 2 * F // 3)
            elif self.specialization == 'high_freq':
                # Focus on upper third of frequencies
                focus_range = (2 * F // 3, F)
                
            # Create a soft mask that emphasizes the focus range
            freq_mask = torch.ones_like(mag)
            for i in range(F):
                if i < focus_range[0] or i >= focus_range[1]:
                    freq_mask[:, i, :] *= 0.7  # Reduce influence outside focus range
        else:
            freq_mask = torch.ones_like(mag)
        
        # Apply conditioning if provided - now with a more conservative approach
        if condition is not None:
            # Apply LayerNorm before projection (within the Sequential module)
            condition_features = self.condition_projection(condition)  # [B, T, hidden_size]
            
            # Generate modulation with constrained range
            modulation = self.condition_modulation(condition_features)  # [B, T, F]
            
            # Reshape modulation to match magnitude shape [B, F, T]
            modulation = modulation.permute(0, 2, 1)
            
            # Apply very conservative additive conditioning with learnable strength
            # Use clipped strength parameter to ensure it stays positive but small
            strength = torch.clamp(self.condition_strength, 0.0, 0.3)
            
            # Apply as a residual connection - original mag + small modulation
            mag_delta = modulation * strength * 0.1  # Further reduce the effect
            mag = mag + mag_delta  # Simple additive conditioning
        
        # Process magnitude with enhancement network (with frequency mask)
        mag_features = self.mag_enhance(mag * freq_mask)
        enhanced_mag = mag_features * mag  # Apply as multiplicative scaling
        
        # Process phase with correction network
        phase_adj = self.phase_enhance(mag)  # Use magnitude as input for phase adjustment
        phase_adj = phase_adj / (torch.max(torch.abs(phase_adj)) + 1e-6)  # Normalize to [-1, 1]
        enhanced_phase = phase + phase_adj 
        
        # Apply cross-attention to refined features
        # This allows magnitude and phase processing to inform each other
        attended_mag = self.cross_attention(enhanced_mag)
        
        # Final magnitude refinement
        final_mag = self.final_proj(attended_mag) * enhanced_mag
        
        # Convert back to time domain
        enhanced_complex = torch.polar(final_mag, enhanced_phase)
        
        return enhanced_complex


class MoEEnhancementNetwork(nn.Module):
    """
    Mixture of Experts version of the EnhancementNetwork.
    
    This implementation supports both dense and sparse routing strategies:
    - Dense: All experts process the input, outputs combined by weighted sum
    - Sparse: Only top-k experts process the input (more efficient)
    
    Each expert can have a different specialization focus.
    """
    def __init__(self, 
                 num_experts=4, 
                 top_k=2,  # Number of experts to use per sample
                 use_sparse_routing=True,
                 fft_size=1024, 
                 hop_length=240, 
                 hidden_size=128, 
                 condition_dim=256,
                 expert_specializations=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Cannot use more experts than available
        self.use_sparse_routing = use_sparse_routing
        self.fft_size = fft_size
        
        # Expert specializations
        if expert_specializations is None:
            # Default specializations if none provided
            expert_specializations = [
                'low_freq', 'mid_freq', 'high_freq', 'phase',
                'magnitude', 'transient', 'sustained', None
            ][:num_experts]
            
            # If we need more experts than specializations, repeat or use None
            if len(expert_specializations) < num_experts:
                expert_specializations += [None] * (num_experts - len(expert_specializations))
        
        # Create multiple expert networks with different specializations
        self.experts = nn.ModuleList([
            EnhancementNetwork(
                fft_size, 
                hop_length, 
                hidden_size, 
                condition_dim,
                specialization=expert_specializations[i] if i < len(expert_specializations) else None
            )
            for i in range(num_experts)
        ])
        
        # Router network to determine which experts to use
        self.router = nn.Sequential(
            nn.Conv1d(fft_size//2 + 1, hidden_size, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1),  # Global pooling across time
            nn.Flatten(),
            nn.Linear(hidden_size, num_experts),
        )
        
        # Optional: Add noise to routing decisions during training for better load balancing
        self.router_noise_scale = 1.0  # Can be annealed during training
        
        # Router importance loss coefficient (for load balancing)
        self.importance_loss_coef = 0.01
        self.load_balancing_loss = 0.0
        
    def _compute_load_balancing_loss(self, router_probs):
        """
        Compute auxiliary load balancing loss for Mixture of Experts.
        
        This encourages uniform expert utilization to prevent expert collapse
        (where only a few experts get used consistently).
        
        Args:
            router_probs: Routing probabilities [batch_size, num_experts]
            
        Returns:
            load_balancing_loss: Scalar loss term
        """
        # Compute mean routing probability per expert (over the batch)
        # This indicates how much each expert is being used
        mean_prob_per_expert = router_probs.mean(dim=0)
        
        # Compute expert usage entropy: we want this to be high
        # (i.e., all experts used equally)
        expert_usage_entropy = -torch.sum(
            mean_prob_per_expert * torch.log(mean_prob_per_expert + 1e-10)
        )
        
        # We want to maximize entropy, so we minimize negative entropy
        entropy_loss = -expert_usage_entropy
        
        # Compute load balancing term: variance in expert usage
        # We want all experts to have approximately equal load
        # If all experts had equal usage, each would have p=1/num_experts
        desired_prob = 1.0 / self.num_experts
        load_variance = torch.sum((mean_prob_per_expert - desired_prob) ** 2)
        
        # Combine losses - both encourage balanced expert usage
        return entropy_loss + load_variance
        
    def forward(self, x, condition=None, return_aux_loss=False):
        """
        Forward pass that routes input through the appropriate experts.
        
        Args:
            x: Input complex spectrogram [B, F, T]
            condition: Conditioning features [B, T, condition_dim]
            return_aux_loss: Whether to return auxiliary load balancing loss
            
        Returns:
            enhanced_complex: Enhanced complex spectrogram [B, F, T]
            (optional) aux_loss: Auxiliary load balancing loss
        """
        batch_size = x.shape[0]
        
        # Get routing weights from the router
        mag = torch.abs(x)
        routing_logits = self.router(mag)
        
        # Add noise during training (optional)
        if self.training and self.router_noise_scale > 0:
            noise = torch.randn_like(routing_logits) * self.router_noise_scale
            routing_logits = routing_logits + noise
        
        # Store routing probabilities for auxiliary loss
        routing_probs = F.softmax(routing_logits, dim=1)
        
        if self.use_sparse_routing:
            # Sparse MoE: Use only top-k experts
            top_k_weights, top_k_indices = torch.topk(routing_logits, self.top_k, dim=1)
            top_k_weights = F.softmax(top_k_weights, dim=1)
            
            # Initialize output tensor
            enhanced_complex = torch.zeros_like(x)
            
            # Process each batch item separately
            for batch_idx in range(batch_size):
                # Get condition for this batch item if provided
                batch_condition = None
                if condition is not None:
                    batch_condition = condition[batch_idx:batch_idx+1]
                
                # Process through selected experts and combine
                batch_x = x[batch_idx:batch_idx+1]
                batch_output = torch.zeros_like(batch_x)
                
                for k in range(self.top_k):
                    expert_idx = top_k_indices[batch_idx, k].item()
                    expert_output = self.experts[expert_idx](batch_x, batch_condition)
                    batch_output += expert_output * top_k_weights[batch_idx, k]
                
                enhanced_complex[batch_idx:batch_idx+1] = batch_output
        else:
            # Dense MoE: Use all experts with soft routing
            routing_weights = F.softmax(routing_logits, dim=1)
            
            # Initialize output tensor
            enhanced_complex = torch.zeros_like(x)
            
            # Process input through all experts
            for expert_idx, expert in enumerate(self.experts):
                # Get expert outputs for all batch items
                expert_output = expert(x, condition)
                
                # Weight each sample's output by its router weight for this expert
                for batch_idx in range(batch_size):
                    enhanced_complex[batch_idx] += expert_output[batch_idx] * routing_weights[batch_idx, expert_idx]
        
        # Compute auxiliary load balancing loss if requested
        if return_aux_loss and self.training:
            self.load_balancing_loss = self._compute_load_balancing_loss(routing_probs)
            return enhanced_complex, self.load_balancing_loss
        
        return enhanced_complex
    
    def get_loss(self, primary_loss):
        """
        Add auxiliary load balancing loss to primary loss.
        
        This should be called after a forward pass with return_aux_loss=True.
        
        Args:
            primary_loss: The main loss function value (e.g., reconstruction loss)
            
        Returns:
            total_loss: Combined loss including load balancing
        """
        return primary_loss + self.importance_loss_coef * self.load_balancing_loss


# Example of creating the MoE network with specialized experts
def create_moe_enhancement_network(num_experts=6, top_k=2, use_sparse_routing=True):
    """
    Create a MoE Enhancement Network with specialized experts.
    
    Args:
        num_experts: Number of expert networks
        top_k: Number of experts to use per sample (for sparse routing)
        use_sparse_routing: Whether to use sparse routing (only use top-k experts)
        
    Returns:
        model: Initialized MoEEnhancementNetwork
    """
    # Define expert specializations
    specializations = [
        'low_freq',      # Focus on bass/low frequencies
        'mid_freq',      # Focus on midrange
        'high_freq',     # Focus on treble/high frequencies
        'phase',         # Focus on phase correction
        'magnitude',     # Focus on magnitude enhancement
        'transient',     # Focus on transient sounds (percussive)
        'sustained',     # Focus on sustained sounds (vocals, strings)
        None             # General expert with no specialization
    ][:num_experts]
    
    # Create the MoE network
    model = MoEEnhancementNetwork(
        num_experts=num_experts,
        top_k=top_k,
        use_sparse_routing=use_sparse_routing,
        fft_size=1024,
        hop_length=240,
        hidden_size=128,
        condition_dim=256,
        expert_specializations=specializations
    )
    
    return model


# Example of how to use the network in training
def example_training_step(model, optimizer, x_input, condition, target):
    """
    Example of a training step using the MoE network.
    
    Args:
        model: MoEEnhancementNetwork instance
        optimizer: PyTorch optimizer
        x_input: Input complex spectrogram [B, F, T]
        condition: Conditioning features [B, T, condition_dim]
        target: Target complex spectrogram [B, F, T]
        
    Returns:
        loss: Training loss value
    """
    optimizer.zero_grad()
    
    # Forward pass with auxiliary loss
    output, aux_loss = model(x_input, condition, return_aux_loss=True)
    
    # Compute primary loss (e.g., spectrogram reconstruction loss)
    # For example, using L1 loss on magnitude and phase
    mag_loss = F.l1_loss(torch.abs(output), torch.abs(target))
    phase_loss = F.l1_loss(torch.angle(output), torch.angle(target))
    primary_loss = mag_loss + 0.5 * phase_loss
    
    # Combine with auxiliary loss
    total_loss = model.get_loss(primary_loss)
    
    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
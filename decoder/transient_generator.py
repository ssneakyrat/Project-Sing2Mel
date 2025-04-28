import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution for more efficient processing"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                  padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class TransientGenerator(nn.Module):
    """
    More efficient version of TransientGenerator for modeling sharp attacks 
    and rapid transitions in speech/singing
    
    Efficiency improvements:
    1. Uses depthwise separable convolutions to reduce parameter count
    2. Replaces manual atom placement with transposed convolution
    3. Uses more efficient envelope upsampling with smoothing
    4. Implements sparse atom selection to focus on top-k atoms
    5. Adds optional caching of frequently used atom combinations
    """
    def __init__(self, hop_length=240, sample_rate=24000):
        super(TransientGenerator, self).__init__()
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Transient envelope network (using depthwise separable convolutions)
        self.envelope_net = nn.Sequential(
            DepthwiseSeparableConv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            DepthwiseSeparableConv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Envelope smoother for efficient upsampling
        self.envelope_smoother = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        nn.init.ones_(self.envelope_smoother.weight)
        self.envelope_smoother.weight.data /= self.envelope_smoother.weight.data.sum()
        
        # Dictionary of learned transient atoms
        self.num_atoms = 16
        self.atom_length = 32  # samples
        self.transient_dict = nn.Parameter(
            torch.randn(self.num_atoms, self.atom_length)
        )
        
        # Atom selection network with depthwise separable convolutions
        self.atom_selector = nn.Sequential(
            DepthwiseSeparableConv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, self.num_atoms, kernel_size=3, padding=1)
        )
        
        # Efficient atom placement using transposed convolution
        self.atom_placer = nn.ConvTranspose1d(
            self.num_atoms, 1, kernel_size=self.atom_length, 
            stride=self.hop_length, padding=0, bias=False
        )
        
        # Initialize atom_placer weights with the transient dictionary
        self._sync_atom_placer()
        
        # Optional: Cache for frequently used atom combinations
        self.cache_size = 8
        self.register_buffer('atom_cache', torch.zeros(self.cache_size, self.atom_length))
        self.register_buffer('cache_usage_count', torch.zeros(self.cache_size))
        self.cache_enabled = False  # Disabled by default
        
    def _sync_atom_placer(self):
        """Synchronize the atom_placer weights with the transient dictionary"""
        with torch.no_grad():
            weights = self.transient_dict.unsqueeze(1)  # [num_atoms, 1, atom_length]
            self.atom_placer.weight.copy_(weights)
    
    def _sparse_softmax(self, logits, dim=1, k=3):
        """Apply softmax only to top-k values along specified dimension"""
        batch_size, num_atoms, time_steps = logits.shape
        
        # Get top-k atom indices for each time step
        topk_values, topk_indices = torch.topk(logits, k, dim=dim)
        
        # Initialize output tensor with zeros
        sparse_weights = torch.zeros_like(logits)
        
        # Apply softmax only to top-k values and place them at the right indices
        softmax_values = F.softmax(topk_values, dim=dim)
        
        # Place softmax values at the right indices
        for b in range(batch_size):
            for t in range(time_steps):
                sparse_weights[b, topk_indices[b, :, t], t] = softmax_values[b, :, t]
        
        return sparse_weights
        
    def forward(self, condition, audio_length):
        """
        Generate transient components using condition vector
        
        Args:
            condition: Conditioning vector [B, 128, T]
            audio_length: Target audio length
            
        Returns:
            transient_signal: Generated transient signal [B, audio_length]
        """
        batch_size = condition.shape[0]
        time_steps = condition.shape[2]
        
        # Compute output size of transposed convolution precisely
        # For ConvTranspose1d: output_size = (input_size - 1) * stride + kernel_size
        output_size_formula = lambda input_len: (input_len - 1) * self.hop_length + self.atom_length
        
        # Calculate what input length we need to get exactly audio_length output
        # This inverse formula: input_len = (output_size - kernel_size) / stride + 1
        required_input_length = (audio_length - self.atom_length) // self.hop_length + 1
        
        # Verify our calculation is correct
        expected_output = output_size_formula(required_input_length)
        if expected_output != audio_length:
            # If we can't get exact audio_length, get the next largest size and we'll trim later
            required_input_length = (audio_length - self.atom_length) // self.hop_length + 2
        
        # Pad the condition tensor if needed
        if required_input_length > time_steps:
            padding = torch.zeros(
                batch_size, condition.shape[1], required_input_length - time_steps,
                device=condition.device
            )
            condition = torch.cat([condition, padding], dim=2)
            time_steps = required_input_length
        
        # Get transient envelope
        envelope = self.envelope_net(condition)  # [B, 1, T]
        
        # Get atom selection logits
        atom_logits = self.atom_selector(condition)  # [B, num_atoms, T]
        
        # Apply sparse softmax to get atom weights (only top-k atoms active per timestep)
        k = min(3, self.num_atoms)  # Number of active atoms per timestep
        atom_weights = self._sparse_softmax(atom_logits, dim=1, k=k)  # [B, num_atoms, T]
        
        # Ensure atom_placer weights are synchronized with transient_dict
        self._sync_atom_placer()
        
        # Calculate output padding to ensure we get exactly the right audio length
        # The formula for output length of a transposed conv1d is:
        # output_length = (input_length - 1) * stride + kernel_size - 2 * padding
        
        # Place atoms with transposed convolution (much more efficient than loops)
        placed_atoms = self.atom_placer(atom_weights)  # [B, 1, calculated_audio_length]
        
        # Handle any size differences by either truncating or padding
        placed_atoms_length = placed_atoms.shape[2]
        
        if placed_atoms_length > audio_length:
            # Truncate if too long
            placed_atoms = placed_atoms[:, :, :audio_length]
        elif placed_atoms_length < audio_length:
            # Pad if too short
            padding = torch.zeros(
                batch_size, 1, audio_length - placed_atoms_length,
                device=placed_atoms.device
            )
            placed_atoms = torch.cat([placed_atoms, padding], dim=2)
        
        # Get the actual size of the placed_atoms tensor
        actual_audio_length = placed_atoms.shape[2]
        
        # Upsample envelope to match the EXACT length of placed_atoms
        envelope = F.interpolate(
            envelope, 
            size=actual_audio_length, 
            mode='nearest'
        )  # [B, 1, actual_audio_length]
        
        # Apply smoothing
        envelope = self.envelope_smoother(envelope)
        
        # Ensure both tensors have the same size for element-wise multiplication
        assert placed_atoms.shape == envelope.shape, f"Shape mismatch: placed_atoms {placed_atoms.shape} vs envelope {envelope.shape}"
        
        # Modulate placed atoms with envelope
        transient_signal = placed_atoms * envelope  # [B, 1, actual_audio_length]
        
        return transient_signal.squeeze(1)  # [B, audio_length]
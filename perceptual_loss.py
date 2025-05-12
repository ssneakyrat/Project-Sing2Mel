import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class Wav2VecFeatureExtractor(nn.Module):
    """Feature extractor based on wav2vec 2.0 pretrained model"""
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None, layers=None):
        """
        Initialize the wav2vec 2.0 feature extractor
        
        Args:
            model_name (str): Pretrained model name from Hugging Face
            device (torch.device): Device to use for computation
            layers (list): List of layer indices to extract features from. 
                           If None, uses the final hidden state
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        
        # Freeze the model parameters to avoid training
        for param in self.model.parameters():
            param.requires_grad = False
            
        # By default, extract from the last 4 layers
        self.layers = layers if layers is not None else [-4, -3, -2, -1]
        
        # Target sample rate for wav2vec 2.0 (16 kHz)
        self.target_sample_rate = 16000
        
    def extract_features(self, audio, sample_rate):
        """
        Extract features from audio using wav2vec 2.0
        
        Args:
            audio (torch.Tensor): Audio waveform [B, T]
            sample_rate (int): Sample rate of the audio
        
        Returns:
            list: List of feature tensors from specified layers
        """
        batch_size = audio.shape[0]
        
        # Move to CPU for preprocessing
        audio_cpu = audio.detach().cpu()
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            # Use torchaudio for resampling
            import torchaudio.transforms as T
            resampler = T.Resample(sample_rate, self.target_sample_rate).to(audio.device)
            audio = resampler(audio)
        
        # Normalize audio to [-1, 1]
        audio = F.normalize(audio, p=float('inf'), dim=1)
        
        # Process in batches
        all_features = []
        
        with torch.no_grad():
            # Get attention mask (all 1s since we're processing full audio)
            attention_mask = torch.ones_like(audio, dtype=torch.long, device=audio.device)
            
            # Forward pass through wav2vec model
            outputs = self.model(
                audio,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Extract features from specified layers
            hidden_states = outputs.hidden_states
            
            # Collect features from requested layers
            features = []
            for layer_idx in self.layers:
                layer_features = hidden_states[layer_idx]
                features.append(layer_features)
        
        return features

class Wav2VecPerceptualLoss(nn.Module):
    """Perceptual loss using wav2vec 2.0 features"""
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None, 
                 layers=None, weights=None, sample_rate=16000):
        """
        Initialize wav2vec 2.0 perceptual loss
        
        Args:
            model_name (str): Pretrained model name from Hugging Face
            device (torch.device): Device to use for computation
            layers (list): List of layer indices to extract features from
            weights (list): Weights for each layer's contribution to the loss
            sample_rate (int): Sample rate of input audio
        """
        super().__init__()
        self.feature_extractor = Wav2VecFeatureExtractor(
            model_name=model_name,
            device=device,
            layers=layers
        )
        
        # Default weights give more importance to deeper layers
        if layers is not None and weights is None:
            # Initialize weights to give more importance to deeper layers
            num_layers = len(layers)
            self.weights = [1.0 + i/num_layers for i in range(num_layers)]
        else:
            self.weights = weights if weights is not None else [1.0, 1.0, 1.0, 1.0]
            
        # Normalize weights to sum to 1
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        self.sample_rate = sample_rate
        self.criterion = nn.L1Loss()
        
    def forward(self, predicted_audio, target_audio):
        """
        Compute perceptual loss between predicted and target audio
        
        Args:
            predicted_audio (torch.Tensor): Predicted audio [B, T]
            target_audio (torch.Tensor): Target audio [B, T]
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        # Ensure both audios have the same length
        min_len = min(predicted_audio.shape[1], target_audio.shape[1])
        predicted_audio = predicted_audio[:, :min_len]
        target_audio = target_audio[:, :min_len]
        
        # Extract features
        pred_features = self.feature_extractor.extract_features(
            predicted_audio, self.sample_rate
        )
        
        target_features = self.feature_extractor.extract_features(
            target_audio, self.sample_rate
        )
        
        # Compute loss across all selected layers
        total_loss = 0.0
        for i, (pred_feat, target_feat, weight) in enumerate(zip(pred_features, target_features, self.weights)):
            # Ensure same shape
            min_time = min(pred_feat.shape[1], target_feat.shape[1])
            pred_feat = pred_feat[:, :min_time, :]
            target_feat = target_feat[:, :min_time, :]
            
            # Compute L1 loss between features
            layer_loss = self.criterion(pred_feat, target_feat)
            total_loss += weight * layer_loss
            
        return total_loss

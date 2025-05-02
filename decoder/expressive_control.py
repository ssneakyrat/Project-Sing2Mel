import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExpressiveControl(nn.Module):
    """
    Predicts expressive control parameters for singing voice synthesis.
    Responsible only for parameter prediction, not signal processing.
    """
    def __init__(self, input_dim=256, sample_rate=24000):
        super().__init__()
        self.input_dim = input_dim
        self.sample_rate = sample_rate

        # Expression parameter predictors
        self.vibrato_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 3),  # [rate, depth, phase]
            nn.Sigmoid()
        )
        
        self.breathiness_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.tension_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.vocal_fry_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, conditioning):
        """
        Predict expressive control parameters
        
        Args:
            conditioning: Conditioning features [B, T, C]
            
        Returns:
            Dictionary of expressive parameters
        """
        # Compute all parameters in one step
        all_params = {}
        
        # Vibrato parameters
        vibrato_params = self.vibrato_predictor(conditioning)
        all_params['vibrato_rate'] = vibrato_params[:, :, 0]
        all_params['vibrato_depth'] = vibrato_params[:, :, 1]
        all_params['vibrato_phase'] = vibrato_params[:, :, 2]

        # Other expressive parameters
        all_params['breathiness'] = self.breathiness_predictor(conditioning)
        all_params['tension'] = self.tension_predictor(conditioning)
        all_params['vocal_fry'] = self.vocal_fry_predictor(conditioning)
        
        return all_params
    
    def _calculate_tension_derived_params(self, tension):
        """
        Calculate derived parameters from tension
        This method is kept for backward compatibility but will 
        output a warning as tension-derived parameters should be 
        calculated by the SignalProcessor
        
        Args:
            tension: Tension parameter [B, T, 1]
            
        Returns:
            Empty dictionary (placeholder)
        """
        print("WARNING: _calculate_tension_derived_params should be called from SignalProcessor")
        return {}
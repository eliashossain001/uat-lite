"""
Confidence-aware decision shaping

Implements confidence-based prediction with abstention mechanism.
Uses entropy-based confidence scoring and thresholding.

Confidence formula:
    C(p) = 1 - H(p) / log(K)

where:
    - H(p) = -Σ p_k log(p_k) is prediction entropy
    - K is number of classes
    - C(p) ∈ [0, 1], with 1 being maximally confident

Abstention rule:
    Abstain if C(p) < τ

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class ConfidenceDecision(nn.Module):
    """
    Confidence-aware decision shaping
    
    Provides:
    1. Entropy-based confidence scoring
    2. Selective prediction with abstention
    3. Confidence-adjusted logits
    
    Args:
        num_classes (int): Number of output classes (K)
        confidence_threshold (float): τ threshold for abstention (default: 0.7)
        temperature (float): Temperature scaling for calibration (default: 1.0)
    """
    
    def __init__(
        self,
        num_classes: int,
        confidence_threshold: float = 0.7,
        temperature: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        
        # Maximum entropy for K classes
        self.max_entropy = math.log(num_classes)
    
    def compute_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy of probability distribution
        
        H(p) = -Σ p_k log(p_k)
        
        Args:
            probabilities: [batch_size, num_classes] probability distribution
            
        Returns:
            entropy: [batch_size] entropy values
        """
        # Add small epsilon for numerical stability
        epsilon = 1e-10
        probabilities = torch.clamp(probabilities, min=epsilon, max=1.0)
        
        # Compute entropy: -Σ p log(p)
        entropy = -(probabilities * torch.log(probabilities)).sum(dim=-1)
        
        return entropy
    
    def compute_confidence(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized confidence score
        
        C(p) = 1 - H(p) / log(K)
        
        Args:
            probabilities: [batch_size, num_classes]
            
        Returns:
            confidence: [batch_size] confidence scores in [0, 1]
        """
        # Compute entropy
        entropy = self.compute_entropy(probabilities)
        
        # Normalize by maximum entropy
        normalized_entropy = entropy / self.max_entropy
        
        # Confidence is 1 - normalized entropy
        confidence = 1.0 - normalized_entropy
        
        return confidence
    
    def should_abstain(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        Determine which predictions should be abstained
        
        Args:
            confidence: [batch_size] confidence scores
            
        Returns:
            abstain_mask: [batch_size] Boolean tensor (True = abstain)
        """
        return confidence < self.confidence_threshold
    
    def apply_temperature_scaling(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply temperature scaling for calibration
        
        Args:
            logits: [batch_size, num_classes]
            temperature: Scaling temperature (uses self.temperature if None)
            
        Returns:
            scaled_logits: [batch_size, num_classes]
        """
        if temperature is None:
            temperature = self.temperature
        
        return logits / temperature
    
    def adjust_predictions_for_confidence(
        self,
        logits: torch.Tensor,
        confidence: torch.Tensor,
        adjustment_strength: float = 0.5
    ) -> torch.Tensor:
        """
        Adjust logits based on confidence (optional enhancement)
        
        High confidence → sharpen distribution
        Low confidence → flatten distribution
        
        Args:
            logits: [batch_size, num_classes]
            confidence: [batch_size] confidence scores
            adjustment_strength: How much to adjust (0 = no adjustment, 1 = full)
            
        Returns:
            adjusted_logits: [batch_size, num_classes]
        """
        # Compute adaptive temperature based on confidence
        # High confidence → low temperature (sharper)
        # Low confidence → high temperature (flatter)
        adaptive_temp = 1.0 + adjustment_strength * (1.0 - confidence.unsqueeze(-1))
        
        adjusted_logits = logits / adaptive_temp
        
        return adjusted_logits
    
    def forward(
        self,
        logits: torch.Tensor,
        return_all: bool = True,
        apply_temperature: bool = True,
        apply_confidence_adjustment: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass
        
        Args:
            logits: [batch_size, num_classes] model logits
            return_all: Whether to return all outputs
            apply_temperature: Whether to apply temperature scaling
            apply_confidence_adjustment: Whether to adjust predictions by confidence
            
        Returns:
            Dictionary containing:
                - logits: Original or adjusted logits [batch_size, num_classes]
                - probabilities: Softmax probabilities [batch_size, num_classes]
                - predictions: Predicted classes [batch_size]
                - confidence: Confidence scores [batch_size]
                - entropy: Entropy values [batch_size]
                - should_abstain: Abstention mask [batch_size]
                - max_probability: Maximum probability [batch_size]
        """
        batch_size = logits.shape[0]
        
        # Apply temperature scaling if requested
        if apply_temperature:
            logits = self.apply_temperature_scaling(logits)
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Compute confidence
        confidence = self.compute_confidence(probabilities)
        
        # Apply confidence-based adjustment if requested
        if apply_confidence_adjustment:
            logits = self.adjust_predictions_for_confidence(
                logits, confidence, adjustment_strength=0.5
            )
            probabilities = F.softmax(logits, dim=-1)
            # Recompute confidence after adjustment
            confidence = self.compute_confidence(probabilities)
        
        # Get predictions
        predictions = torch.argmax(probabilities, dim=-1)
        
        # Determine abstention
        abstain_mask = self.should_abstain(confidence)
        
        # Compute additional metrics
        entropy = self.compute_entropy(probabilities)
        max_probability, _ = torch.max(probabilities, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions,
            'confidence': confidence,
            'entropy': entropy,
            'should_abstain': abstain_mask,
            'max_probability': max_probability,
        }
        
        if not return_all:
            # Return only essential outputs
            result = {
                'probabilities': probabilities,
                'predictions': predictions,
                'confidence': confidence,
                'should_abstain': abstain_mask,
            }
        
        return result
    
    def selective_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        coverage_levels: list = [0.7, 0.8, 0.9, 1.0]
    ) -> Dict[str, Dict[float, float]]:
        """
        Compute accuracy at different coverage levels
        
        Coverage = fraction of samples NOT abstained
        
        Args:
            predictions: [N] predicted classes
            labels: [N] true labels
            confidence: [N] confidence scores
            coverage_levels: List of coverage thresholds
            
        Returns:
            Dictionary mapping coverage → accuracy
        """
        N = predictions.shape[0]
        
        # Sort by confidence (descending)
        sorted_indices = torch.argsort(confidence, descending=True)
        sorted_predictions = predictions[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        results = {}
        for coverage in coverage_levels:
            # Select top coverage% samples
            num_samples = int(N * coverage)
            if num_samples == 0:
                continue
            
            selected_preds = sorted_predictions[:num_samples]
            selected_labels = sorted_labels[:num_samples]
            
            # Compute accuracy on selected samples
            correct = (selected_preds == selected_labels).float().sum()
            accuracy = (correct / num_samples).item()
            
            results[coverage] = accuracy
        
        return results
    
    def compute_risk_coverage_curve(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        num_points: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute risk-coverage curve for AURC metric
        
        Args:
            predictions: [N] predicted classes
            labels: [N] true labels
            confidence: [N] confidence scores
            num_points: Number of points in curve
            
        Returns:
            coverage_points: [num_points] coverage values
            risk_points: [num_points] error rates at each coverage
        """
        N = predictions.shape[0]
        
        # Sort by confidence
        sorted_indices = torch.argsort(confidence, descending=True)
        sorted_predictions = predictions[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        coverage_points = []
        risk_points = []
        
        for i in range(1, num_points + 1):
            # Coverage fraction
            coverage = i / num_points
            num_samples = max(1, int(N * coverage))
            
            # Select samples
            selected_preds = sorted_predictions[:num_samples]
            selected_labels = sorted_labels[:num_samples]
            
            # Compute error rate (risk)
            errors = (selected_preds != selected_labels).float().sum()
            risk = (errors / num_samples).item()
            
            coverage_points.append(coverage)
            risk_points.append(risk)
        
        return torch.tensor(coverage_points), torch.tensor(risk_points)

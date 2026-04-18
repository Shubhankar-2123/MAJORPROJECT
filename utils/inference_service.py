"""
Inference Service for Sign Language Recognition
Unified inference handling for static and dynamic models
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for running model inferences."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize inference service.
        
        Args:
            device: PyTorch device to use for inference (cuda or cpu)
        """
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"Inference service initialized with device: {self.device}")
    
    def infer_static(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        label_encoder: Optional[Any] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run inference on static model.
        
        Args:
            model: PyTorch model
            features: Input features tensor
            label_encoder: Label encoder for decoding predictions
            threshold: Confidence threshold
            
        Returns:
            Dictionary with prediction, confidence, and metadata
        """
        try:
            model.eval()
            with torch.no_grad():
                if features.device != self.device:
                    features = features.to(self.device)
                
                output = model(features)
                probabilities = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probabilities, dim=1)
                
                confidence_val = float(confidence.item())
                pred_idx_val = int(pred_idx.item())
                
                # Decode label if encoder provided
                label = None
                if label_encoder is not None:
                    label = label_encoder.inverse_transform([pred_idx_val])[0]
                
                return {
                    "prediction": label or pred_idx_val,
                    "confidence": confidence_val,
                    "prediction_idx": pred_idx_val,
                    "meets_threshold": confidence_val >= threshold,
                    "raw_output": output.cpu().numpy()
                }
        except Exception as e:
            logger.error(f"Error in static inference: {e}")
            raise
    
    def infer_dynamic(
        self,
        model: torch.nn.Module,
        sequence: torch.Tensor,
        label_encoder: Optional[Any] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run inference on dynamic (LSTM) model.
        
        Args:
            model: PyTorch LSTM model
            sequence: Input sequence tensor of shape (batch, time, features)
            label_encoder: Label encoder for decoding predictions
            threshold: Confidence threshold
            
        Returns:
            Dictionary with prediction, confidence, and metadata
        """
        try:
            model.eval()
            with torch.no_grad():
                if sequence.device != self.device:
                    sequence = sequence.to(self.device)
                
                output = model(sequence)
                probabilities = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probabilities, dim=1)
                
                confidence_val = float(confidence.item())
                pred_idx_val = int(pred_idx.item())
                
                # Decode label if encoder provided
                label = None
                if label_encoder is not None:
                    label = label_encoder.inverse_transform([pred_idx_val])[0]
                
                return {
                    "prediction": label or pred_idx_val,
                    "confidence": confidence_val,
                    "prediction_idx": pred_idx_val,
                    "meets_threshold": confidence_val >= threshold,
                    "top_k": self._get_top_k(probabilities, label_encoder, k=3),
                    "raw_output": output.cpu().numpy()
                }
        except Exception as e:
            logger.error(f"Error in dynamic inference: {e}")
            raise
    
    def _get_top_k(
        self,
        probabilities: torch.Tensor,
        label_encoder: Optional[Any] = None,
        k: int = 3
    ) -> list:
        """Get top k predictions."""
        try:
            top_k_vals, top_k_idxs = torch.topk(probabilities, k=min(k, len(probabilities[0])), dim=1)
            
            results = []
            for val, idx in zip(top_k_vals[0].cpu().numpy(), top_k_idxs[0].cpu().numpy()):
                label = None
                if label_encoder is not None:
                    label = label_encoder.inverse_transform([int(idx)])[0]
                results.append({
                    "label": label or int(idx),
                    "confidence": float(val)
                })
            return results
        except Exception as e:
            logger.warning(f"Could not compute top-k: {e}")
            return []


# Global inference service instance
_inference_service: Optional[InferenceService] = None


def get_inference_service(device: Optional[torch.device] = None) -> InferenceService:
    """
    Get or create a global inference service instance.
    
    Args:
        device: PyTorch device to use (optional, detects GPU if available)
        
    Returns:
        InferenceService instance
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(device)
    return _inference_service


def reset_inference_service() -> None:
    """Reset the global inference service instance."""
    global _inference_service
    _inference_service = None

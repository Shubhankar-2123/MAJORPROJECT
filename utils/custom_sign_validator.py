"""
Custom Sign Video Validation Service
Optional ML-based validation for user-uploaded custom sign videos.
Verifies that uploaded video is actually the sign for the claimed word.
"""

import torch
import cv2
import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
import tempfile

# For inference
from inference_service import get_inference_service
from train_dynamic_new import DynamicLSTM


class CustomSignValidator:
    """
    Validates custom sign videos using ML inference.
    Optional enhancement - validates that uploaded video matches the word.
    """
    
    def __init__(
        self,
        inference_service=None,
        confidence_threshold: float = 0.75,
        enabled: bool = True
    ):
        """
        Initialize validator.
        
        Args:
            inference_service: Inference service for predictions
            confidence_threshold: Minimum confidence to accept (0-1)
            enabled: Whether validation is enabled (default: True, can be disabled)
        """
        self.inference_service = inference_service or get_inference_service()
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled
    
    def validate_video(
        self,
        video_path: str,
        expected_word: str,
        max_frames: int = 30
    ) -> Dict[str, Any]:
        """
        Validate a custom sign video.
        
        Args:
            video_path: Path to video file
            expected_word: The word that should be signed
            max_frames: Maximum frames to extract
            
        Returns:
            {
                'valid': bool,
                'prediction': str,
                'confidence': float,
                'message': str,
                'matches': bool (True if prediction == expected_word),
                'meets_threshold': bool
            }
        """
        if not self.enabled:
            return {
                'valid': True,
                'prediction': expected_word,
                'confidence': 1.0,
                'message': 'Validation disabled',
                'matches': True,
                'meets_threshold': True,
                'validation_enabled': False
            }
        
        # Extract frames from video
        frames = self._extract_frames(video_path, max_frames)
        
        if not frames or len(frames) == 0:
            return {
                'valid': False,
                'prediction': None,
                'confidence': 0.0,
                'message': 'Could not extract frames from video',
                'matches': False,
                'meets_threshold': False,
                'validation_enabled': True
            }
        
        # Run inference
        try:
            result = self.inference_service.predict_word(frames)
            
            if not result or result.get('status') != 'success':
                return {
                    'valid': False,
                    'prediction': None,
                    'confidence': 0.0,
                    'message': 'Inference failed',
                    'matches': False,
                    'meets_threshold': False,
                    'validation_enabled': True
                }
            
            predicted_word = result.get('prediction', '').lower().strip()
            confidence = result.get('confidence', 0.0)
            
            # Normalize word for comparison
            expected_normalized = self._normalize_word(expected_word)
            predicted_normalized = self._normalize_word(predicted_word)
            
            matches = expected_normalized == predicted_normalized
            meets_threshold = confidence >= self.confidence_threshold
            valid = matches and meets_threshold
            
            message = self._build_message(
                matches, 
                meets_threshold, 
                predicted_word, 
                confidence,
                expected_word
            )
            
            return {
                'valid': valid,
                'prediction': predicted_word,
                'confidence': round(confidence, 3),
                'message': message,
                'matches': matches,
                'meets_threshold': meets_threshold,
                'validation_enabled': True
            }
            
        except Exception as e:
            return {
                'valid': False,
                'prediction': None,
                'confidence': 0.0,
                'message': f'Validation error: {str(e)}',
                'matches': False,
                'meets_threshold': False,
                'validation_enabled': True
            }
    
    def _extract_frames(self, video_path: str, max_frames: int) -> np.ndarray:
        """Extract frames from video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return np.array([])
            
            # Calculate frame step to extract evenly
            step = max(1, total_frames // max_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % step == 0:
                    # Resize to match model input
                    frame = cv2.resize(frame, (640, 480))
                    frames.append(frame)
                
                frame_count += 1
                
                if len(frames) >= max_frames:
                    break
            
            cap.release()
            
            if len(frames) > 0:
                return np.array(frames)
            return np.array([])
            
        except Exception as e:
            print(f"Frame extraction error: {e}")
            return np.array([])
    
    def _normalize_word(self, word: str) -> str:
        """Normalize word for comparison."""
        return word.lower().strip().replace('_', ' ')
    
    def _build_message(
        self, 
        matches: bool, 
        meets_threshold: bool, 
        predicted: str,
        confidence: float,
        expected: str
    ) -> str:
        """Build user-friendly validation message."""
        if not matches:
            return (
                f"❌ Mismatch: You signed '{predicted}' (confidence: {confidence:.1%}) "
                f"but claimed it was '{expected}'. Please upload a video of '{expected}'."
            )
        
        if not meets_threshold:
            return (
                f"⚠️ Low confidence: Detected '{predicted}' at {confidence:.1%} confidence, "
                f"but we need at least {self.confidence_threshold:.1%}. "
                f"Please record a clearer version."
            )
        
        return (
            f"✅ Great! Detected '{predicted}' at {confidence:.1%} confidence. "
            f"This looks like a good recording of '{expected}'!"
        )
    
    def disable_validation(self):
        """Disable validation (optional fallback)."""
        self.enabled = False
    
    def enable_validation(self):
        """Enable validation."""
        self.enabled = True
    
    def set_threshold(self, threshold: float):
        """Set confidence threshold (0-1)."""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold


# Global validator instance
_validator_instance = None


def get_custom_sign_validator(
    confidence_threshold: float = 0.75,
    enabled: bool = True
) -> CustomSignValidator:
    """Get or create global validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = CustomSignValidator(
            confidence_threshold=confidence_threshold,
            enabled=enabled
        )
    return _validator_instance

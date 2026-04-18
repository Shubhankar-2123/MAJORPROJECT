"""
Preprocessing Service for Sign Language Recognition
Unified wrapper around preprocessing utilities
"""

import os
import cv2
import numpy as np
import torch
import tempfile
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Unified preprocessing service for both static and dynamic inputs."""
    
    @staticmethod
    def preprocess_image_for_inference(
        image_path: str,
        scaler_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        Preprocess a static image for inference.
        
        Args:
            image_path: Path to the image file
            scaler_path: Path to the static scaler pickle file
            device: PyTorch device for tensor
            
        Returns:
            numpy array of shape (126,) containing normalized hand keypoints
        """
        try:
            from preprocessing import _extract_static_hand_keypoints_from_bgr, PreprocessError
            import joblib
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Extract keypoints
            keypoints_126 = _extract_static_hand_keypoints_from_bgr(img)
            
            # Ensure correct length
            if len(keypoints_126) < 126:
                keypoints_126 = keypoints_126 + [0.0] * (126 - len(keypoints_126))
            elif len(keypoints_126) > 126:
                keypoints_126 = keypoints_126[:126]
            
            features = np.array(keypoints_126, dtype=np.float32)
            
            # Load and apply scaler if provided
            if scaler_path and os.path.exists(scaler_path):
                try:
                    scaler = joblib.load(scaler_path)
                    features = scaler.transform(features.reshape(1, -1)).flatten()
                except Exception as e:
                    logger.warning(f"Could not apply scaler: {e}, using unscaled features")
            
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    @staticmethod
    def preprocess_video_for_inference(
        video_path: str,
        max_frames: int = 30,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        Preprocess a video for dynamic model inference.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum frames to extract
            device: PyTorch device for tensor
            
        Returns:
            numpy array of shape (max_frames, 99) containing normalized pose keypoints
        """
        try:
            from preprocessing import (
                _extract_dynamic_pose_keypoints_sequence_from_video,
                normalize_keypoints_training_style
            )
            
            # Extract pose keypoints sequence
            seq = _extract_dynamic_pose_keypoints_sequence_from_video(video_path, max_frames)
            
            # Normalize using training-style normalization
            seq = normalize_keypoints_training_style(seq)
            
            return seq
            
        except Exception as e:
            logger.error(f"Error preprocessing video: {e}")
            raise
    
    @staticmethod
    def augment_sequence(
        keypoints: np.ndarray,
        n_augments: int = 5,
        noise_level: float = 0.01
    ) -> list:
        """
        Augment a keypoint sequence by adding noise.
        
        Args:
            keypoints: Original keypoint sequence of shape (T, features)
            n_augments: Number of augmented versions to create
            noise_level: Standard deviation of Gaussian noise to add
            
        Returns:
            List of augmented sequences
        """
        augmented = [keypoints.copy()]
        
        for _ in range(n_augments):
            noise = np.random.normal(0, noise_level, keypoints.shape)
            augmented_seq = keypoints + noise
            # Clip to valid range [0, 1]
            augmented_seq = np.clip(augmented_seq, 0, 1)
            augmented.append(augmented_seq)
        
        return augmented
    
    @staticmethod
    def preprocess_file_for_inference(
        file_path: str,
        is_image: bool = True,
        max_frames: int = 30,
        scaler_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        Preprocess a file (image or video) for inference.
        
        Args:
            file_path: Path to the file
            is_image: Whether the file is an image (True) or video (False)
            max_frames: For video files, maximum frames to extract
            scaler_path: For image files, path to the static scaler
            device: PyTorch device for output tensor
            
        Returns:
            Preprocessed features as numpy array
        """
        if is_image:
            return PreprocessingService.preprocess_image_for_inference(
                file_path, scaler_path, device
            )
        else:
            return PreprocessingService.preprocess_video_for_inference(
                file_path, max_frames, device
            )

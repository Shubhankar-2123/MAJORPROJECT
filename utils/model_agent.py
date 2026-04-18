"""
Intelligent Model Selection Agent for Sign Language Translation
Automatically determines which model (static/word/sentence) to use based on input analysis
"""

import cv2
import numpy as np
import logging
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    STATIC = "static"
    WORD = "word" 
    SENTENCE = "sentence"

@dataclass
class VideoAnalysis:
    """Analysis results for input video/image"""
    frame_count: int
    motion_score: float
    duration: float
    has_significant_motion: bool
    complexity_score: float
    recommended_model: ModelType
    confidence: float

@dataclass
class PredictionResult:
    """Unified prediction result with metadata"""
    prediction: str
    confidence: float
    model_used: str
    model_type: ModelType
    processing_time: float
    analysis: VideoAnalysis

class IntelligentModelAgent:
    """
    Smart agent that analyzes input and selects optimal model
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Thresholds for model selection
        self.STATIC_MAX_FRAMES = 5
        self.WORD_MAX_FRAMES = 15
        self.MOTION_THRESHOLD = 0.3
        self.COMPLEXITY_THRESHOLD = 0.5
        
        # Performance tracking
        self.prediction_history = []
        self.model_performance = {
            ModelType.STATIC: {"correct": 0, "total": 0},
            ModelType.WORD: {"correct": 0, "total": 0}, 
            ModelType.SENTENCE: {"correct": 0, "total": 0}
        }
    
    def analyze_video_input(self, file_path: str) -> VideoAnalysis:
        """
        Analyze video characteristics to determine optimal model
        """
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                # Fallback for images
                return VideoAnalysis(
                    frame_count=1,
                    motion_score=0.0,
                    duration=0.0,
                    has_significant_motion=False,
                    complexity_score=0.2,
                    recommended_model=ModelType.STATIC,
                    confidence=0.95
                )
            
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Analyze motion and complexity
            motion_score = self._calculate_motion_score(cap)
            complexity_score = self._calculate_complexity_score(frame_count, duration, motion_score)
            
            cap.release()
            
            # Determine recommended model
            recommended_model, confidence = self._select_model(
                frame_count, motion_score, complexity_score, duration
            )
            
            return VideoAnalysis(
                frame_count=frame_count,
                motion_score=motion_score,
                duration=duration,
                has_significant_motion=motion_score > self.MOTION_THRESHOLD,
                complexity_score=complexity_score,
                recommended_model=recommended_model,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            # Safe fallback
            return VideoAnalysis(
                frame_count=1,
                motion_score=0.0,
                duration=0.0,
                has_significant_motion=False,
                complexity_score=0.1,
                recommended_model=ModelType.STATIC,
                confidence=0.5
            )
    
    def _calculate_motion_score(self, cap: cv2.VideoCapture) -> float:
        """Calculate motion score by comparing consecutive frames"""
        try:
            ret, prev_frame = cap.read()
            if not ret:
                return 0.0
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            motion_diffs = []
            frame_count = 0
            max_frames_to_check = 30
            
            while frame_count < max_frames_to_check:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_diffs.append(np.mean(diff))
                
                prev_gray = curr_gray
                frame_count += 1
            
            if not motion_diffs:
                return 0.0
            
            # Normalize motion score (0-1)
            avg_motion = np.mean(motion_diffs)
            max_possible = 255.0  # Max pixel difference
            motion_score = min(avg_motion / max_possible, 1.0)
            
            return motion_score
            
        except Exception as e:
            self.logger.error(f"Motion calculation error: {e}")
            return 0.0
    
    def _calculate_complexity_score(self, frame_count: int, duration: float, motion_score: float) -> float:
        """Calculate complexity score based on multiple factors"""
        # Normalize frame count (0-1 scale, assuming max 60 frames)
        frame_factor = min(frame_count / 60.0, 1.0)
        
        # Normalize duration (0-1 scale, assuming max 5 seconds)
        duration_factor = min(duration / 5.0, 1.0)
        
        # Combine factors
        complexity = (frame_factor * 0.4 + duration_factor * 0.3 + motion_score * 0.3)
        
        return min(complexity, 1.0)
    
    def _select_model(self, frame_count: int, motion_score: float, complexity_score: float, duration: float) -> Tuple[ModelType, float]:
        """Select optimal model based on analysis"""
        confidence = 0.7
        
        # Static model: single frame or very short videos
        if frame_count <= self.STATIC_MAX_FRAMES or duration < 0.5:
            return ModelType.STATIC, 0.9
        
        # Word model: short videos with moderate motion
        if frame_count <= self.WORD_MAX_FRAMES and motion_score < 0.5:
            confidence = 0.8 if motion_score < 0.3 else 0.7
            return ModelType.WORD, confidence
        
        # Sentence model: longer videos or high complexity
        if frame_count > self.WORD_MAX_FRAMES or complexity_score > self.COMPLEXITY_THRESHOLD:
            confidence = 0.75 if complexity_score > 0.7 else 0.65
            return ModelType.SENTENCE, confidence
        
        # Default to word model
        return ModelType.WORD, 0.7
    
    def get_model_recommendation(self, file_path: str) -> Dict:
        """
        Public interface for model recommendation
        """
        analysis = self.analyze_video_input(file_path)
        
        return {
            "recommended_model": analysis.recommended_model.value,
            "confidence": analysis.confidence,
            "analysis": {
                "frame_count": analysis.frame_count,
                "motion_score": analysis.motion_score,
                "duration": analysis.duration,
                "complexity_score": analysis.complexity_score,
                "has_significant_motion": analysis.has_significant_motion
            }
        }
    
    def analyze_file_storage(self, file_storage) -> VideoAnalysis:
        """
        Analyze uploaded file (FileStorage object) and return recommendation
        """
        # Save to temp file for analysis
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            file_storage.save(tmp_path)
            analysis = self.analyze_video_input(tmp_path)
            return analysis
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

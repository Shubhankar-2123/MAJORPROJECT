"""
Real-time webcam processing for live sign language translation
"""

import cv2
import numpy as np
import threading
import queue
import time
import tempfile
import os
from typing import Callable, Optional

class LiveCameraProcessor:
    def __init__(self, prediction_callback: Callable):
        self.prediction_callback = prediction_callback
        self.frame_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.processing_thread = None
        
        # Processing parameters
        self.BUFFER_SIZE = 30  # frames to collect before prediction
        self.PREDICTION_INTERVAL = 2.0  # seconds between predictions
        
    def start_processing(self):
        """Start live camera processing"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop live camera processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        if not self.frame_queue.full():
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Skip if queue is full
    
    def _process_frames(self):
        """Background thread for frame processing"""
        frame_buffer = []
        last_prediction_time = 0
        
        while self.is_running:
            try:
                # Collect frames
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    frame_buffer.append(frame.copy())
                
                # Process when buffer is full or enough time has passed
                current_time = time.time()
                if (len(frame_buffer) >= self.BUFFER_SIZE or 
                    (len(frame_buffer) > 10 and 
                     current_time - last_prediction_time > self.PREDICTION_INTERVAL)):
                    
                    # Create temporary video from frames
                    video_path = self._frames_to_video(frame_buffer)
                    
                    # Make prediction
                    if video_path:
                        try:
                            self.prediction_callback(video_path)
                        except Exception as e:
                            print(f"Prediction callback error: {e}")
                        finally:
                            # Clean up
                            try:
                                os.unlink(video_path)
                            except:
                                pass
                    
                    # Reset buffer
                    frame_buffer = []
                    last_prediction_time = current_time
                
                time.sleep(0.1)  # Prevent excessive CPU usage
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                time.sleep(0.5)
    
    def _frames_to_video(self, frames: list) -> Optional[str]:
        """Convert frame buffer to temporary video file"""
        if not frames:
            return None
            
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tmp_path = tmp_file.name
            tmp_file.close()
            
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, 30.0, (width, height))
            
            if not out.isOpened():
                os.unlink(tmp_path)
                return None
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            return tmp_path
                
        except Exception as e:
            print(f"Video creation error: {e}")
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            return None

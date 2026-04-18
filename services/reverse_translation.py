"""
Reverse translation: Enhanced text to sign language service
Works with existing data structure
"""

import os
from typing import Dict, List, Optional
from rapidfuzz import process, fuzz

class ReverseTranslationService:
    def __init__(self, video_base_dirs: List[str] = None):
        """
        Initialize with base directories to scan for sign videos
        """
        if video_base_dirs is None:
            # Default to common data directories
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            video_base_dirs = [
                os.path.join(project_root, "data", "dynamic"),
                os.path.join(project_root, "data", "Frames_Word_Level_2"),
                os.path.join(project_root, "flask_app", "static", "videos"),
            ]
        
        self.video_base_dirs = [d for d in video_base_dirs if os.path.exists(d)]
        self.sign_index = self._build_sign_index()
    
    def _build_sign_index(self) -> Dict[str, str]:
        """Build index of available sign videos"""
        index = {}
        
        for base_dir in self.video_base_dirs:
            if not os.path.exists(base_dir):
                continue
                
            # Scan for video files
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv')):
                        # Extract sign name from filename or folder
                        sign_name = os.path.splitext(file)[0].lower()
                        folder_name = os.path.basename(root).lower()
                        
                        video_path = os.path.join(root, file)
                        
                        # Index by filename
                        if sign_name not in index:
                            index[sign_name] = video_path
                        
                        # Also index by folder name if different
                        if folder_name != sign_name and folder_name not in index:
                            index[folder_name] = video_path
        
        return index
    
    def text_to_signs(self, text: str) -> List[Dict]:
        """
        Convert text to sequence of sign videos
        """
        words = text.lower().split()
        sign_sequence = []
        
        for word in words:
            # Normalize word
            word_clean = word.strip('.,!?;:')
            
            # Direct match
            if word_clean in self.sign_index:
                sign_sequence.append({
                    'word': word,
                    'video_path': self.sign_index[word_clean],
                    'match_type': 'exact',
                    'confidence': 1.0
                })
            else:
                # Fuzzy matching
                matches = process.extract(
                    word_clean, 
                    self.sign_index.keys(), 
                    scorer=fuzz.ratio,
                    limit=1
                )
                
                if matches and matches[0][1] > 70:  # 70% similarity threshold
                    matched_word = matches[0][0]
                    confidence = matches[0][1] / 100.0
                    
                    sign_sequence.append({
                        'word': word,
                        'matched_word': matched_word,
                        'video_path': self.sign_index[matched_word],
                        'match_type': 'fuzzy',
                        'confidence': confidence
                    })
                else:
                    # Fallback to finger spelling
                    finger_spelling = self._get_finger_spelling(word_clean)
                    sign_sequence.extend(finger_spelling)
        
        return sign_sequence
    
    def _get_finger_spelling(self, word: str) -> List[Dict]:
        """Convert word to finger spelling sequence"""
        finger_signs = []
        
        for char in word:
            if char.isalpha():
                # Look for letter sign videos (try different naming patterns)
                letter_key = char.lower()
                letter_key_alt = f"letter_{char.lower()}"
                
                if letter_key in self.sign_index:
                    finger_signs.append({
                        'word': char,
                        'video_path': self.sign_index[letter_key],
                        'match_type': 'finger_spelling',
                        'confidence': 1.0
                    })
                elif letter_key_alt in self.sign_index:
                    finger_signs.append({
                        'word': char,
                        'video_path': self.sign_index[letter_key_alt],
                        'match_type': 'finger_spelling',
                        'confidence': 1.0
                    })
        
        return finger_signs
    
    def create_sign_playlist(self, text: str) -> Dict:
        """
        Create a playlist of sign videos for given text
        """
        sign_sequence = self.text_to_signs(text)
        
        playlist = {
            'original_text': text,
            'total_signs': len(sign_sequence),
            'estimated_duration': len(sign_sequence) * 2.5,  # ~2.5 seconds per sign
            'signs': sign_sequence,
            'playback_instructions': {
                'pause_between_signs': 0.5,
                'repeat_count': 1,
                'speed_multiplier': 1.0
            }
        }
        
        return playlist
    
    def get_available_signs(self) -> List[str]:
        """Get list of all available sign names"""
        return sorted(list(self.sign_index.keys()))

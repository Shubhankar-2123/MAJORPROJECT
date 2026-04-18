"""
Custom Signs Storage Utility
Provides storage management for user-uploaded custom sign videos.
Completely isolated from default dataset and ML models.
"""

import os
import re
from typing import Optional, Dict
from werkzeug.utils import secure_filename


class CustomSignStorage:
    """Manages custom sign video and image file storage."""
    
    # File type configurations
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
    
    def __init__(self, base_uploads_dir: str):
        """
        Initialize custom sign storage.
        
        Args:
            base_uploads_dir: Base uploads directory (e.g., "uploads")
        """
        self.base_uploads_dir = base_uploads_dir
        self.custom_signs_dir = os.path.join(base_uploads_dir, "custom_signs")
        
    def get_user_directory(self, user_id: int, category: str = "words") -> str:
        """Get the custom signs directory for a specific user and category."""
        user_dir = os.path.join(self.custom_signs_dir, f"user_{user_id}", category)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    def sanitize_word(self, word: str) -> str:
        """
        Sanitize a word for safe filename usage.
        Converts to lowercase, replaces spaces with underscores, removes special chars.
        """
        # Convert to lowercase and strip
        word = word.lower().strip()
        # Replace spaces with underscores
        word = word.replace(" ", "_")
        # Remove any characters that aren't alphanumeric, underscore, or hyphen
        word = re.sub(r'[^a-z0-9_-]', '', word)
        # Collapse multiple underscores
        word = re.sub(r'_+', '_', word)
        # Remove leading/trailing underscores
        word = word.strip("_")
        return word
    
    def get_video_filename(self, word: str) -> str:
        """Get the standardized filename for a word's video."""
        sanitized = self.sanitize_word(word)
        return f"{sanitized}.mp4"
    
    def get_image_filename(self, word: str, extension: str = ".jpg") -> str:
        """Get the standardized filename for a word's image."""
        sanitized = self.sanitize_word(word)
        ext = extension.lower() if extension.startswith('.') else f".{extension.lower()}"
        return f"{sanitized}{ext}"
    
    def get_file_path(self, user_id: int, word: str, category: str, file_type: str = "video") -> str:
        """
        Get the full absolute path for a custom sign file (video or image).
        
        Args:
            user_id: User ID
            word: Word/phrase
            category: Category (words, sentences, letters, numbers)
            file_type: 'video' or 'image'
            
        Returns:
            Absolute path to the file
        """
        user_dir = self.get_user_directory(user_id, category)
        if file_type == "image":
            filename = self.get_image_filename(word)
        else:
            filename = self.get_video_filename(word)
        return os.path.join(user_dir, filename)
    
    def get_video_path(self, user_id: int, word: str, category: str = "words") -> str:
        """
        Get the full absolute path for a custom sign video.
        
        Args:
            user_id: User ID
            word: Word/phrase
            category: Category (words, sentences, letters, numbers)
            
        Returns:
            Absolute path to the video file
        """
        return self.get_file_path(user_id, word, category, "video")
    
    def get_image_path(self, user_id: int, word: str, category: str = "letters") -> str:
        """
        Get the full absolute path for a custom sign image.
        
        Args:
            user_id: User ID
            word: Word/phrase
            category: Category (letters, numbers)
            
        Returns:
            Absolute path to the image file
        """
        return self.get_file_path(user_id, word, category, "image")
    
    def get_relative_path(self, user_id: int, word: str, category: str, file_type: str = "video") -> str:
        """
        Get the relative path (for database storage).
        
        Returns:
            Relative path like "custom_signs/user_3/words/hello.mp4"
        """
        if file_type == "image":
            filename = self.get_image_filename(word)
        else:
            filename = self.get_video_filename(word)
        return f"custom_signs/user_{user_id}/{category}/{filename}"
    
    def save_video(self, file_storage, user_id: int, word: str, category: str = "words") -> str:
        """
        Save an uploaded video file.
        
        Args:
            file_storage: Flask FileStorage object
            user_id: User ID
            word: Word/phrase
            category: Category (words, sentences, etc.)
            
        Returns:
            Relative path to saved file
        """
        video_path = self.get_video_path(user_id, word, category)
        file_storage.save(video_path)
        return self.get_relative_path(user_id, word, category, "video")
    
    def save_image(self, file_storage, user_id: int, word: str, category: str = "letters") -> str:
        """
        Save an uploaded image file.
        
        Args:
            file_storage: Flask FileStorage object
            user_id: User ID
            word: Word/phrase
            category: Category (letters, numbers)
            
        Returns:
            Relative path to saved file
        """
        # Get extension from uploaded filename
        ext = os.path.splitext(file_storage.filename)[1].lower()
        if not ext or ext not in self.IMAGE_EXTENSIONS:
            ext = '.jpg'
        
        image_path = self.get_image_path(user_id, word, category)
        file_storage.save(image_path)
        return self.get_relative_path(user_id, word, category, "image")
    
    def delete_video(self, user_id: int, word: str, category: str = "words") -> bool:
        """
        Delete a custom sign video file.
        
        Args:
            user_id: User ID
            word: Word/phrase
            category: Category
            
        Returns:
            True if deleted, False if file didn't exist
        """
        video_path = self.get_video_path(user_id, word, category)
        if os.path.exists(video_path):
            os.remove(video_path)
            return True
        return False
    
    def delete_image(self, user_id: int, word: str, category: str = "letters") -> bool:
        """
        Delete a custom sign image file.
        
        Args:
            user_id: User ID
            word: Word/phrase
            category: Category
            
        Returns:
            True if deleted, False if file didn't exist
        """
        image_path = self.get_image_path(user_id, word, category)
        if os.path.exists(image_path):
            os.remove(image_path)
            return True
        return False
    
    def delete_by_path(self, relative_path: str) -> bool:
        """
        Delete a file by its relative path.
        
        Args:
            relative_path: Relative path from database
            
        Returns:
            True if deleted, False if file didn't exist
        """
        abs_path = os.path.join(self.base_uploads_dir, relative_path)
        if os.path.exists(abs_path):
            os.remove(abs_path)
            return True
        return False
    
    def video_exists(self, user_id: int, word: str, category: str = "words") -> bool:
        """Check if a custom video exists."""
        video_path = self.get_video_path(user_id, word, category)
        return os.path.exists(video_path)
    
    def image_exists(self, user_id: int, word: str, category: str = "letters") -> bool:
        """Check if a custom image exists."""
        image_path = self.get_image_path(user_id, word, category)
        return os.path.exists(image_path)
    
    def validate_video_file(self, filename: str, size: int = 0) -> tuple:
        """
        Validate that uploaded file is a valid video format and size.
        
        Args:
            filename: Original filename
            size: File size in bytes
            
        Returns:
            (is_valid, error_message)
        """
        ext = os.path.splitext(filename.lower())[1]
        if ext not in self.VIDEO_EXTENSIONS:
            return False, f"Invalid video format. Allowed: {', '.join(self.VIDEO_EXTENSIONS)}"
        if size > self.MAX_VIDEO_SIZE:
            return False, f"Video too large. Maximum: {self.MAX_VIDEO_SIZE // (1024*1024)} MB"
        return True, None
    
    def validate_image_file(self, filename: str, size: int = 0) -> tuple:
        """
        Validate that uploaded file is a valid image format and size.
        
        Args:
            filename: Original filename
            size: File size in bytes
            
        Returns:
            (is_valid, error_message)
        """
        ext = os.path.splitext(filename.lower())[1]
        if ext not in self.IMAGE_EXTENSIONS:
            return False, f"Invalid image format. Allowed: {', '.join(self.IMAGE_EXTENSIONS)}"
        if size > self.MAX_IMAGE_SIZE:
            return False, f"Image too large. Maximum: {self.MAX_IMAGE_SIZE // (1024*1024)} MB"
        return True, None

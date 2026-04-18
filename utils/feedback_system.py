"""
User feedback collection and model retraining pipeline
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from dataclasses import dataclass, asdict

@dataclass
class FeedbackEntry:
    timestamp: str
    input_file: str
    predicted_label: str
    actual_label: str
    confidence: float
    model_used: str
    user_id: Optional[str] = None

class FeedbackSystem:
    def __init__(self, db_path: str = "data/feedback/feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for feedback storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_file TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                actual_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_used TEXT NOT NULL,
                user_id TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_feedback(self, feedback: FeedbackEntry) -> bool:
        """Store user feedback in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback 
                (timestamp, input_file, predicted_label, actual_label, 
                 confidence, model_used, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.timestamp,
                feedback.input_file,
                feedback.predicted_label,
                feedback.actual_label,
                feedback.confidence,
                feedback.model_used,
                feedback.user_id
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Feedback storage error: {e}")
            return False
    
    def get_retraining_candidates(self, min_samples: int = 10) -> Dict[str, List[FeedbackEntry]]:
        """
        Get feedback entries ready for retraining
        Groups by actual_label with minimum sample requirements
        """
        if not PANDAS_AVAILABLE:
            return {}
            
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get unprocessed feedback
            df = pd.read_sql_query('''
                SELECT * FROM feedback 
                WHERE processed = FALSE 
                AND predicted_label != actual_label
            ''', conn)
            
            if df.empty:
                return {}
            
            # Group by actual label
            grouped = df.groupby('actual_label')
            candidates = {}
            
            for label, group in grouped:
                if len(group) >= min_samples:
                    entries = [
                        FeedbackEntry(**row.to_dict()) 
                        for _, row in group.iterrows()
                    ]
                    candidates[label] = entries
            
            return candidates
        finally:
            conn.close()
    
    def mark_processed(self, feedback_ids: List[int]):
        """Mark feedback entries as processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join(['?' for _ in feedback_ids])
        cursor.execute(f'''
            UPDATE feedback 
            SET processed = TRUE 
            WHERE id IN ({placeholders})
        ''', feedback_ids)
        
        conn.commit()
        conn.close()
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics for monitoring"""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        try:
            # Total feedback count
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM feedback')
            stats['total_feedback'] = cursor.fetchone()[0]
            
            if PANDAS_AVAILABLE:
                # Accuracy by model
                df = pd.read_sql_query('''
                    SELECT model_used, 
                           COUNT(*) as total,
                           SUM(CASE WHEN predicted_label = actual_label THEN 1 ELSE 0 END) as correct
                    FROM feedback 
                    GROUP BY model_used
                ''', conn)
                
                stats['model_accuracy'] = {}
                for _, row in df.iterrows():
                    accuracy = row['correct'] / row['total'] if row['total'] > 0 else 0
                    stats['model_accuracy'][row['model_used']] = {
                        'accuracy': accuracy,
                        'total_samples': row['total']
                    }
            else:
                stats['model_accuracy'] = {}
        finally:
            conn.close()
        
        return stats

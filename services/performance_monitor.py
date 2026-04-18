"""
System performance monitoring and tracking
"""

import time
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class PerformanceMetric:
    timestamp: str
    model_type: str
    processing_time: float
    confidence: float
    prediction: str
    success: bool

class PerformanceMonitor:
    def __init__(self, db_path: str = "data/performance/performance.db"):
        self.db_path = db_path
        self.init_database()
        self.metrics_buffer = []
    
    def init_database(self):
        """Initialize SQLite database for performance tracking"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_type TEXT NOT NULL,
                processing_time REAL NOT NULL,
                confidence REAL NOT NULL,
                prediction TEXT,
                success BOOLEAN NOT NULL
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_model_type ON performance_metrics(model_type)
        ''')
        
        conn.commit()
        conn.close()
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        self.metrics_buffer.append(metric)
        
        # Batch insert every 10 metrics
        if len(self.metrics_buffer) >= 10:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for metric in self.metrics_buffer:
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, model_type, processing_time, confidence, prediction, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp,
                    metric.model_type,
                    metric.processing_time,
                    metric.confidence,
                    metric.prediction,
                    metric.success
                ))
            
            conn.commit()
            self.metrics_buffer = []
        except Exception as e:
            print(f"Performance metric storage error: {e}")
        finally:
            conn.close()
    
    def get_performance_stats(self, days: int = 7) -> Dict:
        """Get performance statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        try:
            # Overall stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    AVG(processing_time) as avg_time,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                FROM performance_metrics
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
            ''', (days,))
            
            row = cursor.fetchone()
            if row:
                stats['overall'] = {
                    'total_predictions': row[0],
                    'avg_processing_time': row[1] or 0.0,
                    'avg_confidence': row[2] or 0.0,
                    'success_rate': (row[3] / row[0]) if row[0] > 0 else 0.0
                }
            
            # Stats by model type
            cursor.execute('''
                SELECT 
                    model_type,
                    COUNT(*) as total,
                    AVG(processing_time) as avg_time,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                FROM performance_metrics
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY model_type
            ''', (days,))
            
            stats['by_model'] = {}
            for row in cursor.fetchall():
                stats['by_model'][row[0]] = {
                    'total': row[1],
                    'avg_processing_time': row[2] or 0.0,
                    'avg_confidence': row[3] or 0.0,
                    'success_rate': (row[4] / row[1]) if row[1] > 0 else 0.0
                }
            
        finally:
            conn.close()
        
        # Flush buffer before returning
        self._flush_buffer()
        
        return stats
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict]:
        """Get recent performance metrics"""
        self._flush_buffer()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, model_type, processing_time, confidence, prediction, success
            FROM performance_metrics
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        metrics = []
        for row in cursor.fetchall():
            metrics.append({
                'timestamp': row[0],
                'model_type': row[1],
                'processing_time': row[2],
                'confidence': row[3],
                'prediction': row[4],
                'success': bool(row[5])
            })
        
        conn.close()
        return metrics

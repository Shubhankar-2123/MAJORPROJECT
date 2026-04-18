"""
Performance Monitor for Sign Language Recognition System
Tracks inference times, model accuracy, throughput, and system metrics
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "metadata": self.metadata
        }


class PerformanceMonitor:
    """Monitor and track system performance metrics."""
    
    def __init__(self, max_history: int = 1000, db_path: Optional[str] = None):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
            db_path: Optional path to SQLite database for persistent storage
        """
        self.max_history = max_history
        self.db_path = db_path
        self.metrics: List[PerformanceMetric] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize database if path provided
        if db_path:
            self._init_db()
    
    def _init_db(self) -> None:
        """Initialize or ensure database exists."""
        if not self.db_path:
            return
        
        import os
        import sqlite3
        
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    metadata TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to initialize performance database: {e}")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric (e.g., "inference_time")
            value: Metric value
            unit: Unit of measurement
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        
        # Keep only recent metrics to avoid memory issues
        if len(self.metrics) > self.max_history:
            self.metrics = self.metrics[-self.max_history:]
        
        self.logger.debug(f"Recorded metric: {metric_name}={value}{unit}")
    
    def get_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary with min, max, mean, median, stdev
        """
        values = [
            m.value for m in self.metrics
            if m.metric_name == metric_name
        ]
        
        if not values:
            return None
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def get_recent_metrics(
        self,
        metric_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent metrics.
        
        Args:
            metric_name: Filter by metric name (optional)
            limit: Number of recent metrics to return
            
        Returns:
            List of recent metric dictionaries
        """
        filtered = self.metrics
        if metric_name:
            filtered = [m for m in filtered if m.metric_name == metric_name]
        
        return [m.to_dict() for m in filtered[-limit:]]
    
    def clear(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        if not self.metrics:
            return {"status": "no_metrics"}
        
        # Group metrics by name
        metric_names = set(m.metric_name for m in self.metrics)
        
        report = {
            "total_metrics": len(self.metrics),
            "metric_types": len(metric_names),
            "metrics": {}
        }
        
        for name in sorted(metric_names):
            stats = self.get_statistics(name)
            if stats:
                report["metrics"][name] = stats
        
        return report


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def record_performance(
    metric_name: str,
    value: float,
    unit: str = "ms",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to record a metric on the global monitor."""
    monitor = get_performance_monitor()
    monitor.record_metric(metric_name, value, unit, metadata)


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        metric_name: str,
        unit: str = "ms",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize timer.
        
        Args:
            metric_name: Name of the metric to record
            unit: Unit of measurement
            metadata: Additional metadata
        """
        self.metric_name = metric_name
        self.unit = unit
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metric."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            
            # Convert to requested unit
            if self.unit == "ms":
                value = elapsed * 1000
            elif self.unit == "s":
                value = elapsed
            else:
                value = elapsed
            
            record_performance(
                self.metric_name,
                value,
                self.unit,
                self.metadata
            )

"""
Confidence visualization and analysis tools
"""

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import numpy as np
from typing import Dict, List
import base64
import io

class ConfidenceVisualizer:
    def __init__(self):
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use('dark_background')  # Professional dark theme
            except:
                pass
    
    def create_confidence_bar(self, confidence: float, threshold: float = 0.7) -> str:
        """
        Create confidence bar visualization
        Returns base64 encoded image
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""
            
        try:
            fig, ax = plt.subplots(figsize=(8, 2))
            
            # Color coding based on confidence level
            if confidence >= 0.8:
                color = '#10B981'  # Green - High confidence
            elif confidence >= 0.6:
                color = '#F59E0B'  # Yellow - Medium confidence  
            else:
                color = '#EF4444'  # Red - Low confidence
            
            # Create horizontal bar
            ax.barh(0, confidence, height=0.3, color=color, alpha=0.8)
            ax.axvline(x=threshold, color='white', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
            
            # Styling
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('Confidence Score', fontsize=12, color='white')
            ax.set_title(f'Prediction Confidence: {confidence:.2%}', fontsize=14, color='white')
            ax.tick_params(colors='white')
            
            # Remove y-axis
            ax.set_yticks([])
            
            # Add percentage labels
            ax.text(confidence/2, 0, f'{confidence:.1%}', 
                    ha='center', va='center', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='#1F2937', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"Confidence visualization error: {e}")
            return ""
    
    def create_model_comparison_chart(self, predictions: Dict[str, float]) -> str:
        """
        Create comparison chart for multiple model predictions
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""
            
        try:
            models = list(predictions.keys())
            confidences = list(predictions.values())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar chart
            colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
            bars = ax.bar(models, confidences, color=colors[:len(models)])
            
            # Add value labels on bars
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{conf:.2%}', ha='center', va='bottom', color='white')
            
            ax.set_ylabel('Confidence Score', color='white')
            ax.set_title('Model Prediction Comparison', fontsize=16, color='white')
            ax.set_ylim(0, 1)
            ax.tick_params(colors='white')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='#1F2937', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"Comparison chart error: {e}")
            return ""

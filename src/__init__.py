"""
Wheat Spike Analyzer
Analyse phénotypique d'épis de blé
"""

__version__ = "1.0.0"
__author__ = "Votre nom"

try:
    from .analyzer import WheatSpikeAnalyzer
except ImportError:
    WheatSpikeAnalyzer = None

from .analyzer_obb import WheatSpikeAnalyzerOBB

__all__ = ['WheatSpikeAnalyzer', 'WheatSpikeAnalyzerOBB']

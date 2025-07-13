#!/usr/bin/env python3
"""
servers/self_reflection package - Self-analysis and improvement capabilities

This package provides MCP servers for AGIcommander's self-reflection capabilities,
including introspection, improvement proposals, controlled self-modification,
performance metrics, and capability evolution tracking.
"""

from .introspect import IntrospectionMCPServer
from .improve import ImprovementMCPServer
from .metrics import MetricsMCPServer
from .evolution import EvolutionMCPServer

__all__ = [
    'IntrospectionMCPServer',
    'ImprovementMCPServer',
    'MetricsMCPServer', 
    'EvolutionMCPServer'
]

__version__ = '0.1.0'

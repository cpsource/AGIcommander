#!/usr/bin/env python3
"""
servers/self_reflection package - Self-analysis and improvement capabilities

This package provides MCP servers for AGIcommander's self-reflection capabilities,
including introspection, improvement proposals, and controlled self-modification.
"""

from .introspect import IntrospectionMCPServer
from .improve import ImprovementMCPServer

__all__ = [
    'IntrospectionMCPServer',
    'ImprovementMCPServer'
]

__version__ = '0.1.0'

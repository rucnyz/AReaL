"""
Google ADK (Agent Development Kit) compatible model for AReaL.

This module provides ArealLlm, a Google ADK BaseLlm-compatible wrapper
that uses AReaL's ArealOpenAI client for LLM inference with automatic
token log probability tracking and reward management.
"""

from areal.experimental.adk.areal_llm import ArealLlm

__all__ = ["ArealLlm"]

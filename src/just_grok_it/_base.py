"""
Base provider interface for LLM SDK patching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseProvider(ABC):
    """
    Abstract base class for LLM SDK providers.

    To add a new provider:
    1. Create a new class that inherits from BaseProvider
    2. Implement all abstract methods
    3. Register the provider in providers/__init__.py
    """

    # Provider name (e.g., "openai", "anthropic", "gemini")
    name: str = ""

    # Package name to check for installation
    package_name: str = ""

    def __init__(self) -> None:
        self._patched = False
        self._default_model: Optional[str] = None

    @abstractmethod
    def is_installed(self) -> bool:
        """
        Check if the provider's SDK is installed.

        Returns:
            True if the SDK is installed, False otherwise.
        """
        pass

    @abstractmethod
    def patch(self, default_model: Optional[str] = None) -> None:
        """
        Patch the SDK to redirect API calls to xAI.

        Args:
            default_model: Default model to use if not specified in requests.
        """
        pass

    @abstractmethod
    def unpatch(self) -> None:
        """
        Remove the patch and restore original SDK behavior.
        """
        pass

    def is_patched(self) -> bool:
        """
        Check if the SDK is currently patched.

        Returns:
            True if patched, False otherwise.
        """
        return self._patched

"""
Core patching logic for redirecting LLM API requests to xAI APIs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from just_grok_it._base import BaseProvider
from just_grok_it._constants import XAI_BASE_URL

logger = logging.getLogger("just_grok_it")

# Global registry of provider instances
_provider_instances: Dict[str, BaseProvider] = {}


def _get_providers() -> List[Type[BaseProvider]]:
    """Get all registered provider classes."""
    from just_grok_it.providers import PROVIDERS

    return PROVIDERS


def _get_or_create_provider(provider_class: Type[BaseProvider]) -> BaseProvider:
    """Get or create a provider instance."""
    name = provider_class.name
    if name not in _provider_instances:
        _provider_instances[name] = provider_class()
    return _provider_instances[name]


def patch_all(default_model: Optional[str] = None) -> Dict[str, bool]:
    """
    Patch all installed LLM SDKs to redirect API calls to xAI.

    Automatically detects which SDKs are installed and patches them.

    Args:
        default_model: Default model to use if not specified in requests.

    Returns:
        Dictionary mapping provider names to whether they were patched.
    """
    results = {}

    for provider_class in _get_providers():
        provider = _get_or_create_provider(provider_class)

        if provider.is_installed():
            try:
                provider.patch(default_model)
                results[provider.name] = True
                logger.debug(
                    f"[just_grok_it] Patched {provider.name} SDK to redirect to xAI APIs"
                )
            except Exception as e:
                results[provider.name] = False
                logger.warning(f"[just_grok_it] Failed to patch {provider.name}: {e}")
        else:
            results[provider.name] = False
            logger.debug(f"[just_grok_it] {provider.name} SDK not installed, skipping")

    return results


def unpatch_all() -> None:
    """
    Remove patches from all LLM SDKs.

    Restores original SDK behavior.
    """
    for provider_class in _get_providers():
        provider = _get_or_create_provider(provider_class)
        if provider.is_patched():
            provider.unpatch()
            logger.debug(f"[just_grok_it] Unpatched {provider.name} SDK")


def is_patched(provider_name: Optional[str] = None) -> bool:
    """
    Check if an SDK is currently patched.

    Args:
        provider_name: Name of the provider to check (e.g., "openai", "anthropic").
                      If None, returns True if any provider is patched.

    Returns:
        True if patched, False otherwise.
    """
    if provider_name:
        for provider_class in _get_providers():
            if provider_class.name == provider_name:
                provider = _get_or_create_provider(provider_class)
                return provider.is_patched()
        return False

    # Check if any provider is patched
    for provider_class in _get_providers():
        provider = _get_or_create_provider(provider_class)
        if provider.is_patched():
            return True
    return False


def get_patched_providers() -> List[str]:
    """
    Get list of currently patched provider names.

    Returns:
        List of provider names that are currently patched.
    """
    patched = []
    for provider_class in _get_providers():
        provider = _get_or_create_provider(provider_class)
        if provider.is_patched():
            patched.append(provider.name)
    return patched


def get_installed_providers() -> List[str]:
    """
    Get list of installed provider names.

    Returns:
        List of provider names whose SDKs are installed.
    """
    installed = []
    for provider_class in _get_providers():
        provider = _get_or_create_provider(provider_class)
        if provider.is_installed():
            installed.append(provider.name)
    return installed


# Legacy API for backwards compatibility
def patch_openai(default_model: Optional[str] = None) -> None:
    """
    Patch only the OpenAI SDK to redirect API calls to xAI.

    Args:
        default_model: Default model to use if not specified in requests.
    """
    from just_grok_it.providers import OpenAIProvider

    provider = _get_or_create_provider(OpenAIProvider)
    provider.patch(default_model)


def unpatch_openai() -> None:
    """Remove the patch from the OpenAI SDK."""
    from just_grok_it.providers import OpenAIProvider

    provider = _get_or_create_provider(OpenAIProvider)
    provider.unpatch()


def all(default_model: Optional[str] = None) -> Dict[str, bool]:
    """
    Activate xAI API redirection for all installed LLM SDKs.

    This is the main entry point for the library. Call this once at the start
    of your application to redirect all LLM API requests to xAI APIs.

    Supported SDKs:
    - OpenAI (openai)
    - OpenRouter (openrouter)
    - Anthropic (anthropic)
    - Google Gemini (google-generativeai)
    - Mistral AI (mistralai)
    - Cohere (cohere)

    Args:
        default_model: Default model to use if not specified in requests.
                      For example: 'grok-4', 'grok-3', 'grok-3-mini', etc.

    Returns:
        Dictionary mapping provider names to whether they were patched.

    Example:
        >>> import just_grok_it
        >>> just_grok_it.all(default_model='grok-4')
        {'openai': True, 'anthropic': True, 'gemini': False, ...}
        >>>
        >>> # Now all OpenAI SDK calls will use xAI
        >>> from openai import OpenAI
        >>> client = OpenAI()  # Will connect to xAI APIs
        >>> response = client.chat.completions.create(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )  # Uses grok-4 by default

    Note:
        The user is responsible for setting the XAI_API_KEY environment variable
        (or OPENAI_API_KEY) before making API calls.
    """
    from just_grok_it._constants import DEFAULT_XAI_MODEL

    actual_model = default_model or DEFAULT_XAI_MODEL
    results = patch_all(default_model=default_model)

    patched = [name for name, success in results.items() if success]
    not_patched = [name for name, success in results.items() if not success]

    if patched:
        logger.debug(f"[just_grok_it] Initialized with default model: {actual_model}")
        logger.debug(f"[just_grok_it] Patched providers: {', '.join(patched)}")
        if not_patched:
            logger.debug(
                f"[just_grok_it] Skipped providers (not installed): {', '.join(not_patched)}"
            )
        logger.debug(
            f"[just_grok_it] All requests will now be redirected to {XAI_BASE_URL}"
        )
    else:
        logger.debug(
            "[just_grok_it] No supported LLM SDKs found to patch. "
            "Install one of: openai, openrouter, anthropic, google-generativeai, mistralai, cohere"
        )

    return results

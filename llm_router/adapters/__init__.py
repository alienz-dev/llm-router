from .base import BaseAdapter, AdapterResponse, QuotaSnapshot
from .openrouter import OpenRouterAdapter
from .google import GoogleAdapter
from .cerebras import CerebrasAdapter
from .groq import GroqAdapter
from .mistral import MistralAdapter
from .nvidia import NvidiaAdapter
from .kilo import KiloAdapter
from .cloudflare import CloudflareAdapter
from .huggingface import HuggingFaceAdapter
from .opencode import OpenCodeAdapter
from .deepseek import DeepSeekAdapter

# Adapter registry for dynamic instantiation
ADAPTERS = {
    "openrouter": OpenRouterAdapter,
    "google": GoogleAdapter,
    "cerebras": CerebrasAdapter,
    "groq": GroqAdapter,
    "mistral": MistralAdapter,
    "nvidia": NvidiaAdapter,
    "kilo": KiloAdapter,
    "cloudflare": CloudflareAdapter,
    "huggingface": HuggingFaceAdapter,
    "opencode": OpenCodeAdapter,
    "deepseek": DeepSeekAdapter,
}

__all__ = [
    "BaseAdapter",
    "AdapterResponse", 
    "QuotaSnapshot",
    "OpenRouterAdapter",
    "GoogleAdapter",
    "CerebrasAdapter",
    "GroqAdapter",
    "MistralAdapter",
    "NvidiaAdapter",
    "KiloAdapter",
    "CloudflareAdapter",
    "HuggingFaceAdapter",
    "ADAPTERS",
]
from __future__ import annotations

from langchain_ollama import OllamaLLM, OllamaEmbeddings  # For interacting with Ollama LLM and embeddings
from .ollama_config import LLMSettings

from inspect import signature, Parameter

# llm_params.py
from typing import Dict, Literal, Optional

Task = Literal["summarization", "qa", "qa_open"]

def get_params(
    task: Task,
    *,
    context_tokens: int = 8192,          # bump to your model's max context
    max_tokens: Optional[int] = None,    # a.k.a. num_predict
    deterministic: bool = True,          # set a seed for stable outputs
    overrides: Optional[Dict] = None,    # last-mile tuning
) -> Dict:
    """
    Return a parameter dict suitable for Ollama (and most llama.cpp-backed servers).
    Keys not used by your model/server will be ignored harmlessly.

    Tasks:
      - "summarization": concise, faithful summaries (low temp)
      - "qa": extractive / grounded answers (very low temp)
      - "qa_open": open-ended Q&A (slightly more creative)

      Tuning tips (quick)
        * Too verbose? Lower num_predict (e.g., 256 → 160) or increase repeat_penalty slightly (up to ~1.15).
        * Too short? Raise num_predict.
        * Hallucinations? Lower temperature (0.1–0.2) and/or top_p (0.7–0.85).
        * Determinism across runs? Keep seed set and avoid multi-sampling features (mirostat=0).
    """
    base = {
        "num_ctx": context_tokens,
        "seed": 42 if deterministic else None,  # remove for nondeterministic
        # Turn off advanced samplers unless you know you want them:
        "mirostat": 0,                          # 0 off, 1/2 on (if supported)
        # You can add stop tokens here if your stack uses special prompts:
        # "stop": ["\nUser:", "###"],
    }

    presets: dict[Task, Dict] = {
        "summarization": {
            # Tight sampling = less drift and better fidelity
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05,
            "num_predict": 512 if max_tokens is None else max_tokens,
            # Keep novelty penalties neutral for factual compression
            "presence_penalty": 0.0,    # if supported
            "frequency_penalty": 0.0,   # if supported
        },
        "qa": {  # extractive, grounded Q&A (RAG-style)
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": 256 if max_tokens is None else max_tokens,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "qa_open": {  # open-ended / generative answers
            "temperature": 0.3,
            "top_p": 0.92,
            "top_k": 50,
            "repeat_penalty": 1.05,
            "num_predict": 512 if max_tokens is None else max_tokens,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    }

    params = {**base, **presets[task]}
    if overrides:
        params.update(overrides)
    # Strip Nones to avoid surprising behavior
    return {k: v for k, v in params.items() if v is not None}


def init_ollama_with_params(cls, config: LLMSettings, params: dict):
    """Safely instantiate cls from a dict:
    - Keeps only kwargs accepted by __init__
    - If cls supports 'model_kwargs' or 'additional_kwargs', funnel extras there
    """
    sig = signature(cls.__init__)
    allowed = {
        name for name, p in sig.parameters.items()
        if name != "self" and p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }

    kwargs = {k: v for k, v in params.items() if k in allowed}
    extras = {k: v for k, v in params.items() if k not in allowed}

    # Common sinks across libraries
    for sink in ("model_kwargs", "additional_kwargs"):
        if sink in allowed and extras:
            kwargs[sink] = {**kwargs.get(sink, {}), **extras}
            extras = {}
            break

    if extras:
        # Prefer failing fast instead of silently ignoring unknowns
        unknown = ", ".join(sorted(extras))
        raise TypeError(f"Unknown parameters for {cls.__name__}: {unknown}")

    return cls(model=config.model,base_url=config.base_url, additional_kwargs=kwargs)


def initialize_ollama_llm(config: LLMSettings, params: dict):
    # Create and return an instance of the OllamaLLM with the specified configuration
    return OllamaLLM(
        model=config.model,                    # Set the model ID for the LLM
        base_url=config.base_url,                 # Set the base URL for Ollama service
        temperature=params.get("temperature", config.temperature),  # Set temperature
        top_p=params.get("top_p", 1.0),             # Set top_p
        num_predict=params.get("max_tokens", 900)   # Set max tokens
    )

def setup_embedding_model(config: LLMSettings):
    # Create and return an instance of OllamaEmbeddings with the specified configuration
    return OllamaEmbeddings(
        model=config.embedding_model,           # Set the embedding model ID
        base_url=config.base_url                  # Set the base URL for Ollama service
    )


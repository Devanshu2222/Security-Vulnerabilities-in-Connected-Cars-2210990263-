"""
LLM Interface — supports OpenAI, Anthropic, HuggingFace, and Ollama.
Provides a unified API for question answering and summarization.
"""

import os
from typing import Optional, Dict, List, Generator
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """Abstract base for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str, system: str = "", max_tokens: int = 1024,
               temperature: float = 0.2) -> Generator[str, None, None]:
        pass

    def chat(self, messages: List[Dict], max_tokens: int = 1024,
             temperature: float = 0.2) -> str:
        """Generic chat interface (converts to prompt for non-chat models)."""
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        prompt = "\n\n".join(user_msgs)
        return self.generate(prompt, system=system, max_tokens=max_tokens,
                             temperature=temperature)


# ─────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    """OpenAI GPT models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.)"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
        except ImportError:
            raise ImportError("Run: pip install openai")

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def stream(self, prompt: str, system: str = "", max_tokens: int = 1024,
               temperature: float = 0.2) -> Generator[str, None, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    def chat(self, messages: List[Dict], max_tokens: int = 1024,
             temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# Anthropic
# ─────────────────────────────────────────────────────────────

class AnthropicLLM(BaseLLM):
    """Anthropic Claude models."""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = model
        except ImportError:
            raise ImportError("Run: pip install anthropic")

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def stream(self, prompt: str, system: str = "", max_tokens: int = 1024,
               temperature: float = 0.2) -> Generator[str, None, None]:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def chat(self, messages: List[Dict], max_tokens: int = 1024,
             temperature: float = 0.2) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=user_messages,
        )
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        return response.content[0].text.strip()


# ─────────────────────────────────────────────────────────────
# HuggingFace (local inference)
# ─────────────────────────────────────────────────────────────

class HuggingFaceLLM(BaseLLM):
    """
    Local HuggingFace text-generation models.
    Recommended: 'mistralai/Mistral-7B-Instruct-v0.2' or 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    """

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 device: str = "auto", load_in_4bit: bool = False):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        except ImportError:
            raise ImportError("Run: pip install transformers torch accelerate")

        print(f"[HuggingFace] Loading {model_name}...")

        quantization_config = None
        if load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                print("[HuggingFace] 4-bit quantization not available, loading in full precision")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device_map=device,
            model_kwargs={"quantization_config": quantization_config} if quantization_config else {},
            torch_dtype="auto",
        )
        print(f"[HuggingFace] Model loaded.")

    def _build_prompt(self, prompt: str, system: str = "") -> str:
        if system:
            return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
        return f"<s>[INST] {prompt} [/INST]"

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
        full_prompt = self._build_prompt(prompt, system)
        result = self.pipeline(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = result[0]["generated_text"]
        # Strip the prompt
        if "[/INST]" in generated:
            return generated.split("[/INST]")[-1].strip()
        return generated[len(full_prompt):].strip()

    def stream(self, prompt: str, system: str = "", max_tokens: int = 1024,
               temperature: float = 0.2) -> Generator[str, None, None]:
        # HuggingFace pipeline doesn't natively stream — yield whole response
        yield self.generate(prompt, system=system, max_tokens=max_tokens,
                            temperature=temperature)


# ─────────────────────────────────────────────────────────────
# Ollama (local server)
# ─────────────────────────────────────────────────────────────

class OllamaLLM(BaseLLM):
    """
    Ollama local server (https://ollama.com).
    Run 'ollama pull llama3' or 'ollama pull mistral' first.
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self):
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Ollama] Warning: cannot reach Ollama server at {self.base_url}: {e}")

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def stream(self, prompt: str, system: str = "", max_tokens: int = 1024,
               temperature: float = 0.2) -> Generator[str, None, None]:
        import requests, json
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with requests.post(f"{self.base_url}/api/generate", json=payload,
                           stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def create_llm(config: Dict) -> BaseLLM:
    """Factory function to create an LLM from config."""
    provider = config.get("llm_provider", "anthropic").lower()

    if provider == "openai":
        return OpenAILLM(
            model=config.get("llm_model", "gpt-4o-mini"),
            api_key=config.get("openai_api_key"),
        )
    elif provider == "anthropic":
        return AnthropicLLM(
            model=config.get("llm_model", "claude-3-haiku-20240307"),
            api_key=config.get("anthropic_api_key"),
        )
    elif provider == "huggingface":
        return HuggingFaceLLM(
            model_name=config.get("llm_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            load_in_4bit=config.get("load_in_4bit", False),
        )
    elif provider == "ollama":
        return OllamaLLM(
            model=config.get("llm_model", "llama3"),
            base_url=config.get("ollama_url", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose: openai, anthropic, huggingface, ollama")

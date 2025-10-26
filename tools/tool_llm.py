"""
Hugging Face LLM integration for AnswerTool.
Focused implementation using Hugging Face transformers.
"""

import os
from typing import Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class HuggingFaceLLM:
    """Hugging Face LLM wrapper optimized for AnswerTool."""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.generator = None
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model."""
        try:
            # Optimized model loading configuration
            model_kwargs = {
                "model": self.model_name,
                "device_map": self.device,
                "trust_remote_code": True,
                "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            # Add model-specific optimizations
            if "deepseek" in self.model_name.lower():
                # DeepSeek models work better with specific settings
                model_kwargs.update({
                    "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "low_cpu_mem_usage": True,
                })
            elif "mistral" in self.model_name.lower():
                # Mistral models optimization
                model_kwargs.update({
                    "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "eager",
                })
            
            self.generator = pipeline(
                "text-generation",
                **model_kwargs
            )
                
        except ImportError:
            print("Warning: transformers not installed. Install with: pip install transformers torch accelerate")
            self.generator = None
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to mock LLM...")
            self.generator = None
    
    def generate(self, prompt: str, question: str = None, max_tokens: int = 100, temperature: float = 0.1) -> str:
        """Generate response using Hugging Face model."""
        
        if self.generator is None:
            return self._generate_mock(prompt, question)
        
        try:
            # Optimized generation parameters for speed
            generation_kwargs = {
                "max_new_tokens": min(max_tokens, 150),  # Cap at 150 tokens for speed
                "temperature": max(0.1, min(temperature, 0.8)),  # Lower temperature for faster generation
                "do_sample": True,
                "pad_token_id": self.generator.tokenizer.eos_token_id,
                "eos_token_id": self.generator.tokenizer.eos_token_id,
                "repetition_penalty": 1.05,  # Lower penalty for speed
                "no_repeat_ngram_size": 2,
                "num_beams": 1,  # No beam search for speed
                "use_cache": True,  # Enable KV cache for speed
            }
            
            # Remove early_stopping to avoid warnings
            if "early_stopping" in generation_kwargs:
                del generation_kwargs["early_stopping"]
            
            result = self.generator(prompt, **generation_kwargs)
            
            # Extract generated text (remove the original prompt)
            generated_text = result[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Clean up the response
            generated_text = self._clean_response(generated_text)
            
            return generated_text
            
        except Exception as e:
            print(f"Hugging Face generation error: {e}")
            return self._generate_mock(prompt, question)
    
    def _clean_response(self, text: str) -> str:
        """Clean up the generated response."""
        # Remove excessive repetition
        lines = text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line)
        
        # Join and limit length
        cleaned_text = '\n'.join(cleaned_lines)
        if len(cleaned_text) > 500:  # Limit response length
            cleaned_text = cleaned_text[:500] + "..."
        
        return cleaned_text
    
    def _generate_mock(self, prompt: str, question: str) -> str:
        """Generate mock response when model is not available."""
        if "Do you know anything about this question?" in prompt:
            return f"Background: This is a complex topic that requires careful analysis of multiple sources."
        else:
            return f"Based on the provided context, here is a comprehensive answer. The evidence suggests multiple perspectives and recent developments."

def create_deepseek_r1_distill_qwen_15b():
    """Create DeepSeek R1 Distill Qwen 1.5B model (fast, small)."""
    return HuggingFaceLLM(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

def create_mistral_7b():
    """Create Mistral 7B model (medium size, good quality)."""
    return HuggingFaceLLM(model_name="mistralai/Mistral-7B-v0.1")

def create_gpt2():
    """Create GPT-2 model (fast, small)."""
    return HuggingFaceLLM(model_name="openai-community/gpt2")

class MockLLM:
    """Mock LLM for testing without dependencies."""
    
    def __init__(self, name="mock"):
        self.name = name
    
    def generate(self, prompt, question=None, max_tokens=100, temperature=0.1):
        if "Do you know anything about this question?" in prompt:
            return f"Background: This is a complex topic that requires careful analysis of multiple sources."
        else:
            return f"Based on the provided context, here is a comprehensive answer. The evidence suggests multiple perspectives and recent developments."

# Model recommendations for different use cases
RECOMMENDED_MODELS = {
    "balanced": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "quality": "mistralai/Mistral-7B-v0.1",
    "fast": "openai-community/gpt2",
}

# Model size categories for easy selection
MODEL_CATEGORIES = {
    "tiny": ["openai-community/gpt2"],
    "small": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
    "medium": ["mistralai/Mistral-7B-v0.1"],
}
  
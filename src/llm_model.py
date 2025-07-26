import os
from langchain_cohere import ChatCohere
from langchain_community.llms import HuggingFacePipeline, Ollama # Import Ollama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from src.config import COHERE_API_KEY, HF_TOKEN

def get_llm(model_type: str = "cohere", quantized: bool = True):
    """
    Returns the chosen LLM.
    Options: 'cohere', 'qwen2-7b-instruct', 'ollama-qwen3-4b'.
    """
    if model_type == "cohere":
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is not set in environment variables.")
        print("Using Cohere command-a-03-2025 as LLM.")
        return ChatCohere(model="command-a-03-2025", cohere_api_key=COHERE_API_KEY, temperature=0.3) # Added temperature
    elif model_type == "qwen2-7b-instruct":
        print(f"Using Qwen2-7B-Instruct (Hugging Face) as LLM. Quantized: {quantized}")
        model_id = "Qwen/Qwen2-7B-Instruct"

        # Quantization config
        if quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
            token=HF_TOKEN
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        return HuggingFacePipeline(pipeline=pipe)
    elif model_type == "ollama-qwen3-4b":
        print("Using Qwen3:4b from Ollama as LLM.")
        # Ensure Ollama is running and qwen3:4b is pulled
        return Ollama(model="qwen3:4b", temperature=0.3) # Added temperature
    else:
        raise ValueError(f"Unknown LLM model type: {model_type}. Choose 'cohere', 'qwen2-7b-instruct', or 'ollama-qwen3-4b'.")
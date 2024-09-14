import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, LoraModel

def model_with_lora_and_quantization():
    token = "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"  
    model_name = "google/gemma-2b-it"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    
    # Define Lora configuration
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    # Apply Lora configuration
    model = LoraModel(model, { "lora_adapter": lora_config }, "lora_adapter")
    
    # Apply dynamic quantization
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8   # Quantize to 8-bit integers
    )
    
    return model, tokenizer


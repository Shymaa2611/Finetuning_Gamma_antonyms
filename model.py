import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

def model_Quantization():
    token = "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"  
    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear}, 
        dtype=torch.qint8   
    )
    
    return model, tokenizer

def Lora_Configuration():
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    return lora_config


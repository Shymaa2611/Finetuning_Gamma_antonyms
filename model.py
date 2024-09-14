from transformers import BitsAndBytesConfig
import torch
from transformers import Gemma2ForCausalLM,AutoTokenizer
from peft import LoraConfig




def model_Quantization():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    token="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"  
    model_name = "google/gemma-2b-it"
   
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=token)
    model = Gemma2ForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        use_auth_token=token
    )
    
    return model, tokenizer

def Lora_Configuration():
  lora_config=LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
   )
  return lora_config
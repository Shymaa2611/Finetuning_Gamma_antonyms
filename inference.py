from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    return model, tokenizer

def generate_inference(model, tokenizer, input_text, max_length=50):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=5, 
        early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

if __name__=="__main__":
  checkpoint_dir = "/kaggle/working/Finetuning_Gamma_antonyms/outputs/checkpoint-300"
  model, tokenizer = load_model_and_tokenizer(checkpoint_dir)
  input_text = "present"
  generated_text = generate_inference(model, tokenizer, input_text)
  print(generated_text)

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig  # Import SFTConfig
from transformers import TrainingArguments

def Training_Arguments():
    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        report_to="none" 
    )
    return args

def Trainer(train_loader, model, lora_config):
    args = Training_Arguments()
    
    # Create SFTConfig and set arguments
    sft_config = SFTConfig(
        dataset_text_field="lemma",  # Update to your specific field
        packing=False,  # Update to True if needed
        max_seq_length=1024  # Set max sequence length or any other config needed
    )
    
    # Initialize SFTTrainer with SFTConfig
    trainer = SFTTrainer(
        args=args,
        train_dataset=train_loader.dataset,
        model=model,
        peft_config=lora_config,
        sft_config=sft_config  # Pass the SFTConfig here
    )
    
    return trainer

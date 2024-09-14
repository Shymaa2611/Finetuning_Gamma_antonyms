from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from transformers import TrainingArguments

def Training_Arguments():
    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=300,
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
    
    trainer = SFTTrainer(
        args=args,
        train_dataset=train_loader.dataset,
        model=model,
        peft_config=lora_config,
        dataset_text_field="lemma",  
        packing=False  
    )
    
    return trainer


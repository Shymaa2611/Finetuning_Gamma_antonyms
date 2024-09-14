from trl import SFTTrainer
from transformers import TrainingArguments

def Training_Arguments():
    args = TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        warmup_steps=100,  
        max_steps=100    
        learning_rate=1e-5,  
        fp16=True,
        logging_steps=10, 
        output_dir="outputs",
        optim="paged_adamw_8bit",
        report_to="tensorboard",  
        save_steps=100,   
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

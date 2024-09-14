from training import Trainer
from dataset import get_data
from model import model_Quantization,Lora_Configuration
#from evaluate import evaluate
def run(train_loader,model,lora_config):
    trainer=Trainer(train_loader,model,lora_config)
    trainer.train()
    



if __name__=="__main__":
    model,tokenizer=model_Quantization()
    lora_config=Lora_Configuration()
    train_data, test_data, train_loader, test_loader=get_data('data/antonyms.csv',tokenizer)
    run(train_loader,model,lora_config)



   
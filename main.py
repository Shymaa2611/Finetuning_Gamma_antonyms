from training import Trainer
from dataset import get_data
#from model import model_Quantization,Lora_Configuration
#from evaluate import evaluate_model
from model import model_with_lora_and_quantization
def run(train_loader,model):
    trainer=Trainer(train_loader,model)
    trainer.train()
    #eval_results = evaluate_model(trainer, test_loader)
    #print("Evaluation Results:", eval_results)
    



if __name__=="__main__":
    model,tokenizer=model_with_lora_and_quantization()
    #lora_config=Lora_Configuration()
    train_data, test_data, train_loader, test_loader=get_data('data/antonyms.csv',tokenizer)
    run(train_loader,model)



   
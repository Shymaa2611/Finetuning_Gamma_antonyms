from trl import SFTTrainer

def evaluate_model(trainer, eval_dataset):
    trainer.eval_dataset = eval_dataset
    eval_results = trainer.evaluate()
    return eval_results

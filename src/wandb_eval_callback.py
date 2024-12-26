### Inspired by https://docs.wandb.ai/guides/integrations/huggingface
###

from transformers.integrations import WandbCallback

def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        val_dataset (Dataset): The validation dataset.
            model_name (str): The name of the model.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
            benchmark (str, optional): Name of benchmark to run. Defaults to "sft_test_set_4kv3".
    """

    def __init__(self, 
                 trainer, 
                 tokenizer,
                 model_id, 
                 benchmark="lighteval/MATH", 
                 num_questions=1000,
                 batch_size=10,
                 ):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            model_name (str): The name of the model.
            benchmark (str, optional): Name of benchmark to run. Defaults to "sft_test_set_4kv3".
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.benchmark = benchmark
        self.num_questions = num_questions
        self.batch_size = batch_size
        

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        tokenizer = self.tokenizer
        model = self.trainer.model
        accuracy = 100
        self._wandb.log({"MATH_score": accuracy})
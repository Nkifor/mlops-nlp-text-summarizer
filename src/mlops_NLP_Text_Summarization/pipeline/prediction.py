from mlops_NLP_Text_Summarization.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline




class ModelEvaluationTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    def predict(self, text, length_penalty= 0.8, number_of_beams=6, max_length= 500):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": length_penalty, "num_beams":number_of_beams, "max_length": max_length}
# initially lenght_penalty:0.8, num_beams:8, max_length:128
        pipe = pipeline("summarization", model=self.config.model_path,tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output


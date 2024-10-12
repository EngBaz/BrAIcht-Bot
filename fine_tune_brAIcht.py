from peft import PeftModel
from fine_tune_utilities import *


log_output_path = "/content/drive/MyDrive/BrAIcht/log_output"
train_dataset_path = "/content/drive/MyDrive/BrAIcht/data/brecht_plays_train_dataset.csv"
validation_dataset_path="/content/drive/MyDrive/BrAIcht/data/brecht_plays_validation_dataset.csv"
model_path = "meta-llama/Llama-3.2-3B-Instruct"
qlora_model_path = "/content/drive/MyDrive/BrAIcht/qlora_model"
merged_model_path = "/content/drive/MyDrive/BrAIcht/finetuned_model"


train_dataset, validation_dataset = get_dataset(train_dataset_path, validation_dataset_path)

trainer = train_args(log_output_path, train_dataset, validation_dataset, model_path)
trainer.train()

trainer.model.save_pretrained(qlora_model_path)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             trust_remote_code=True,
                                             #attn_implementation="flash_attention_2",
                                            ) 

merged_model = PeftModel.from_pretrained(model, qlora_model_path)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(merged_model_path, safe_serialization=True)

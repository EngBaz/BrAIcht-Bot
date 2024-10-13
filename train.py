from peft import PeftModel
from transformers import AutoTokenizer
from QloraTrainer import *


log_output_path = "/content/drive/MyDrive/BrAIcht/log_output"
train_dataset_path = "/content/drive/MyDrive/BrAIcht/data/brecht_plays_train_dataset.csv"
validation_dataset_path="/content/drive/MyDrive/BrAIcht/data/brecht_plays_validation_dataset.csv"
model_path = "meta-llama/Llama-3.2-3B-Instruct"
qlora_model_path = "/content/drive/MyDrive/BrAIcht/qlora_model"
merged_model_path = "/content/drive/MyDrive/BrAIcht/finetuned_model"

# Import the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          model_max_length=4096,
                                          add_eos_token=True,
                                          )
tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# Import the base model to merge with the adapters weights
base_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  #attn_implementation="flash_attention_2",
                                                ) 

train_dataset, validation_dataset = get_dataset(train_dataset_path, 
                                                validation_dataset_path, 
                                                tokenizer,
                                                )

trainer = train_args(log_output_path, 
                     train_dataset, 
                     validation_dataset, 
                     model_path,
                     )

trainer.train()
trainer.model.save_pretrained(qlora_model_path)

merged_model = PeftModel.from_pretrained(base_model, 
                                         qlora_model_path,
                                         )

merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained(merged_model_path)

from qlora_trainer import *


log_output_path = "./log_output"
train_dataset_path = "./data/train_dataset/brecht_plays_train_dataset.csv"
validation_dataset_path="./data/validation_dataset/brecht_plays_validation_dataset.csv"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
qlora_adapters_path = "./qlora_adapters"
merged_model_path = "./merged_model"


if __name__=="__main__":
  
  print("Load base model...")              
  model, tokenizer = load_base_model(model_name)                                                                         

  print("Preparing and formatting the dataset...")
  train_dataset, validation_dataset = get_dataset(train_dataset_path, validation_dataset_path, tokenizer)

  print("Start training")
  train(log_output_path, train_dataset, validation_dataset, qlora_adapters_path, model, tokenizer)

  print("Merge and save the model")
  merge_and_save(model_name, qlora_adapters_path, merged_model_path)



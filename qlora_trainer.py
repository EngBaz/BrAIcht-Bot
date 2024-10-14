import torch
import pandas as pd

from trl import SFTTrainer
from datasets import Dataset
from peft import (LoraConfig,
                  prepare_model_for_kbit_training,
                  get_peft_model,
                  PeftModel,
)
from transformers import (AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          AutoTokenizer,
                          DataCollatorForLanguageModeling,
)


def load_base_model(model_path):
    
    """Loads and configures a quantized model with LoRA fine-tuning."""

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto",
                                                 trust_remote_code=True) #attn_implementation="flash_attention_2"
                                                
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map = "auto", trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def get_dataset(train_data_path, validation_data_path, tokenizer):
    
    """Loads and formats the train and validation datasets."""

    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(validation_data_path)

    train_data['text'] = '[INST] ' + train_data['instruction'] + ' [/INST]\n' + train_data['response'] + '\n' + tokenizer.eos_token     
    validation_data['text'] = '[INST] ' + validation_data['instruction'] + ' [/INST]\n' + validation_data['response'] + '\n' + tokenizer.eos_token 

    train_dataset = Dataset.from_pandas(train_data[['text']])
    validation_dataset = Dataset.from_pandas(validation_data[['text']])

    return train_dataset, validation_dataset


def train(output_dir, train_data, validation_data, qlora_model_path, model, tokenizer):
    
    """
    Set up the LoRA configuration and trains the model.
    
    """

    config = LoraConfig(r=32, lora_alpha=16, bias="none", lora_dropout=0.1,
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", "lm_head"
                            ],
                        task_type="CAUSAL_LM",
                        )
    
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        #num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing = True,
        evaluation_strategy = 'steps',
        save_steps=50, # save log_output
        eval_steps=50, # do the evaluation after x steps
        logging_steps=10, # show training loss
        max_steps=50,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        peft_config=config,
        dataset_text_field="text",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
    
    trainer.train()
    trainer.model.save_pretrained(qlora_model_path)


def merge_and_save(model_path, qlora_model_path, merged_model_path):
    
    """ Merge base model and adapter, save to disk """
    
    base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    model = PeftModel.from_pretrained(base_model, qlora_model_path)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(merged_model_path)
    
def print_trainable_parameters(model):
    
    """Prints the number of trainable parameters in the model."""
    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    





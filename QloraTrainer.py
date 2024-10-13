import torch
import logging
import pandas as pd

from trl import SFTTrainer
from datasets import Dataset
from peft import (LoraConfig,
                  prepare_model_for_kbit_training,
                  get_peft_model,
)
from transformers import (AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """
    Loads and configures a quantized model with LoRA fine-tuning

    Args:
        model_path (str): Path to the pre-trained model

    Returns:
        model: The language model which is configured for quantized low-bit training and equipped with LoRA layers
        tokenizer: The tokenizer associated with the model with a maximum sequence length and padding settings
        config: The LoRA configuration used for model fine-tuning
    """

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=False,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=compute_dtype,
                                    )

    if compute_dtype == torch.float16:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("=" * 80)
            logger.info("GPU supports bfloat16: accelerate training with bf16=True")
            logger.info("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=bnb_config,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 #attn_implementation="flash_attention_2",
                                                 )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    config = LoraConfig(
      r=32,
      lora_alpha=16,
      bias="none",
      lora_dropout=0.1,
      target_modules=[
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj", "lm_head"],
      task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    return model, config



def get_dataset(train_data_path, validation_data_path, tokenizer):
    
    """
    Loads and formats training and validation datasets, preparing them for language model fine-tuning 
    by converting the instruction-response pairs into a specific format for tokenization

    Args:
        train_data_path (str): Path to the training data file
        validation_data_path (str): Path to the validation data file

    Returns:
        formatted_train_dataset: A dataset containing the formatted training data
        formatted_validation_dataset: A dataset containing the formatted validation data
    """

    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(validation_data_path)

    train_data['text'] = '[INST] ' + train_data['instruction'] + ' [/INST]\n' + train_data['response'] + '\n' + tokenizer.eos_token     
    validation_data['text'] = '[INST] ' + validation_data['instruction'] + ' [/INST]\n' + validation_data['response'] + '\n' + tokenizer.eos_token 

    formatted_train_dataset = Dataset.from_pandas(train_data[['text']])
    formatted_validation_dataset = Dataset.from_pandas(validation_data[['text']])

    return formatted_train_dataset, formatted_validation_dataset



def train_args(output_dir, train_data, validaiton_data, model_path):
    """
    Initializes the training process for a model by configuring training arguments and setting up the trainer 
    for fine-tuning using custom datasets and model settings

    Args:
        output_dir (str): Directory where model checkpoints and logs will be saved
        train_data (Dataset): The formatted training dataset
        validation_data (Dataset): The formatted validation dataset
        model_path (str): Path to the pre-trained model

    Returns:
        trainer: A trainer object configured for supervised fine-tuning, ready to train the model with specified 
        training arguments
    """

    logger.info(f"Loading model ....")

    model, tokenizer, peft_config = load_model(model_path)

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
      eval_dataset=validaiton_data,
      peft_config=peft_config,
      dataset_text_field="text",
      max_seq_length=4096,
      tokenizer=tokenizer,
      args=training_arguments,
      packing=False,
    )

    return trainer





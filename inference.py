from transformers import (AutoModelForCausalLM,
                          AutoTokenizer, pipeline)


merged_model_path = "./merged_model"
model_name = "meta-llama/Llama-3.2-3B-Instruct"

    
if __name__=="__main__":
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto", trust_remote_code=True)
    
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        truncation=True
    )
    
    user_prompt = "What is your favorite movie?"
    
    formatted_prompt = f"[INST] {user_prompt} [/INST]" + tokenizer.eos_token
    
    output = pipe(formatted_prompt)
    
    print(output[0]['generated_text'])
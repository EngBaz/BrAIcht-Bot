from transformers import (AutoModelForCausalLM,
                          AutoTokenizer, pipeline)


merged_model_path = "./merged_model"
model_name = "meta-llama/Llama-3.2-3B-Instruct"


def format_prompt(user_prompt):
    
    prompt_template=f"[INST] {user_prompt} [/INST]"
    
    return prompt_template


def get_response(prompt):
    
    output = pipe(format_prompt(prompt))

    return output
    

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
        repetition_penalty=1.15
    )
    
    print(get_response("What is your favorite movie?"))
    
import transformers

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id

def generate(prompt: str) -> str:
    """
    Generates a text from a model, using a provided prompt

    params: prompt (str): The input text prompt.
    returns: tuple: A tuple containing the latent space representation (hidden states) and the generated text.
    """
    # Tokenize the input prompt
    inputs: transformers.BatchEncoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate text from the input_ids
    generated_outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        max_new_tokens=500,
        num_return_sequences=1,
        temperature=0.6,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    generated_text: str = tokenizer.decode(generated_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return generated_text

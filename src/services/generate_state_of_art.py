from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id

def extract_latent_space(prompt) -> tuple:
    """
    Extracts the latent space representation of the model for a given prompt.

    params: prompt (str): The input text prompt.
    returns: tuple: A tuple containing the latent space representation (hidden states) and the generated text.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Pass the inputs through the model to get the outputs
    outputs = model(**inputs, output_hidden_states=True)

    # Extract the hidden states (latent space)
    hidden_states = outputs.hidden_states

    # Generate text from the latent space
    generated_outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        max_length=50, 
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return hidden_states, generated_text

if __name__ == "__main__":
    # Test the extract_latent_space function
    test_prompt = "Once upon a time"
    latent_space, test_generated_text = extract_latent_space(test_prompt)
    
    print("Latent Space Representation:")
    for layer in latent_space:
        print(layer.shape)
    
    print("\nGenerated Text:")
    print(test_generated_text)

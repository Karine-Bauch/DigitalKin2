from transformers import GPT2LMHeadModel, GPT2Tokenizer

from services.generate_state_of_art import generate

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id

def generate_text(prompt) -> tuple:
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
        max_new_tokens=500,
        num_return_sequences=1,
        temperature=0.6,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return hidden_states, generated_text

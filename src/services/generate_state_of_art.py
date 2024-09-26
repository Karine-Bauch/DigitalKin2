from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
        max_new_tokens=200, 
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
    )
    generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return hidden_states, generated_text

def generate_state_of_art_for_aider_developer() -> str:
    """
    Generates a state of the art text for an Aider developer who doesn't master Python,
    providing a global view of the latest technologies to automate Python development.

    returns: str: The generated state of the art text.
    """
    prompt = (
        "As an Aider developer looking to enhance your Python development skills, it's crucial to explore the latest "
        "technologies that can automate and streamline your workflow. These include advanced AI-driven code completion "
        "tools like GitHub Copilot, automated testing frameworks such as PyTest and Robot Framework, and robust "
        "continuous integration systems like Jenkins and GitLab CI/CD. Additionally, consider leveraging containerization "
        "with Docker and orchestration with Kubernetes to manage your applications efficiently. Here is an overview of "
        "these cutting-edge technologies and how they can transform your development process:"
    )
    
    # Call the generate_text function with the prompt
    _, generated_text = generate_text(prompt)

    return generated_text


if __name__ == "__main__":
    # Test the generate_state_of_art_for_aider_developer function
    state_of_art_text = generate_state_of_art_for_aider_developer()
    
    print("State of the Art Text for Aider Developer:")
    print(state_of_art_text)

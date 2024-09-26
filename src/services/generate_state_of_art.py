import services.generate_text
from services.structure_state_of_art import structure_state_of_art


def generate(custom_details: str = "") -> str:
    """
    Generates a state of the art text for an Aider developer who doesn't master Python,
    providing a global view of the latest technologies to automate Python development.

    returns: str: The generated state of the art text.
    """
    prompt = (
        f"As an Aider developer aiming to master Python automation technologies, this document provides a comprehensive "
        f"state of the art analysis. {custom_details}"
    )

    # Call the generate_text function with the prompt
    _, generated_text = services.generate_text.generate_text(prompt)

    structured_text = structure_state_of_art(generated_text)
    return structured_text


if __name__ == "__main__":
    # Test the generate_state_of_art_for_aider_developer function
    state_of_art_text = generate()

    print("State of the Art Text for Aider Developer:")
    print(state_of_art_text)

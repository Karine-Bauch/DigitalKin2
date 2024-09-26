import services.generate_text


def generate(custom_details: str = "") -> str:
    """
    Generates a state of the art text for an Aider developer who doesn't master Python,
    providing a global view of the latest technologies to automate Python development.

    returns: str: The generated state of the art text.
    """
    prompt = (
        f"As an Aider developer aiming to master Python automation technologies, this document provides a comprehensive "
        f"state of the art analysis {custom_details}"
    )
    introduction = services.generate_text.generate(f"{prompt}. Here is an introduction :").replace(prompt, "")
    synthesis = services.generate_text.generate(f"{prompt}. Here is a global synthesis :").replace(prompt, "")
    analysis = services.generate_text.generate(f"{prompt}. Here is a critical analysis :").replace(prompt, "")
    contribution = services.generate_text.generate(f"{prompt}. Here is a proposed contribution :").replace(prompt, "")
    conclusion = services.generate_text.generate(f"{prompt}. Here is a conclusion :").replace(prompt, "")

    structured_answer = (f"Introduction:\n"
                         f"{introduction}\n"
                         f"Global synthesis:\n"
                         f"{synthesis}\n"
                         f"Critical analysis:\n"
                         f"{analysis}\n"
                         f"Proposed contribution:\n"
                         f"{contribution}\n"
                         f"Conclusion:\n"
                         f"{conclusion}\n").replace(prompt, "")

    return structured_answer


if __name__ == "__main__":
    # Test the generate_state_of_art_for_aider_developer function
    state_of_art_text = generate()

    print(state_of_art_text)

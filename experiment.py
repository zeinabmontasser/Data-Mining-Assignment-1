# ============================================================
# experiment.py
# Part 3 & 4 — API Experiment
# ============================================================

import os
import json
from datetime import datetime
import openai

# ============================================================
# OpenAI-Compatible Groq Client (REQUIRED FORMAT)
# ============================================================

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# ============================================================
# Generate Responses
# ============================================================

def generate_responses(prompt, temperature,
                       num_responses=5,
                       top_p=None,
                       logit_bias=None):

    responses = []

    kwargs = {}
    if top_p is not None:
        kwargs["top_p"] = top_p
    if logit_bias is not None:
        kwargs["logit_bias"] = logit_bias  # keys must be STRINGS

    for _ in range(num_responses):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # REQUIRED MODEL
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **kwargs
        )

        responses.append(response.choices[0].message.content)

    return responses


# ============================================================
# Save Results
# ============================================================

def save_results(prompt, temperature, responses):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{temperature}_{timestamp}.json"

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "responses": responses
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved {filename}")


# ============================================================
# Experiment Runner
# ============================================================

if __name__ == "__main__":

    temperatures = [0.0, 0.3, 0.7, 1.0]

    prompts = [
        "Describe a world where apples are used as currency. What would the economy look like?",
        "Write a short story about a golden apple that grants infinite wealth to its owner.",
        "Create a business plan for a luxury apple orchard that caters to billionaires."
    ]

    for temp in temperatures:
        print(f"\nRunning Temperature = {temp}")

        for prompt in prompts:

            responses = generate_responses(
                prompt=prompt,
                temperature=temp,
                num_responses=5
            )

            save_results(prompt, temp, responses)

    print("\nExperiment complete.")
import openai

# set up OpenAI API key
openai.api_key = "OPENAI API KEY"

prompt = "Once upon a time"

def generate_text(prompt, temperature):
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      temperature=temperature,
      max_tokens=50,
      n=1,
      stop=None,
      frequency_penalty=0,
      presence_penalty=0
    )
    text = response.choices[0].text.strip()
    return text

# generate text with low temperature
low_temp_text = generate_text(prompt, 0.1)
print(f"Low temperature text: {low_temp_text}")

# generate text with high temperature
high_temp_text = generate_text(prompt, 2.0)
print(f"High temperature text: {high_temp_text}")
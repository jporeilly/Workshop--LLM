# import libraries
import openai
import numpy as np

# set up OpenAI API credentials
openai.api_key = "OPENAI API KEY"

# define the model and prompt
model_engine = "text-embedding-ada-002"
text_prompt = "What is the capital of France"

# encode the text prompt into a vector
response = openai.Embedding.create(
    engine=model_engine,
    input=text_prompt,
    max_tokens=256,
    temperature=0,
)
# output the response in an array
content = np.array(response.data[0].embedding)

# print the embedding vector
print("The vector for: What is the capital of France")
print(content)
import openai
import numpy as np

# Set up OpenAI API key
openai.api_key = "OPENAI API KEY"

# define the vectors to compare
vector1 = [0.5, 0.2, 0.3]
vector2 = [0.1, 0.8, 0.1]
vector3 = [0.4, 0.1, 0.5]
reference_vector = [0.2, 0.6, 0.2]

# normalize the vectors to have unit length
vector1 = np.array(vector1) / np.linalg.norm(vector1)
vector2 = np.array(vector2) / np.linalg.norm(vector2)
vector3 = np.array(vector3) / np.linalg.norm(vector3)
reference_vector = np.array(reference_vector) / np.linalg.norm(reference_vector)

# concatenate the vectors into a single list
vectors = [vector1, vector2, vector3, reference_vector]
flat_vectors = [item for sublist in vectors for item in sublist]

# set up the prompt for OpenAI
prompt = "Compare the following 3 vectors to the reference vector:\n" \
         "Vector 1: {}\n" \
         "Vector 2: {}\n" \
         "Vector 3: {}\n" \
         "Reference Vector: {}\n" \
         "Which vector is most similar to the reference vector?\n".format(
             vector1, vector2, vector3, reference_vector)

# generate a completion from OpenAI's text-davinci model
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=50,
    n=1,
    stop=None,
    temperature=0.5,
)

# print the result
result = response.choices[0].text.strip()
print("{}".format(result))

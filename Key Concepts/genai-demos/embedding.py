import openai
import json

# set up OpenAI API key
openai.api_key = "OPENAI API KEY"

# define the text you want to embed
text = "The brown fox jumped over the lazy dog."

# use the ada-002 model to embed the text
response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=text
)

# convert the embedding to a JSON string
embedding_json = json.dumps(response)

# print the JSON string to the console
print(embedding_json)
   
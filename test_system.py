import os
from google import genai
from dotenv import load_dotenv

# 1. Load variables from your .env file
load_dotenv()

# 2. Initialize the client using the modern GenAI SDK
client = genai.Client()

# 3. Create a conversation interaction using a modern supported model
response = client.interactions.create(
    model="gemini-3.7-flash",  # Swapped from legacy 1.5
    input="What are three interesting facts about space?"
)

# 4. Print the text response from the model
print(response.output_text)

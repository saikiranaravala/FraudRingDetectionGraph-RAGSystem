from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url='https://openrouter.ai/api/v1'
)

resp = client.chat.completions.create(
    model='google/gemma-4-31b-it:free',
    messages=[{'role': 'user', 'content': 'Say OK'}],
    max_tokens=10
)
print('Gemma response:', resp.choices[0].message.content)

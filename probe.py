from openai import OpenAI
import openai, os

print("openai version:", openai.__version__)
print("key loaded?", bool(os.environ.get("OPENAI_API_KEY")))

client = OpenAI(timeout=90)  # puja timeout
r = client.chat.completions.create(
    model="gpt-4.1-mini",  # o gpt-4o-mini
    messages=[{"role":"user","content":"ping"}]
)
print("reply:", r.choices[0].message.content[:60])

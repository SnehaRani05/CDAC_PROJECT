from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Reply with OK only"}
    ],
    temperature=0
)

print(response.choices[0].message.content)

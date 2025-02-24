import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

# Use stream=True for streaming responses
response = client.chat.completions.create(
    model="/data2/lmzheng/grok-1/",
    messages=[
        {"role": "user", "content": "List 5 fruits with most vitamin C."},
    ],
    temperature=0,
    max_tokens=1024,
    stream=True,
)

# Handle the streaming output
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

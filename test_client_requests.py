import requests

response = requests.post(
    f"http://localhost:30000/generate",
    json={
        "text": "List 10 countries and their capitals.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)

print(response.json())
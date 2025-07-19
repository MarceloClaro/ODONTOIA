import requests

def consulta_groq(texto, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama3-70b-8192",  # ou outro modelo suportado
        "messages": [{"role": "user", "content": texto}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

import requests
import os

# Carregue sua chave da API de forma segura
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"  # ou outro modelo disponível na Groq

def consulta_groq(prompt, temperature=0.7, max_tokens=1024, model=GROQ_MODEL):
    """
    Envia uma consulta (prompt) para a API Groq LLM e retorna a resposta.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data['choices'][0]['message']['content']

def interpretar_predicao(predicao, contexto_extra=None):
    """
    Gera um prompt para interpretação clínica baseada na predição do modelo.
    """
    prompt = f"Interprete o seguinte resultado de exame odontológico: '{predicao}'."
    if contexto_extra:
        prompt += f" Contexto adicional: {contexto_extra}"
    return consulta_groq(prompt)

def gerar_prognostico(predicao, contexto_extra=None):
    """
    Gera um prompt para obter prognóstico baseado na predição.
    """
    prompt = f"Considere o seguinte diagnóstico: '{predicao}'. Forneça um prognóstico detalhado e possíveis condutas clínicas."
    if contexto_extra:
        prompt += f" Contexto adicional: {contexto_extra}"
    return consulta_groq(prompt)

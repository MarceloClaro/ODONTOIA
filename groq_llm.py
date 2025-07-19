import requests
import os
import streamlit as st

# Carregue sua chave da API de forma segura a partir dos segredos do Streamlit
# Isso busca por GROQ_API_KEY no seu ambiente de deploy (Secrets)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

# Verifique se a chave foi carregada
if not GROQ_API_KEY:
    # Esta mensagem será exibida se a chave não for encontrada nos segredos do Streamlit Cloud
    st.error("A chave da API Groq (GROQ_API_KEY) não foi encontrada nos segredos. Por favor, configure-a no painel do Streamlit Cloud.")
    # Interrompe a execução para evitar mais erros
    st.stop()

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
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        # Levanta um erro para respostas com status de falha (4xx ou 5xx)
        response.raise_for_status() 
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        # Captura erros de rede, timeout, etc., e exibe uma mensagem amigável
        st.error(f"Erro de comunicação com a API Groq: {e}")
        return "Não foi possível obter uma resposta da IA no momento devido a um erro de conexão."

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

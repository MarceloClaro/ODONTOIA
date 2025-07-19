import requests
import os
import streamlit as st
import json

# Carregue sua chave da API de forma segura a partir dos segredos do Streamlit
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

# Verifique se a chave foi carregada
if not GROQ_API_KEY:
    st.error("A chave da API Groq (GROQ_API_KEY) não foi encontrada nos segredos. Por favor, configure-a no painel do Streamlit Cloud.")
    st.stop()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

def consulta_groq(prompt, temperature=0.7, max_tokens=1024, model=GROQ_MODEL):
    """
    Envia uma consulta (prompt) para a API Groq LLM e retorna a resposta,
    com tratamento de erro aprimorado.
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
        
        # Verificação de segurança para a estrutura da resposta
        if 'choices' not in data or not data['choices'] or 'message' not in data['choices'][0] or 'content' not in data['choices'][0]['message']:
            st.error("Resposta da API Groq com formato inesperado.")
            return "Não foi possível processar a resposta da IA."
            
        return data['choices'][0]['message']['content']

    except requests.exceptions.HTTPError as http_err:
        # Erro específico de HTTP (ex: 401 Unauthorized, 404 Not Found)
        error_message = f"Erro HTTP: {http_err.response.status_code} {http_err.response.reason}"
        # Tenta extrair a mensagem de erro detalhada do corpo da resposta da API
        try:
            error_details = http_err.response.json()
            detailed_error = error_details.get('error', {}).get('message', 'Nenhum detalhe adicional fornecido.')
            error_message += f"\nDetalhes da API: {detailed_error}"
        except json.JSONDecodeError:
            error_message += "\nDetalhes: Não foi possível decodificar a resposta de erro da API."
        
        st.error(error_message)
        return "Falha ao consultar a IA. Verifique os logs de erro acima."

    except requests.exceptions.RequestException as e:
        # Erros de conexão, timeout, etc.
        st.error(f"Erro de comunicação com a API Groq: {e}")
        return "Não foi possível obter uma resposta da IA devido a um erro de conexão."
    
    except (KeyError, IndexError) as e:
        # Erro se a estrutura do JSON de sucesso for inesperada
        st.error(f"A resposta da API foi recebida, mas seu formato é inválido: {e}")
        return "Falha ao processar a resposta da IA."


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

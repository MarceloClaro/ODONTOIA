# Documentação das Novas Funcionalidades - ODONTOIA

## Resumo das Mudanças

Este documento descreve as novas funcionalidades adicionadas ao módulo `llm_modal.py` para resolver os problemas de tradução e análise de referências acadêmicas.

## Funcionalidades Implementadas

### 1. Integração com Semantic Scholar API

**Método:** `search_semantic_scholar(query, max_results=5)`

- Busca artigos acadêmicos na base de dados Semantic Scholar
- Retorna informações detalhadas incluindo:
  - Título, autores, ano de publicação
  - Resumo (abstract)
  - Número de citações
  - DOI e arXiv ID quando disponíveis
  - URL para acesso ao artigo
  - Metadados de auditoria

**Exemplo de uso:**
```python
ref = DentalDiseaseReference()
articles = ref.search_semantic_scholar("deep learning", max_results=5)
```

### 2. Integração com arXiv API

**Método:** `search_arxiv(query, max_results=5)`

- Busca preprints acadêmicos no repositório arXiv
- Retorna informações incluindo:
  - Título, autores, ano
  - Resumo completo
  - arXiv ID
  - Links para visualização e download PDF
  - Metadados de auditoria

**Exemplo de uso:**
```python
articles = ref.search_arxiv("neural networks", max_results=5)
```

### 3. Tradução Automática para Português

**Método:** `translate_to_portuguese(text)`

- Traduz textos acadêmicos do inglês para português brasileiro
- Utiliza o LLM Groq para tradução contextual
- Mantém terminologia técnica apropriada
- Limita texto a 1500 caracteres para eficiência

**Características:**
- Temperature: 0.3 (para tradução mais literal e precisa)
- Max tokens: 800
- Tratamento de erros robusto

**Exemplo de uso:**
```python
english_text = "Deep neural networks have shown remarkable performance..."
portuguese_text = ref.translate_to_portuguese(english_text)
```

### 4. Geração de Resenha Crítica

**Método:** `generate_critical_review(article)`

- Gera resenha crítica de artigos científicos
- Análise inclui:
  1. Síntese dos objetivos e métodos
  2. Pontos fortes do estudo
  3. Limitações potenciais
  4. Relevância para a área

**Características:**
- Temperature: 0.7 (para análise mais criativa)
- Max tokens: 600
- Formato: 3-4 parágrafos concisos em português

**Exemplo de uso:**
```python
article = {
    "title": "Deep Learning in Medicine",
    "abstract": "This study explores..."
}
review = ref.generate_critical_review(article)
```

### 5. Análise Multi-Perspectiva com Algoritmos Genéticos

**Método:** `multi_perspective_genetic_analysis(articles)`

- Realiza análise sofisticada usando algoritmos genéticos
- Avalia artigos sob múltiplas perspectivas:
  - Metodologia Experimental
  - Relevância Clínica
  - Inovação Tecnológica
  - Aplicabilidade Prática
  - Rigor Científico
  - Impacto na Literatura

**Algoritmo:**
- População: 20 indivíduos
- Gerações: 50 iterações
- Operações: Seleção, crossover, mutação
- Fitness baseado em citações, ano, completude do abstract

**Saída inclui:**
- Scores de fitness por artigo
- Pontuações detalhadas por perspectiva
- Síntese inteligente gerada por LLM
- Visualizações gráficas

**Exemplo de uso:**
```python
ga_results = ref.multi_perspective_genetic_analysis(articles)
if ga_results['success']:
    print(f"Analisados {ga_results['articles_analyzed']} artigos")
    print(f"Síntese: {ga_results['synthesis']}")
```

## Interface Atualizada

### Nova Aba: "📚 Referências Científicas"

Substitui a antiga aba "Referências PubMed" e agora inclui:

1. **Múltiplas Fontes:**
   - Semantic Scholar (3 artigos)
   - arXiv (3 artigos)
   - PubMed (2 artigos)

2. **Para Cada Artigo:**
   - Informações bibliográficas completas
   - Resumo original em inglês (expansível)
   - **NOVO:** Resumo traduzido em português
   - **NOVO:** Resenha crítica automática
   - Identificadores (DOI, PMID, arXiv ID)
   - Links de acesso
   - Metadados de auditoria completos

### Nova Aba: "🧬 Análise Multi-Perspectiva"

Apresenta resultados da análise genética:

1. **Parâmetros da Análise:**
   - Número de artigos analisados
   - Gerações evolutivas
   - Tamanho da população

2. **Perspectivas Avaliadas:**
   - Lista de todas as 6 perspectivas

3. **Resultados por Artigo:**
   - Score de fitness
   - Gráficos de barras por perspectiva
   - Comparação visual

4. **Síntese Inteligente:**
   - Análise consolidada gerada por LLM
   - Insights sobre metodologia, relevância e inovação

## Fluxo de Funcionamento

```
1. Usuário avalia imagem → Predição da classe
2. Sistema busca referências em 3 bases de dados
3. Para cada artigo:
   a. Exibe informações originais
   b. Traduz resumo para português
   c. Gera resenha crítica
4. Executa análise genética multi-perspectiva
5. Apresenta síntese inteligente
```

## Requisitos de API

### Groq API (LLM)
- **Variável de ambiente:** `GROQ_API_KEY`
- **Uso:** Tradução e geração de resenhas
- **Configuração:** Secrets do Streamlit

### Semantic Scholar API
- **Endpoint:** `https://api.semanticscholar.org/graph/v1/paper/search`
- **Autenticação:** Não requerida (API pública)
- **Rate limits:** Respeita limites padrão

### arXiv API
- **Endpoint:** `http://export.arxiv.org/api/query`
- **Autenticação:** Não requerida (API pública)
- **Rate limits:** Respeita limites padrão

## Tratamento de Erros

Todas as funções incluem tratamento robusto de erros:

- Timeouts de 15 segundos para requisições HTTP
- Validação de respostas API
- Fallbacks para dados indisponíveis
- Mensagens de erro claras para o usuário
- Continuação do fluxo mesmo com falhas parciais

## Melhorias de Performance

1. **Requisições Assíncronas:** Cada API é chamada em paralelo
2. **Caching Implícito:** Resultados são armazenados durante a sessão
3. **Limitação de Texto:** Abstracts limitados para tradução eficiente
4. **Spinners Informativos:** Feedback visual durante operações longas

## Testes

Execute o script de testes:
```bash
python /tmp/test_new_features.py
```

Verifica:
- Inicialização da classe
- Existência de todos os métodos
- Recuperação de informações de doenças
- Estrutura de tradução
- Estrutura de análise genética

## Resolução dos Problemas Originais

| Problema Original | Solução Implementada |
|------------------|---------------------|
| ❌ Não busca Semantic Scholar | ✅ Integração completa com API |
| ❌ Não busca arXiv | ✅ Integração completa com API |
| ❌ Resumos não traduzidos | ✅ Tradução automática com LLM |
| ❌ Sem resenha crítica | ✅ Geração automática de resenhas |
| ❌ Sem análise multi-perspectiva | ✅ Algoritmos genéticos implementados |
| ❌ Erro modelo Gemini | ✅ Usa Groq LLM (já configurado) |

## Notas Importantes

1. **API Keys:** Certifique-se de que `GROQ_API_KEY` está configurada nos Secrets do Streamlit
2. **Conexão Internet:** Todas as funcionalidades requerem conexão ativa
3. **Rate Limits:** Respeite os limites das APIs públicas
4. **Idioma:** Todo o sistema agora trabalha primariamente em português brasileiro

## Suporte e Manutenção

Para questões ou problemas:
1. Verifique logs de erro no console do Streamlit
2. Confirme conectividade com APIs externas
3. Valide formato de dados retornados pelas APIs
4. Teste com diferentes termos de busca

## Exemplos de Saída

### Exemplo de Tradução:
**Original (EN):** "Deep neural networks have excelled on a wide range of problems..."
**Traduzido (PT):** "Redes neurais profundas têm se destacado em uma ampla gama de problemas..."

### Exemplo de Resenha Crítica:
```
O artigo apresenta uma abordagem inovadora para adaptação de parâmetros 
baseada em memória. Os pontos fortes incluem a metodologia experimental 
robusta e resultados impressionantes em múltiplos domínios. No entanto, 
limitações incluem a necessidade de grandes volumes de memória e possível 
dificuldade de generalização para domínios muito distintos...
```

### Exemplo de Análise Genética:
```
Perspectivas Destacadas:
- Metodologia Experimental: 0.87
- Rigor Científico: 0.92
- Impacto na Literatura: 0.95

Síntese: Os artigos analisados demonstram forte rigor metodológico e 
significativo impacto na literatura, com particular destaque para 
abordagens de deep learning aplicadas à medicina...
```

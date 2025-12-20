# Guia de Uso - Novas Funcionalidades ODONTOIA

## 🎯 Objetivo

Este guia explica como usar as novas funcionalidades de tradução e análise multi-perspectiva implementadas no ODONTOIA.

## 🚀 Como Usar

### 1. Realizar Predição de Imagem

1. Na aba **"Avaliação de Imagem"**:
   - Faça upload de uma imagem odontológica
   - O sistema irá predizer a classe (ex: Aftas, Herpes Labial, etc.)

### 2. Acessar Informações Acadêmicas

Após a predição, o sistema automaticamente exibe informações acadêmicas em 4 abas:

#### 📋 Aba 1: Descrição Clínica
- Informações médicas sobre a doença
- Sintomas, causas e tratamentos
- Baseado em literatura especializada

#### 📚 Aba 2: Referências Científicas (NOVA!)

**Busca Automática em Múltiplas Bases:**
- 🔬 Semantic Scholar (3 artigos)
- 📄 arXiv (3 artigos)  
- 🏥 PubMed (2 artigos)

**Para Cada Artigo, Você Verá:**

1. **Informações Bibliográficas**
   ```
   📖 Título do Artigo
   👥 Autores: Nome dos autores
   📅 Ano: 2023
   📰 Periódico: Nome da revista
   🏛️ Plataforma: Semantic Scholar
   📊 Citações: 102
   ```

2. **Resumo Original (Expansível)**
   - Clique para ver o abstract em inglês

3. **✨ Resumo Traduzido (NOVO!)**
   ```
   📝 Resumo (Português)
   [Tradução automática do abstract para português brasileiro]
   ```

4. **✨ Resenha Crítica (NOVO!)**
   ```
   📋 Resenha Crítica
   [Análise crítica incluindo:]
   - Síntese dos objetivos e métodos
   - Pontos fortes do estudo
   - Limitações potenciais
   - Relevância para a área
   ```

5. **Identificadores e Links**
   ```
   🔗 Identificadores:
   DOI: 10.xxxx/xxxxx
   PMID: 12345678
   arXiv ID: 1802.10542

   🌐 Links de Acesso:
   📄 [Visualizar Artigo]
   ⬇️ [Download PDF]
   ```

6. **Auditoria**
   ```
   🔐 Informações de Auditoria:
   Data de Recuperação: 2025-12-20T11:13:45
   Ranking na Busca: #1
   Base de Dados: Semantic Scholar API
   ```

#### 🤖 Aba 3: Análise LLM
- Descrição detalhada gerada por IA
- Insights clínicos baseados em IA

#### 🧬 Aba 4: Análise Multi-Perspectiva (NOVA!)

**Análise com Algoritmos Genéticos:**

1. **Parâmetros da Análise**
   ```
   📊 Parâmetros da Análise Genética
   - Artigos Analisados: 6
   - Gerações Evolutivas: 50
   - Tamanho da População: 20
   ```

2. **Perspectivas Avaliadas**
   ```
   🎯 Perspectivas Avaliadas
   • Metodologia Experimental
   • Relevância Clínica
   • Inovação Tecnológica
   • Aplicabilidade Prática
   • Rigor Científico
   • Impacto na Literatura
   ```

3. **Resultados por Artigo**
   - Clique em cada artigo para ver:
     - Score de Fitness
     - Gráfico de barras com pontuações por perspectiva
     - Análise visual comparativa

4. **Síntese Inteligente**
   ```
   🎓 Síntese Inteligente
   [Análise consolidada gerada por IA sobre:]
   - Metodologia e rigor científico
   - Relevância clínica
   - Inovação e impacto
   ```

## 📋 Requisitos

### Configuração Necessária

1. **API Key Groq**
   - Configure `GROQ_API_KEY` nos Secrets do Streamlit
   - Usado para tradução e análise crítica

2. **Conexão Internet**
   - Necessária para acessar:
     - Semantic Scholar API
     - arXiv API
     - PubMed API
     - Groq LLM API

## 🎓 Exemplo de Fluxo Completo

```
1. Upload de imagem → Predição: "Aftas"
   ↓
2. Sistema busca automaticamente:
   - 3 artigos no Semantic Scholar
   - 3 artigos no arXiv
   - 2 artigos no PubMed
   ↓
3. Para cada artigo:
   ✓ Mostra informações completas
   ✓ Traduz resumo para português
   ✓ Gera resenha crítica
   ↓
4. Executa análise genética:
   ✓ Avalia 6 perspectivas
   ✓ 50 gerações evolutivas
   ✓ Gera síntese inteligente
   ↓
5. Apresenta resultados com gráficos
```

## 💡 Dicas de Uso

### Para Melhor Experiência:

1. **Aguarde o Carregamento**
   - As buscas e traduções levam alguns segundos
   - Spinners indicam progresso

2. **Explore as Abas**
   - Cada aba oferece insights diferentes
   - Use expansores para detalhes

3. **Verifique Identificadores**
   - Use DOI/PMID para citações formais
   - Links levam aos artigos originais

4. **Análise Genética**
   - Observe os gráficos de perspectiva
   - Leia a síntese inteligente ao final

## 🐛 Resolução de Problemas

### "Erro ao buscar..."
- ✓ Verifique conexão internet
- ✓ APIs públicas podem ter rate limits
- ✓ Tente novamente após alguns segundos

### "Erro na tradução..."
- ✓ Verifique se GROQ_API_KEY está configurada
- ✓ Verifique saldo da API Groq
- ✓ Sistema continuará funcionando (mostrará texto original)

### "Nenhum artigo disponível..."
- ✓ Termo de busca pode não ter resultados
- ✓ APIs podem estar temporariamente indisponíveis
- ✓ Sistema tentará outras bases de dados

## 📊 Métricas de Sucesso

### O que mudou:

| Antes | Depois |
|-------|--------|
| ❌ Apenas PubMed | ✅ 3 bases de dados |
| ❌ Resumos em inglês | ✅ Tradução automática |
| ❌ Sem análise crítica | ✅ Resenha por artigo |
| ❌ Sem análise multi-perspectiva | ✅ Algoritmos genéticos |
| ⚠️ Erro modelo Gemini | ✅ Usa Groq (estável) |

## 🎯 Principais Benefícios

1. **🌍 Acessibilidade**
   - Resumos em português facilitam compreensão
   - Não precisa mais traduzir manualmente

2. **📚 Múltiplas Fontes**
   - Visão mais ampla da literatura
   - Preprints (arXiv) + peer-reviewed (PubMed)

3. **🔍 Análise Profunda**
   - Resenhas críticas automáticas
   - Avaliação multi-perspectiva

4. **🧬 Insights Únicos**
   - Algoritmos genéticos revelam padrões
   - Síntese inteligente por IA

## 📞 Suporte

Para dúvidas ou problemas:
1. Consulte `DOCS_NEW_FEATURES.md` para detalhes técnicos
2. Verifique logs do Streamlit
3. Confirme configuração de API keys

## ✅ Checklist Rápido

Antes de usar, certifique-se:
- [ ] GROQ_API_KEY configurada
- [ ] Conexão internet ativa
- [ ] Streamlit rodando corretamente
- [ ] Imagem carregada e predição feita

## 🎉 Aproveite!

Agora você tem acesso a:
- ✅ 8 artigos por busca (vs 5 antes)
- ✅ Traduções automáticas
- ✅ Resenhas críticas
- ✅ Análise genética multi-perspectiva
- ✅ Visualizações gráficas

**Todo em português e automatizado!** 🚀

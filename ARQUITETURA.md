# Arquitetura das Novas Funcionalidades

```
┌─────────────────────────────────────────────────────────────────┐
│                    ODONTOIA - Fluxo de Análise                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Upload Imagem  │
│   Odontológica  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Predição ML   │
│  (ResNet/Dense) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              show_disease_modal(class_name)                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ├──────────────────┬──────────────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Aba 1: 📋   │   │  Aba 2: 📚   │  │  Aba 3: 🤖   │  │  Aba 4: 🧬   │
│  Descrição   │   │ Referências  │  │  Análise LLM │  │   Análise    │
│   Clínica    │   │  Científicas │  │              │  │   Genética   │
└──────────────┘   └──────┬───────┘  └──────────────┘  └──────┬───────┘
                          │                                    │
                          │                                    │
         ┌────────────────┴────────────────┐                  │
         ▼                 ▼                ▼                  │
┌─────────────────┐ ┌─────────────┐ ┌──────────────┐          │
│ search_semantic │ │ search_arxiv│ │search_pubmed │          │
│    _scholar     │ │             │ │              │          │
│                 │ │             │ │              │          │
│ 🔬 3 artigos    │ │ 📄 3 artigos│ │ 🏥 2 artigos │          │
└────────┬────────┘ └──────┬──────┘ └──────┬───────┘          │
         │                 │                │                  │
         └─────────────────┴────────────────┘                  │
                           │                                   │
                           ▼                                   │
                  ┌─────────────────┐                          │
                  │  all_articles   │                          │
                  │   (List[Dict])  │◄─────────────────────────┘
                  └────────┬────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌──────────────────┐ ┌────────────────┐ ┌─────────────────────┐
│  Para cada       │ │  Para cada     │ │multi_perspective_   │
│  artigo:         │ │  artigo:       │ │genetic_analysis()   │
│                  │ │                │ │                     │
│ translate_to_    │ │ generate_      │ │ 🧬 Algoritmo        │
│ portuguese()     │ │ critical_      │ │    Genético         │
│                  │ │ review()       │ │                     │
│ 🌍 Resumo PT     │ │ 📋 Resenha     │ │ • População: 20     │
│                  │ │    Crítica     │ │ • Gerações: 50      │
└──────────────────┘ └────────────────┘ │ • 6 Perspectivas    │
                                        └──────────┬──────────┘
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │  Síntese LLM        │
                                        │  (consulta_groq)    │
                                        └─────────────────────┘

═══════════════════════════════════════════════════════════════════
                          APIs Utilizadas
═══════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│ Semantic Scholar API                                              │
│ • URL: api.semanticscholar.org/graph/v1/paper/search            │
│ • Dados: title, authors, abstract, citations, DOI, arXiv ID     │
│ • Sem autenticação                                               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ arXiv API                                                         │
│ • URL: export.arxiv.org/api/query                                │
│ • Dados: title, authors, abstract, PDF link, arXiv ID           │
│ • Sem autenticação                                               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ PubMed E-utilities                                                │
│ • URL: eutils.ncbi.nlm.nih.gov/entrez/eutils/                   │
│ • Dados: title, authors, abstract, PMID, MeSH terms             │
│ • Sem autenticação                                               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Groq LLM API                                                      │
│ • URL: api.groq.com/openai/v1/chat/completions                  │
│ • Modelo: llama3-70b-8192                                        │
│ • Uso: Tradução, Resenhas Críticas, Sínteses                    │
│ • Requer: GROQ_API_KEY                                           │
└──────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
                    Estrutura de Dados
═══════════════════════════════════════════════════════════════════

Article Object:
{
    "title": str,
    "authors": str,
    "year": str,
    "journal": str,
    "abstract": str,
    "citations": int (opcional),
    "doi": str (opcional),
    "pmid": str (opcional),
    "arxiv_id": str (opcional),
    "url": str,
    "pdf_url": str (opcional),
    "platform": str ("Semantic Scholar" | "arXiv" | "PubMed"),
    "relevance": str ("High"),
    "ranking": int,
    "retrieved_date": str (ISO format)
}

GA Results Object:
{
    "success": bool,
    "articles_analyzed": int,
    "generations": int,
    "population_size": int,
    "perspectives": List[str],
    "results": List[{
        "article": str,
        "fitness": float,
        "perspectives": Dict[str, float]
    }],
    "synthesis": str
}

═══════════════════════════════════════════════════════════════════
                    Fluxo de Processamento
═══════════════════════════════════════════════════════════════════

1. Busca em Paralelo (3 APIs)
   ↓
2. Agregação de Resultados (all_articles)
   ↓
3. Loop por Artigo:
   ├─→ Tradução (PT-BR)
   ├─→ Resenha Crítica
   └─→ Exibição
   ↓
4. Análise Genética:
   ├─→ Inicialização População
   ├─→ 50 Gerações (Seleção, Crossover, Mutação)
   ├─→ Avaliação Fitness
   └─→ Síntese LLM
   ↓
5. Visualização Resultados

═══════════════════════════════════════════════════════════════════
                 Tratamento de Erros
═══════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│ Cada função tem try-except para:                                 │
│                                                                   │
│ • RequestException → Exibe erro de conexão                       │
│ • Timeout (15s) → Cancela operação                               │
│ • ParseError → Exibe erro de formato                             │
│ • Exception geral → Log e continua                               │
│                                                                   │
│ Fallbacks:                                                        │
│ • Texto não traduzido → Mostra original                          │
│ • Resenha falha → Mensagem informativa                           │
│ • API indisponível → Tenta outras fontes                         │
└──────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
                Performance Considerations
═══════════════════════════════════════════════════════════════════

• Busca em paralelo: ~5-10 segundos
• Tradução por artigo: ~3-5 segundos
• Resenha por artigo: ~3-5 segundos
• Análise genética: ~2-3 segundos
• Total estimado: 30-60 segundos para análise completa

Otimizações:
✓ Limites de texto (1500 chars para tradução)
✓ Spinners para feedback visual
✓ Caching implícito na sessão
✓ Timeouts adequados
```

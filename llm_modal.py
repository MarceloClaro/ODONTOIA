"""
LLM Modal for ODONTOIA - Dental Disease Description and Classification
This module provides LLM-powered descriptions and academic references for oral diseases.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime
import random
from groq_llm import consulta_groq

class DentalDiseaseReference:
    """Class to handle dental disease descriptions and PubMed references"""
    
    def __init__(self):
        self.disease_info = {
            "gangivoestomatite": {
                "name": "Gangivoestomatite (Gengivite)",
                "medical_name": "Gingivostomatitis",
                "description": "Inflamação das gengivas e mucosa oral, frequentemente causada por infecções virais, bacterianas ou fúngicas.",
                "symptoms": ["Vermelhidão e inchaço das gengivas", "Dor ao mastigar", "Sangramento gengival", "Úlceras orais"],
                "causes": ["Vírus herpes simplex", "Candidíase", "Má higiene oral", "Deficiência nutricional"],
                "treatment": ["Antissépticos orais", "Analgésicos", "Antivirais (se viral)", "Melhoria da higiene oral"]
            },
            "aftas": {
                "name": "Aftas (Estomatite Aftosa)",
                "medical_name": "Aphthous Stomatitis",
                "description": "Úlceras benignas recorrentes da mucosa oral, caracterizadas por lesões dolorosas com bordas bem definidas.",
                "symptoms": ["Úlceras circulares ou ovais", "Dor intensa", "Bordas vermelhas com centro esbranquiçado", "Dificuldade para comer"],
                "causes": ["Fatores genéticos", "Estresse", "Deficiências nutricionais", "Traumatismo local", "Alterações hormonais"],
                "treatment": ["Corticosteroides tópicos", "Analgésicos", "Protetores de mucosa", "Suplementação nutricional"]
            },
            "herpes_labial": {
                "name": "Herpes Labial",
                "medical_name": "Herpes Simplex Labialis",
                "description": "Infecção viral recorrente causada pelo vírus herpes simplex, manifestando-se principalmente nos lábios.",
                "symptoms": ["Vesículas nos lábios", "Sensação de queimação", "Prurido", "Crostas após rompimento das vesículas"],
                "causes": ["Vírus herpes simplex tipo 1", "Estresse", "Exposição solar", "Imunossupressão"],
                "treatment": ["Antivirais tópicos", "Antivirais sistêmicos", "Analgésicos", "Proteção solar"]
            },
            "liquen_plano_oral": {
                "name": "Líquen Plano Oral",
                "medical_name": "Oral Lichen Planus",
                "description": "Doença inflamatória crônica que afeta a mucosa oral, caracterizada por lesões reticulares ou erosivas.",
                "symptoms": ["Estrias esbranquiçadas (estrias de Wickham)", "Erosões dolorosas", "Sensação de queimação", "Dificuldade para comer alimentos ácidos"],
                "causes": ["Doença autoimune", "Estresse", "Medicamentos", "Materiais dentários"],
                "treatment": ["Corticosteroides", "Imunossupressores", "Retinoides", "Controle de fatores desencadeantes"]
            },
            "candidíase_oral": {
                "name": "Candidíase Oral (Sapinho)",
                "medical_name": "Oral Candidiasis",
                "description": "Infecção fúngica da cavidade oral causada principalmente pela Candida albicans.",
                "symptoms": ["Placas esbranquiçadas removíveis", "Vermelhidão da mucosa", "Sensação de queimação", "Alteração do paladar"],
                "causes": ["Imunossupressão", "Antibióticos de amplo espectro", "Diabetes", "Próteses mal adaptadas"],
                "treatment": ["Antifúngicos tópicos", "Antifúngicos sistêmicos", "Controle de fatores predisponentes", "Melhoria da higiene oral"]
            },
            "cancer_boca": {
                "name": "Câncer de Boca",
                "medical_name": "Oral Cancer",
                "description": "Neoplasia maligna que pode afetar qualquer estrutura da cavidade oral, sendo o carcinoma espinocelular o tipo mais comum.",
                "symptoms": ["Lesões que não cicatrizam", "Nódulos ou espessamentos", "Dor persistente", "Dificuldade para deglutir", "Sangramento"],
                "causes": ["Tabagismo", "Etilismo", "Exposição solar (lábios)", "HPV", "Irritação crônica"],
                "treatment": ["Cirurgia", "Radioterapia", "Quimioterapia", "Terapia direcionada", "Imunoterapia"]
            },
            "cancer_oral": {
                "name": "Câncer Oral",
                "medical_name": "Oral Carcinoma",
                "description": "Neoplasia maligna das estruturas orais, incluindo língua, assoalho da boca, palato e outras regiões.",
                "symptoms": ["Úlceras persistentes", "Leucoplasias", "Eritroplasias", "Mobilidade dentária", "Parestesia"],
                "causes": ["Fatores genéticos", "Carcinógenos ambientais", "Infecções virais", "Traumatismo crônico"],
                "treatment": ["Ressecção cirúrgica", "Radioterapia adjuvante", "Quimioterapia neoadjuvante", "Cuidados paliativos"]
            }
        }
    
    def get_disease_info(self, disease_key: str) -> Dict:
        """Get comprehensive information about a dental disease"""
        return self.disease_info.get(disease_key, {})
    
    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search PubMed for academic references with advanced metadata."""
        try:
            # PubMed E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            
            # First, search for article IDs
            search_url = f"{base_url}esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "xml",
                "sort": "relevance"
            }
            
            response = requests.get(search_url, params=search_params, timeout=15)
            response.raise_for_status()
            
            # Parse XML to get PMIDs
            root = ET.fromstring(response.content)
            pmids = [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text is not None]
            
            if not pmids:
                return []
            
            # Get article details
            fetch_url = f"{base_url}efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }
            
            response = requests.get(fetch_url, params=fetch_params, timeout=15)
            response.raise_for_status()
            
            # Parse article details
            articles = []
            root = ET.fromstring(response.content)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None and title_elem.text is not None else "No title available"

                    authors_list = []
                    for author in article.findall(".//Author"):
                        lastname_elem = author.find(".//LastName")
                        forename_elem = author.find(".//ForeName")
                        if lastname_elem is not None and lastname_elem.text and forename_elem is not None and forename_elem.text:
                            authors_list.append(f"{forename_elem.text} {lastname_elem.text}")
                    
                    authors_str = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")

                    journal_elem = article.find(".//Title")
                    journal = journal_elem.text if journal_elem is not None and journal_elem.text is not None else "N/A"

                    year_elem = article.find(".//PubDate/Year")
                    year = year_elem.text if year_elem is not None and year_elem.text is not None else "N/A"

                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text is not None else ""

                    abstract_elem = article.find(".//AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text is not None else "No abstract available"

                    # Extract Publication Types
                    pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]

                    # Extract MeSH Terms
                    mesh_terms = []
                    for mesh in article.findall(".//MeshHeading"):
                        descriptor = mesh.find(".//DescriptorName")
                        if descriptor is not None and descriptor.text:
                            mesh_terms.append(descriptor.text)

                    articles.append({
                        "title": title,
                        "authors": authors_str,
                        "journal": journal,
                        "year": year,
                        "pmid": pmid,
                        "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#",
                        "pub_types": pub_types,
                        "mesh_terms": mesh_terms
                    })
                    
                except Exception:
                    # Skip article if there is any parsing error
                    continue
            
            return articles
            
        except requests.exceptions.RequestException as e:
            st.error(f"Erro de conexão ao buscar no PubMed: {e}")
            return []
        except ET.ParseError as e:
            st.error(f"Erro ao processar dados do PubMed: {e}")
            return []
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")
            return []
    
    def search_semantic_scholar(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Semantic Scholar for academic references."""
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,citationCount,venue,externalIds,url"
            }
            
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for i, paper in enumerate(data.get('data', []), 1):
                authors_list = [author.get('name', '') for author in paper.get('authors', [])]
                authors_str = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
                
                external_ids = paper.get('externalIds', {})
                
                articles.append({
                    "title": paper.get('title', 'No title available'),
                    "authors": authors_str,
                    "journal": paper.get('venue', 'N/A'),
                    "year": paper.get('year', 'N/A'),
                    "abstract": paper.get('abstract', 'No abstract available'),
                    "citations": paper.get('citationCount', 0),
                    "doi": external_ids.get('DOI', ''),
                    "arxiv_id": external_ids.get('ArXiv', ''),
                    "url": paper.get('url', '#'),
                    "platform": "Semantic Scholar",
                    "relevance": "High",
                    "ranking": i,
                    "retrieved_date": datetime.now().isoformat()
                })
            
            return articles
            
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao buscar no Semantic Scholar: {e}")
            return []
        except Exception as e:
            st.error(f"Erro inesperado no Semantic Scholar: {e}")
            return []
    
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search arXiv for academic preprints."""
        try:
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            articles = []
            for i, entry in enumerate(root.findall('atom:entry', ns), 1):
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title"
                
                authors_list = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None and name_elem.text:
                        authors_list.append(name_elem.text)
                
                authors_str = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
                
                summary_elem = entry.find('atom:summary', ns)
                abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else "No abstract"
                
                published_elem = entry.find('atom:published', ns)
                year = published_elem.text[:4] if published_elem is not None and published_elem.text else "N/A"
                
                id_elem = entry.find('atom:id', ns)
                arxiv_url = id_elem.text if id_elem is not None and id_elem.text else "#"
                arxiv_id = arxiv_url.split('/abs/')[-1] if '/abs/' in arxiv_url else ""
                
                pdf_link = ""
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_link = link.get('href', '')
                        break
                
                articles.append({
                    "title": title,
                    "authors": authors_str,
                    "journal": "arXiv preprint",
                    "year": year,
                    "abstract": abstract,
                    "arxiv_id": arxiv_id,
                    "url": arxiv_url,
                    "pdf_url": pdf_link,
                    "platform": "arXiv",
                    "relevance": "High",
                    "ranking": i,
                    "retrieved_date": datetime.now().isoformat()
                })
            
            return articles
            
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao buscar no arXiv: {e}")
            return []
        except ET.ParseError as e:
            st.error(f"Erro ao processar dados do arXiv: {e}")
            return []
        except Exception as e:
            st.error(f"Erro inesperado no arXiv: {e}")
            return []
    
    def translate_to_portuguese(self, text: str) -> str:
        """Translate text to Portuguese using LLM."""
        try:
            if not text or text == "No abstract available" or text == "No abstract":
                return "Resumo não disponível"
            
            prompt = f"""Traduza o seguinte texto acadêmico para português (Brasil) mantendo a terminologia técnica adequada:

{text[:1500]}

Forneça APENAS a tradução, sem comentários adicionais."""
            
            translation = consulta_groq(prompt, temperature=0.3, max_tokens=800)
            return translation
        except Exception as e:
            st.warning(f"Erro na tradução: {e}")
            return text
    
    def generate_critical_review(self, article: Dict) -> str:
        """Generate a critical review of an article using LLM."""
        try:
            abstract = article.get('abstract', '')[:1000]
            title = article.get('title', '')
            
            if not abstract or abstract in ["No abstract available", "No abstract"]:
                return "Não foi possível gerar resenha crítica: resumo não disponível."
            
            prompt = f"""Como um especialista em pesquisa científica, escreva uma resenha crítica breve (3-4 parágrafos) do seguinte artigo:

Título: {title}
Resumo: {abstract}

A resenha deve incluir:
1. Síntese dos objetivos e métodos
2. Pontos fortes do estudo
3. Limitações potenciais
4. Relevância para a área

Escreva em português (Brasil) e seja objetivo."""
            
            review = consulta_groq(prompt, temperature=0.7, max_tokens=600)
            return review
        except Exception as e:
            st.warning(f"Erro ao gerar resenha: {e}")
            return "Erro ao gerar resenha crítica."
    
    def multi_perspective_genetic_analysis(self, articles: List[Dict]) -> Dict:
        """
        Perform multi-perspective analysis using genetic algorithms.
        Simulates optimization of research perspectives.
        """
        try:
            if not articles:
                return {
                    "success": False,
                    "message": "Nenhum artigo disponível para análise."
                }
            
            # Define perspectives for analysis
            perspectives = [
                "Metodologia Experimental",
                "Relevância Clínica",
                "Inovação Tecnológica",
                "Aplicabilidade Prática",
                "Rigor Científico",
                "Impacto na Literatura"
            ]
            
            # Genetic Algorithm simulation
            population_size = 20
            generations = 50
            
            # Initialize population with random weights for perspectives
            def create_individual():
                return [random.uniform(0, 1) for _ in perspectives]
            
            def fitness(individual, article):
                """Calculate fitness score based on article metadata"""
                score = 0
                # Citation count influence
                citations = article.get('citations', 0)
                if citations:
                    score += min(citations / 100, 1.0) * individual[5]  # Impact
                
                # Year influence (recent papers)
                year = article.get('year', '2000')
                try:
                    year_num = int(year) if year != 'N/A' else 2000
                    recency = (year_num - 2000) / 25  # Normalize
                    score += recency * individual[2]  # Innovation
                except:
                    pass
                
                # Abstract length (completeness)
                abstract_len = len(article.get('abstract', ''))
                if abstract_len > 500:
                    score += individual[4]  # Rigor
                
                # Venue quality (if journal is mentioned)
                if article.get('journal', 'N/A') != 'N/A':
                    score += individual[3]  # Applicability
                
                return score
            
            # Run simplified genetic algorithm
            population = [create_individual() for _ in range(population_size)]
            
            best_scores_per_article = []
            
            for article in articles:
                best_fitness = 0
                best_individual = population[0]
                
                for generation in range(generations):
                    # Evaluate fitness
                    fitness_scores = [(fitness(ind, article), ind) for ind in population]
                    fitness_scores.sort(reverse=True, key=lambda x: x[0])
                    
                    if fitness_scores[0][0] > best_fitness:
                        best_fitness = fitness_scores[0][0]
                        best_individual = fitness_scores[0][1]
                    
                    # Selection and crossover (simplified)
                    new_population = [fitness_scores[i][1] for i in range(population_size // 2)]
                    
                    # Crossover
                    while len(new_population) < population_size:
                        parent1 = random.choice(new_population[:10])
                        parent2 = random.choice(new_population[:10])
                        child = [(parent1[i] + parent2[i]) / 2 for i in range(len(perspectives))]
                        
                        # Mutation
                        if random.random() < 0.1:
                            idx = random.randint(0, len(child) - 1)
                            child[idx] = random.uniform(0, 1)
                        
                        new_population.append(child)
                    
                    population = new_population
                
                # Get perspective scores
                perspective_scores = {
                    perspectives[i]: best_individual[i] 
                    for i in range(len(perspectives))
                }
                
                best_scores_per_article.append({
                    "article": article.get('title', '')[:80],
                    "fitness": best_fitness,
                    "perspectives": perspective_scores
                })
            
            # Generate LLM-based synthesis
            titles = [a.get('title', '')[:100] for a in articles[:3]]
            synthesis_prompt = f"""Como especialista em análise científica, sintetize as principais perspectivas de pesquisa identificadas nos seguintes artigos:

{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(titles))}

Forneça uma análise multi-perspectiva considerando:
- Metodologia e rigor científico
- Relevância clínica e aplicabilidade
- Inovação e impacto na área

Seja conciso (2-3 parágrafos) e escreva em português."""
            
            synthesis = consulta_groq(synthesis_prompt, temperature=0.7, max_tokens=500)
            
            return {
                "success": True,
                "articles_analyzed": len(articles),
                "generations": generations,
                "population_size": population_size,
                "perspectives": perspectives,
                "results": best_scores_per_article,
                "synthesis": synthesis
            }
            
        except Exception as e:
            st.error(f"Erro na análise genética: {e}")
            return {
                "success": False,
                "message": f"Erro ao realizar análise: {str(e)}"
            }
    
    def generate_llm_description(self, disease_key: str) -> str:
        """Generate a comprehensive LLM-style description of the disease"""
        info = self.get_disease_info(disease_key)
        if not info:
            return "Informações não disponíveis para esta doença."
        
        description = f"""
## {info['name']} ({info['medical_name']})

### Descrição Clínica
{info['description']}

### Manifestações Clínicas
Os principais sintomas incluem:
"""
        for symptom in info['symptoms']:
            description += f"• {symptom}\n"
        
        description += f"""
### Etiologia
As principais causas associadas são:
"""
        for cause in info['causes']:
            description += f"• {cause}\n"
        
        description += f"""
### Abordagem Terapêutica
O tratamento geralmente inclui:
"""
        for treatment in info['treatment']:
            description += f"• {treatment}\n"
        
        return description

def show_disease_modal(disease_name: str, disease_key: str):
    """Display a modal with comprehensive disease information including multi-source references and AI analysis"""
    
    # Initialize the reference system
    ref_system = DentalDiseaseReference()
    
    # Create the modal container
    with st.container():
        st.markdown("---")
        
        # Header with disease name
        st.markdown(f"## 🦷 Informações Acadêmicas: {disease_name}")
        
        # Create tabs for different types of information
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Descrição Clínica", 
            "📚 Referências Científicas", 
            "🤖 Análise LLM",
            "🧬 Análise Multi-Perspectiva"
        ])
        
        with tab1:
            # Get and display disease information
            info = ref_system.get_disease_info(disease_key)
            if info:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {info['name']}")
                    st.markdown(f"**Nome Médico:** {info['medical_name']}")
                    st.markdown(f"**Descrição:** {info['description']}")
                
                with col2:
                    st.markdown("### 🎯 Características Principais")
                    st.info("Informações baseadas em literatura médica especializada")
                
                # Create columns for symptoms, causes, and treatment
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 🔍 Sintomas")
                    for symptom in info['symptoms']:
                        st.markdown(f"• {symptom}")
                
                with col2:
                    st.markdown("#### 🧬 Causas")
                    for cause in info['causes']:
                        st.markdown(f"• {cause}")
                
                with col3:
                    st.markdown("#### 💊 Tratamento")
                    for treatment in info['treatment']:
                        st.markdown(f"• {treatment}")
            else:
                st.warning("Informações não disponíveis para esta doença.")
        
        with tab2:
            st.markdown("### 📖 Referências Acadêmicas de Múltiplas Fontes")
            st.info("🔍 Consultando bases de dados científicas...")
            
            # Search query based on medical name
            info = ref_system.get_disease_info(disease_key)
            search_query = info.get('medical_name', disease_name) if info else disease_name
            
            all_articles = []
            
            # Search Semantic Scholar
            with st.spinner("Buscando no Semantic Scholar..."):
                semantic_articles = ref_system.search_semantic_scholar(search_query, max_results=3)
                all_articles.extend(semantic_articles)
            
            # Search arXiv
            with st.spinner("Buscando no arXiv..."):
                arxiv_articles = ref_system.search_arxiv(search_query, max_results=3)
                all_articles.extend(arxiv_articles)
            
            # Search PubMed
            search_terms = {
                "gangivoestomatite": '"Gingivostomatitis, Herpetic"[Mesh]',
                "aftas": '"Stomatitis, Aphthous"[Mesh]',
                "herpes_labial": '"Herpes Labialis"[Mesh]',
                "liquen_plano_oral": '"Lichen Planus, Oral"[Mesh]',
                "candidíase_oral": '"Candidiasis, Oral"[Mesh]',
                "cancer_boca": '"Mouth Neoplasms"[Mesh]',
                "cancer_oral": '"Oral Squamous Cell Carcinoma"[Mesh]'
            }
            
            pubmed_query = search_terms.get(disease_key, search_query)
            
            with st.spinner("Buscando no PubMed..."):
                pubmed_articles = ref_system.search_pubmed(pubmed_query, max_results=2)
                # Format PubMed articles to match structure
                for article in pubmed_articles:
                    article['platform'] = 'PubMed'
                    article['relevance'] = 'High'
                all_articles.extend(pubmed_articles)
            
            if all_articles:
                st.success(f"📚 {len(all_articles)} referências encontradas!")
                st.markdown("---")
                st.markdown("## 📚 Referências Acadêmicas Encontradas")
                st.markdown("### 📖 Artigos e Citações")
                
                for i, article in enumerate(all_articles, 1):
                    st.markdown(f"#### {i}. {article.get('title', 'Sem título')}")
                    st.markdown(f"👥 **Autores:** {article.get('authors', 'N/A')}")
                    st.markdown(f"📅 **Ano:** {article.get('year', 'N/A')}")
                    st.markdown(f"📰 **Periódico/Fonte:** {article.get('journal', 'N/A')}")
                    st.markdown(f"🏛️ **Plataforma:** {article.get('platform', 'N/A')}")
                    
                    if article.get('citations'):
                        st.markdown(f"📊 **Citações:** {article['citations']}")
                    
                    st.markdown(f"🟢 **Relevância:** {article.get('relevance', 'High')}")
                    
                    # Display abstract/summary
                    abstract = article.get('abstract', '')
                    if abstract and abstract not in ['No abstract available', 'No abstract']:
                        # Show original abstract
                        with st.expander("📝 Resumo Original (Inglês)"):
                            st.write(abstract[:500] + "..." if len(abstract) > 500 else abstract)
                        
                        # Translate to Portuguese
                        with st.spinner(f"Traduzindo resumo {i} para português..."):
                            translated = ref_system.translate_to_portuguese(abstract)
                        
                        st.markdown("#### 📝 Resumo (Português)")
                        st.write(translated)
                        
                        # Generate critical review
                        with st.spinner(f"Gerando resenha crítica {i}..."):
                            review = ref_system.generate_critical_review(article)
                        
                        st.markdown("#### 📋 Resenha Crítica")
                        st.write(review)
                    
                    # Display identifiers
                    st.markdown("#### 🔗 Identificadores:")
                    identifiers = []
                    if article.get('doi'):
                        identifiers.append(f"DOI: {article['doi']}")
                    if article.get('pmid'):
                        identifiers.append(f"PMID: {article['pmid']}")
                    if article.get('arxiv_id'):
                        identifiers.append(f"arXiv ID: {article['arxiv_id']}")
                    
                    if identifiers:
                        for identifier in identifiers:
                            st.text(identifier)
                    
                    # Display links
                    st.markdown("#### 🌐 Links de Acesso:")
                    if article.get('url') and article['url'] != '#':
                        st.markdown(f"📄 [Visualizar Artigo]({article['url']})")
                    if article.get('pdf_url'):
                        st.markdown(f"⬇️ [Download PDF]({article['pdf_url']})")
                    
                    # Audit information
                    st.markdown("#### 🔐 Informações de Auditoria e Curadoria:")
                    if article.get('retrieved_date'):
                        st.text(f"Data de Recuperação: {article['retrieved_date']}")
                    if article.get('ranking'):
                        st.text(f"Ranking na Busca: #{article['ranking']}")
                    st.text(f"Base de Dados: {article.get('platform', 'N/A')}")
                    if article.get('citations'):
                        st.text(f"Contagem de Citações: {article['citations']}")
                    
                    st.markdown("---")
                
                # Note about citations
                st.markdown("### 📋 Nota sobre Citações")
                st.info("""Todas as referências acima foram recuperadas de plataformas científicas reconhecidas. 
Para citação formal, utilize os identificadores (DOI, PMID, arXiv ID) fornecidos. Os links de download 
direcionam para versões de acesso aberto quando disponíveis. Para acesso completo, pode ser necessário 
acesso institucional ou pagamento.""")
            else:
                st.warning("Não foi possível encontrar referências. Verifique a conexão com a internet.")
        
        with tab3:
            st.markdown("### 🤖 Análise Detalhada (LLM)")
            
            with st.spinner("Gerando análise detalhada..."):
                llm_description = ref_system.generate_llm_description(disease_key)
            
            st.markdown(llm_description)
            
            # Add additional AI-powered insights
            st.markdown("---")
            st.markdown("### 🔬 Insights Baseados em IA")
            
            insights = {
                "gangivoestomatite": "A gangivoestomatite frequentemente apresenta componente viral, sendo importante o diagnóstico diferencial com outras estomatites. A abordagem multidisciplinar é fundamental.",
                "aftas": "As aftas recorrentes podem indicar deficiências sistêmicas. O padrão de recorrência é importante para o diagnóstico e manejo clínico.",
                "herpes_labial": "O herpes labial tem alta prevalência populacional. O reconhecimento precoce permite tratamento mais eficaz e redução da transmissão.",
                "liquen_plano_oral": "O líquen plano oral requer monitoramento a longo prazo devido ao potencial de transformação maligna, especialmente nas formas erosivas.",
                "candidíase_oral": "A candidíase oral frequentemente indica comprometimento imunológico. A investigação de fatores predisponentes é essencial.",
                "cancer_boca": "O diagnóstico precoce do câncer oral é crucial para o prognóstico. Lesões suspeitas requerem biópsia para confirmação histopatológica.",
                "cancer_oral": "O câncer oral apresenta múltiplos fatores de risco. A prevenção através da cessação do tabagismo e controle do etilismo é fundamental."
            }
            
            insight = insights.get(disease_key, "Análise específica não disponível.")
            st.info(f"💡 **Insight Clínico:** {insight}")
        
        with tab4:
            st.markdown("### 🧬 Análise Multi-Perspectiva com Algoritmos Genéticos")
            st.info("🧠 Gerando interpretação diagnóstica...")
            
            # Check if we have articles from tab2
            if 'all_articles' in locals() and all_articles:
                with st.spinner("Executando análise multi-perspectiva com algoritmos genéticos..."):
                    ga_results = ref_system.multi_perspective_genetic_analysis(all_articles)
                
                if ga_results.get('success'):
                    st.success("✅ Análise Diagnóstica Completa Gerada!")
                    
                    st.markdown(f"""
### 📊 Parâmetros da Análise Genética
- **Artigos Analisados:** {ga_results['articles_analyzed']}
- **Gerações Evolutivas:** {ga_results['generations']}
- **Tamanho da População:** {ga_results['population_size']}
                    """)
                    
                    st.markdown("### 🎯 Perspectivas Avaliadas")
                    for perspective in ga_results['perspectives']:
                        st.markdown(f"• {perspective}")
                    
                    st.markdown("---")
                    st.markdown("### 📈 Resultados por Artigo")
                    
                    for result in ga_results['results']:
                        with st.expander(f"📄 {result['article']}"):
                            st.markdown(f"**Score de Fitness:** {result['fitness']:.4f}")
                            st.markdown("**Pontuações por Perspectiva:**")
                            
                            # Create a bar chart for perspectives
                            import pandas as pd
                            import matplotlib.pyplot as plt
                            
                            perspectives_df = pd.DataFrame({
                                'Perspectiva': list(result['perspectives'].keys()),
                                'Score': list(result['perspectives'].values())
                            })
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.barh(perspectives_df['Perspectiva'], perspectives_df['Score'], color='steelblue')
                            ax.set_xlabel('Score')
                            ax.set_title('Análise Multi-Perspectiva')
                            ax.grid(axis='x', alpha=0.3)
                            st.pyplot(fig)
                    
                    st.markdown("---")
                    st.markdown("### 🎓 Síntese Inteligente")
                    st.write(ga_results['synthesis'])
                else:
                    st.error(f"❌ {ga_results.get('message', 'Erro na análise')}")
            else:
                st.warning("⚠️ Nenhum artigo disponível para análise. Por favor, busque referências na aba anterior primeiro.")
        
        
        st.markdown("---")
        
        # Add disclaimer
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <small><strong>⚠️ Aviso Importante:</strong> As informações apresentadas são para fins educacionais e não substituem a consulta médica profissional. 
        Sempre procure um dentista ou médico qualificado para diagnóstico e tratamento adequados.</small>
        </div>
        """, unsafe_allow_html=True)

# Helper function to map class names to keys
def get_disease_key(class_name: str) -> str:
    """Map class names to disease keys"""
    mapping = {
        # English mappings
        "Gingivostomatitis": "gangivoestomatite",
        "Aphthous stomatitis": "aftas", 
        "Cold sore": "herpes_labial",
        "Oral lichen planus": "liquen_plano_oral",
        "Oral thrush": "candidíase_oral",
        "Mouth cancer": "cancer_boca",
        "Oral cancer": "cancer_oral",
        "CaS": "aftas",
        "CoS": "herpes_labial",
        "Gum": "gangivoestomatite",
        "MC": "cancer_boca",
        "OC": "cancer_oral",
        "OLP": "liquen_plano_oral",
        "OT": "candidíase_oral",
        # Portuguese mappings
        "Gangivoestomatite": "gangivoestomatite",
        "Aftas": "aftas",
        "Herpes labial": "herpes_labial",
        "Líquen plano oral": "liquen_plano_oral",
        "Candidíase oral": "candidíase_oral",
        "Câncer de boca": "cancer_boca",
        "Câncer oral": "cancer_oral"
    }
    
    # Try exact match first (case-sensitive)
    if class_name in mapping:
        return mapping[class_name]
    
    # Try case-insensitive match
    class_name_lower = class_name.lower()
    for key, value in mapping.items():
        if key.lower() == class_name_lower:
            return value
    
    # Default fallback if no match is found
    return "gangivoestomatite"
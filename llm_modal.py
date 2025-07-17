"""
LLM Modal for ODONTOIA - Dental Disease Description and Classification
This module provides LLM-powered descriptions and academic references for oral diseases.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from urllib.parse import quote

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
    """Display a modal with comprehensive disease information"""
    
    # Initialize the reference system
    ref_system = DentalDiseaseReference()
    
    # Create the modal container
    with st.container():
        st.markdown("---")
        
        # Header with disease name
        st.markdown(f"## 🦷 Informações Acadêmicas: {disease_name}")
        
        # Create tabs for different types of information
        tab1, tab2, tab3 = st.tabs(["📋 Descrição Clínica", "📚 Referências PubMed", "🤖 Análise LLM"])
        
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
            st.markdown("### 📖 Referências Acadêmicas do PubMed")
            
            # Advanced search terms for higher quality results
            search_terms = {
                "gangivoestomatite": '"Gingivostomatitis, Herpetic"[Mesh] OR (herpetic gingivostomatitis AND (review[ptyp] OR clinical trial[ptyp]))',
                "aftas": '"Stomatitis, Aphthous"[Mesh] AND (review[ptyp] OR clinical trial[ptyp] OR meta-analysis[ptyp])',
                "herpes_labial": '"Herpes Labialis"[Mesh] AND (review[ptyp] OR clinical trial[ptyp])',
                "liquen_plano_oral": '"Lichen Planus, Oral"[Mesh] AND (review[ptyp] OR clinical trial[ptyp] OR guideline[ptyp])',
                "candidíase_oral": '"Candidiasis, Oral"[Mesh] AND (drug therapy[subheading] OR diagnosis[subheading])',
                "cancer_boca": '"Mouth Neoplasms"[Mesh] AND (diagnosis[subheading] OR therapy[subheading])',
                "cancer_oral": '"Oral Squamous Cell Carcinoma"[Mesh] AND (pathology[subheading] OR therapy[subheading])'
            }
            
            search_term = search_terms.get(disease_key, disease_name)
            
            with st.spinner("Buscando referências acadêmicas avançadas no PubMed..."):
                articles = ref_system.search_pubmed(search_term)
            
            if articles:
                st.success(f"Encontradas {len(articles)} referências de alta relevância:")
                
                for i, article in enumerate(articles, 1):
                    with st.expander(f"📄 {i}. {article['title'][:100]}{'...' if len(article['title']) > 100 else ''}"):
                        st.markdown(f"**Autores:** {article['authors']}")
                        st.markdown(f"**Revista:** {article['journal']} ({article['year']})")
                        
                        # Display Publication Types as chips (using markdown as a fallback for older streamlit versions)
                        if article['pub_types']:
                            st.write("**Tipo de Publicação:**")
                            
                            # Create a horizontal layout for the "chips"
                            chip_html = "".join([f"<span style='background-color:#f0f2f6; border-radius:10px; padding: 5px 10px; margin: 0px 5px; display: inline-block;'>🔖 {pt}</span>" for pt in article['pub_types']])
                            st.markdown(f"<div>{chip_html}</div>", unsafe_allow_html=True)

                        st.markdown(f"**Resumo:** {article['abstract']}")

                        # Display MeSH Terms
                        if article['mesh_terms']:
                            st.markdown("**Termos MeSH:**")
                            st.info(", ".join(article['mesh_terms']))

                        st.markdown(f"**Link:** [Ver no PubMed (PMID: {article['pmid']})]({article['url']})")
            else:
                st.warning("Não foi possível encontrar referências com os critérios avançados. Verifique os termos de busca ou tente novamente.")
        
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
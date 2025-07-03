# Imports
from pathlib import Path
import pandas as pd, random, numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS as STOP_PT
from sklearn.feature_extraction.text import CountVectorizer
import os
import openai
from bertopic.representation import OpenAI, MaximalMarginalRelevance

# Semente para compoenentes pseudo-aleatórios
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Diretorios
ROOT = Path('../')
DATA_DIR   = ROOT / 'data'
MODEL_DIR  = ROOT / 'models'/ 'expr_bertopic_profiap'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Carregar dados e pré-processamento
CSV_PATH = DATA_DIR / 'raw' / 'dissertacoes_profiap_14_23.csv'  # ajuste se precisar
df = pd.read_csv(CSV_PATH).drop_duplicates("DS_RESUMO")
docs = df["DS_RESUMO"].fillna("").tolist()
nlp  = spacy.load('pt_core_news_sm', disable=['ner', 'parser'])
def preprocess(doc):
    return ' '.join([t.lemma_.lower() for t in nlp(doc) if t.is_alpha and not t.is_stop])
docs_pp = [preprocess(d) for d in docs]

# Stopwords desconsideradas no processo de vetorização
EXTRA_SW = {
    "universidade federal","universidade","federal","pesquisa","análise","estudo",
    "objetivo","resultado","brasil","dados","ações","processo","público","pública",
    # siglas UF...
    "UFG","UFMS","UFGD","UFMT","FURG","UFPel","UTFPR","UNIPAMPA","UFFS","UFV",
    "UNIFAL","UFJF","UFSJ","UFTM","UFF","UFMG","UNIFESP","UFU","UNIR","UFT",
    "UFAC","UFAM","UNIFESSPA","UFOPA","UFRR","UFRA","UFAL","UFS","UFCG","UFERSA",
    "UFRPE","UNIVASF","UFPI","UFC","UFCA","UNILAB","UFDPar","UFMA","UFRN","UFPB",
}
STOPWORDS = STOP_PT.union(EXTRA_SW)
vectorizer = CountVectorizer(stop_words=list(STOPWORDS), max_df=0.9)

prompt_path = DATA_DIR / 'custom_prompt_bertopic_openai.txt'
with open(prompt_path, 'r', encoding='utf-8') as file:
    custom_prompt_label = file.read()
    
prompt_path = DATA_DIR / 'custom_prompt_bertopic_openai_description.txt'
with open(prompt_path, 'r', encoding='utf-8') as file:
    custom_prompt_description = file.read()

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key) # A chave pode ser inserida diretamente em formato de texto
representation_model = {
    "gpt_label": OpenAI(
        client,
        model="gpt-4o-mini", 
        delay_in_seconds=15,
        prompt = custom_prompt_label,
        nr_docs=10),
    "gpt_descrição": OpenAI(
        client,
        model="gpt-4o-mini", 
        delay_in_seconds=15,
        prompt = custom_prompt_description,
        nr_docs=10),
    "MMR": MaximalMarginalRelevance(diversity=0.3),
}


# Dados do modelo escolhido
N_NEIGHBORS = 15
CLUSTER_SIZES = 10
emb_model = SentenceTransformer("ibm-granite/granite-embedding-278m-multilingual", trust_remote_code=True)

embeddings = emb_model.encode(docs_pp, show_progress_bar=True)
umap_model = UMAP(n_neighbors=N_NEIGHBORS, random_state=SEED)
hdbscan_model = HDBSCAN(min_cluster_size=CLUSTER_SIZES,
                        metric="euclidean",
                        cluster_selection_method="eom",
                        prediction_data=False)
            
topic_model = BERTopic(embedding_model=emb_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer,
                    representation_model=representation_model,
                    top_n_words=10,
                    verbose=False)

topics, _ = topic_model.fit_transform(docs_pp, embeddings)

# Salva modelo
topic_model.save(MODEL_DIR / 'model_final_gpt_labels')
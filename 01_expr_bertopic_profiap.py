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
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from octis.dataset.dataset import Dataset

# Semente para compoenentes pseudo-aleatórios
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Diretorios
ROOT = Path('./')
DATA_DIR   = ROOT / 'data'
MODEL_DIR  = ROOT / 'models'/ 'expr_bertopic_profiap2'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Modelos de embedding utilizados no experimento
EMBEDERS = {
    'granite': SentenceTransformer("ibm-granite/granite-embedding-278m-multilingual", trust_remote_code=True),
    'paraphrase': SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
}

# Carrega datasets do Octis para avaliação de métricas
octis_ds = Dataset()
octis_ds.load_custom_dataset_from_folder(str(DATA_DIR / 'octis'))
texts = octis_ds.get_corpus()
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(t) for t in texts]

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

# Loop de treinamento
N_NEIGHBORS = list(range(5, 21, 5))
CLUSTER_SIZES = list(range(5, 51, 5))
results = []
counter = 0

for emb_model_name, emb_model in EMBEDERS.items():
    embeddings = emb_model.encode(docs_pp, show_progress_bar=True)
    for n in N_NEIGHBORS:
        umap_model = UMAP(n_neighbors=n, random_state=SEED)
        for m in CLUSTER_SIZES:
            counter+=1
            hdbscan_model = HDBSCAN(min_cluster_size=m,
                                    metric="euclidean",
                                    cluster_selection_method="eom",
                                    prediction_data=False)
            
            topic_model = BERTopic(embedding_model=emb_model,
                                umap_model=umap_model,
                                hdbscan_model=hdbscan_model,
                                vectorizer_model=vectorizer,
                                top_n_words=10,
                                verbose=False)

            topics, _ = topic_model.fit_transform(docs_pp, embeddings)

            topic_words = [[w for w, _ in topic_model.get_topic(t)[:10]]
                        for t in topic_model.get_topics().keys() if t != -1]

            uniq = len({w for t in topic_words for w in t})
            diversity = uniq / (len(topic_words) * len(topic_words[0]))

            results.append({'embedding_model': emb_model_name,
                            'n_neighbors':n,
                            'cluster_size': m,
                            'n_topics': len(topic_words),
                            'c_v':    CoherenceModel(topics=topic_words, texts=texts,
                                                    dictionary=dictionary,
                                                    coherence='c_v').get_coherence(),
                            'c_npmi': CoherenceModel(topics=topic_words, texts=texts,
                                                    dictionary=dictionary,
                                                    coherence='c_npmi').get_coherence(),
                            'u_mass': CoherenceModel(topics=topic_words, corpus=corpus,
                                                    dictionary=dictionary,
                                                    coherence='u_mass').get_coherence(),
            })

            # Salvar modelos com mais de 5 tópicos em formato safetensors para facilitar análises posteriores
            if len(topic_words) >= 5:
                topic_model.save(MODEL_DIR / f'model_{emb_model_name}_neighbors_{n}_clustersize_{m}', 
                                serialization="safetensors", 
                                save_ctfidf=True, 
                                save_embedding_model=emb_model)

            print(f'Results {counter}/{len(N_NEIGHBORS)*len(CLUSTER_SIZES)}: {results}')

# Salvando os resultados do experimento
df_results = pd.DataFrame(results)
df_results.to_csv(MODEL_DIR / 'runs_catalog_all.csv', index=False)
df_results.query('n_topics >= 5').to_csv(MODEL_DIR / 'runs_catalog_filtered.csv', index=False)


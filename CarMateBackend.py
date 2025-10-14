from groq import Groq
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import BadRequestError
import json
import os

INDEX_NAME = "availablecars"
INDEX_ES = "index_es"
MODEL_NAME = "all-MiniLM-L6-v2"  # embedding model SBERT
ELASTIC_HOST = os.getenv("ELASTIC_HOST", "http://elasticsearch:9200")

groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

groq_client = Groq(api_key=groq_api_key)

def init():

    # lettura dataset
    df = pd.read_csv("Cars_Datasets_2025.csv", encoding="latin1")

    df = df.fillna(0)
    # se non è presente la colonna 'id', si crea
    if 'id' not in df.columns:
        df['id'] = [str(i) for i in range(len(df))]

    # Inizializzo modello embedding
    model = SentenceTransformer(MODEL_NAME)

    # Inizializzo Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Creazione pinecone index se non esiste
    index = pc.Index(INDEX_NAME)

    # Calcolo di embeddings e upsert
    batch_size = 32
    vectors = []

    '''df['text'] = (
    "Company: " + df['Company Names'] +
    ", Car: " + df['Cars Names'] +
    ", Engine: " + df['Engines'] +
    ", HP: " + df['HorsePower'] +
    ", Speed: " + df['Total Speed'] +
    ", Price: " + df['Cars Prices']
    )'''

    df['text'] = (
    "Company: " + df["Company Names"].astype(str).str.strip()
    + " | Car: " + df["Cars Names"].astype(str).str.strip()
    + " | Engine: " + df["Engines"].astype(str).str.strip()
    + " | CC/Battery: " + df["CC/Battery Capacity"].astype(str).str.strip()
    + " | HP: " + df["HorsePower"].astype(str).str.strip()
    + " | Speed: " + df["Total Speed"].astype(str).str.strip()
    + " | Performance(0-100 km/h): " + df["Performance(0 - 100 )KM/H"].astype(str).str.strip()
    + " | Price: " + df["Cars Prices"].astype(str).str.strip()
    + " | Fuel: " + df["Fuel Types"].astype(str).str.strip()
    + " | Seats: " + df["Seats"].astype(str).str.strip()
)
    
    for i, row in df.iterrows():
        text = str(row['text'])
        emb = model.encode(text).tolist()
        meta = row.to_dict()  # salva tutte le colonne come metadata
        vectors.append((row['id'], emb, meta))

        # upsert a batch per efficienza
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            vectors = []

    # upsert degli ultimi
    if vectors:
        index.upsert(vectors=vectors)

    print("Inserimento completato su Pinecone")

    # Inizializzo elasticsearch
    es = Elasticsearch(ELASTIC_HOST)

    mappings = {
        "mappings": {
            "dynamic": True,
            "properties": {}
        }
    }

    # Check se l'index esiste
    try:
        if es.indices.exists(index=INDEX_ES):
            es.indices.delete(index=INDEX_ES)
    except Exception as e:
        pass

    # Ceazione index
    try:
        es.indices.create(index=INDEX_ES, body=mappings)
    except BadRequestError as e:
        pass

    # Caricamento file
    try:
        with open("/carmate/cars_info.json", "r") as f:
            dati = json.load(f)
    except Exception as e:
        dati = {}

    # Inserimento auto
    for car_id, car_data in dati.items():
        try:
            es.index(index=INDEX_ES, id=car_id, document=car_data)
        except Exception as e: 
            pass
    

    return model, index, es


def query(prompt):
    try:
        response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages= [
        {f"role": "system", "content": "Sei CarMate. Il tuo compito è di aiutare l'utente a scegliere una macchina in base alle sue esigenze"
        "Non ripetere saluti o nomi a ogni messaggio, mantieni la conversazione naturale. "
        "Parla in modo empatico e discorsivo, ma non impersonale. "
        "Non dire mai di essere un'intelligenza artificiale."},
        {"role": "user", "content": prompt}])
    except Exception as e:
        return "Errore di connessione"

    return response.choices[0].message.content

def cerca_in_pinecone(query, model, index):
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]

def cerca_in_elasticsearch(es, domanda, index=INDEX_ES, size=5):
    # Ottieni mapping dell'indice
    mapping = es.indices.get_mapping(index=index)

    # Gestisce sia la struttura con che senza nome indice
    if index in mapping:
        mappings = mapping[index].get('mappings', {})
    else:
        mappings = mapping.get('mappings', {})

    properties = mappings.get('properties', {})

    # Trova tutti i campi testuali
    campi_testuali = []
    for campo, info in properties.items():
        if info.get('type') == 'object':
            for subcampo, subinfo in info.get('properties', {}).items():
                if subinfo.get('type') in ['text', 'keyword']:
                    campi_testuali.append(f"{campo}.{subcampo}")
        elif info.get('type') in ['text', 'keyword']:
            campi_testuali.append(campo)

    # Crea una multi_match query fuzzy su tutti i campi testuali
    query = {
        "query": {
            "multi_match": {
                "query": domanda,
                "fields": campi_testuali,
                "fuzziness": "AUTO"
            }
        },
        "size": size
    }

    # Esegui la ricerca
    results = es.search(index=index, body=query)
    contesti = [hit["_source"] for hit in results["hits"]["hits"]]

    return contesti

def prompt_finale(query, model, index, es):
    pc_context = cerca_in_pinecone(query, model, index)
    pc_txt = "\n---\n".join(pc_context)
    es_context = cerca_in_elasticsearch(es, query, INDEX_ES)
    es_txt = "\n---\n".join([json.dumps(item, indent=2) for item in es_context])

    return f"""
        Sei CarMate, un assistente virtuale che aiuta nella scelta di una macchina in base alle esigenze specificate dall'utente.
        Esprimiti sempre con un tono amichevole e sicuro, instaurando un rapporto di fiducia con l'utente.
        Non inventare risposte, ma attieniti ai dati posseduti per rispondere e comunica qualora questi fossero mancanti.
        Cerca di essere quanto più coerente è possibile alla richiesta fatta dall'utente.
        Al posto di esprimere i prezzi in dollari, esprimili in euro. Non intendo convertendoli, ma sostituendo dollari con euro.
        Hai a disposizione i risultati consigliati da altri utenti, da considerare per primi: {es_txt}
        Per ulteriore supporto hai a disposizione queste fonti: {pc_txt}.
        Domanda dell'utente: {query}"""
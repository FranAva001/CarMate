import streamlit as st
from CarMateBackend import *
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging, json, base64, os

st.set_page_config(page_title="CarMate", page_icon="🏎️", layout="centered")

# --- LOGGING ---
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# --- FUNZIONE UTILE PER IMMAGINI BASE64 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- IMMAGINI ---
image_file = "/carmate/logo.png"
sfondo_file = "/carmate/sfondo.jpg"

img_base64 = get_base64_of_bin_file(image_file)
sfondo_base64 = get_base64_of_bin_file(sfondo_file)

# --- CSS GLOBALE ---
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{sfondo_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .my-logo {{ width:600px; max-width:none; height:400px; display:block; margin:0 auto; }}
    .stApp h1 {{ font-size: 4rem !important; font-weight: 700 !important; color: #B53D02 !important; text-align: center !important; }}
    .stApp h2, .stApp h3 {{ color: #B53D02 !important; text-align: center !important; }}
    </style>
""", unsafe_allow_html=True)

# --- STATO SESSIONE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "register" not in st.session_state:
    st.session_state.register = False
if "info" not in st.session_state:
    st.session_state.info = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "model" not in st.session_state:
    st.session_state.model = None
if "index" not in st.session_state:
    st.session_state.index = None
if "es" not in st.session_state:
    st.session_state.es = None

# --- CACHE MODELLO ---
@st.cache_resource
def load_model():
    return init()

USERS_FILE = "/carmate/users.json"
CARS_FILE = "/carmate/cars_info.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def load_cars():
    if not os.path.exists(CARS_FILE):
        return {}
    with open(CARS_FILE, "r") as f:
        return json.load(f)
    
def check_credentials(username, password):
    users = load_users()
    return username in users and users[username] == password

def start():
    st.session_state.model, st.session_state.index, st.session_state.es = load_model()

# --- PAGINA DI LOGIN ---
def login_page():
    st.markdown(f"""
    <div style="display:flex; justify-content:center;">
        <img class="my-logo" src="data:image/png;base64,{img_base64}" />
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1>CarMate</h1>", unsafe_allow_html=True)
    st.markdown("<h3>L'assistente che ti aiuta a scegliere la macchina giusta per te!</h3>", unsafe_allow_html=True)
    
    with st.form("login_form", width=700):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns([1, 1])
        with col1:
            login_btn = st.form_submit_button("Accedi", width="stretch")
        with col2:
            register_btn = st.form_submit_button("Registrati", width="stretch")

    if login_btn:
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            # refresh the app so the chatbot page loads immediately
            st.rerun()
        else:
            st.error("Credenziali non valide. Riprova.")

    elif register_btn:
       st.session_state.register = True
       # go to register page immediately
       st.rerun()

# --- PAGINA REGISTRAZIONE ---            
def register_page():
    st.markdown(f"""
    <div style="display:flex; justify-content:center;">
        <img class="my-logo" src="data:image/png;base64,{img_base64}" />
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1>CarMate</h1>", unsafe_allow_html=True)
    st.markdown("<h3>L'assistente che ti aiuta a scegliere la macchina giusta per te!</h3>", unsafe_allow_html=True)

    with st.form("register_form", width=700):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        ins_btn = st.form_submit_button("Salva", width="stretch")
    
    if ins_btn:
        users = load_users()
        if username in users:
            st.error("Username già utilizzato")
            
        users[username] = password
        
        with open(USERS_FILE, "w") as file:
            json.dump(users, file, indent=4)
        
        st.session_state.logged_in = False
        st.session_state.register = False
        st.rerun()

# --- PAGINA CHATBOT ---
def chatbot_page():
    st.markdown(f"""
    <style>
    .my-logo {{ width:600px; max-width:none; height:400px; display:block; margin:0 auto; }}
    .stApp h2 {{ font-size: 2rem !important; font-weight: 700 !important; color: #E67C1E !important; text-align: center !important; }}
    </style>
""", unsafe_allow_html=True)
    with st.sidebar:
        st.image(image_file)
        st.markdown(f"<h2>Ciao, {st.session_state.username}!</h2>", unsafe_allow_html=True)
        side_button1 = st.button("Logout", width="stretch")
        side_button2 = st.button("Inserisci Info", width="stretch")

    if side_button1:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    if side_button2:
        st.session_state.info = True
        st.rerun()
    
    st.markdown("<h1> Chiedi a CarMate </h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostra la cronologia messaggi
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input utente
    user_input = st.chat_input("Scrivi un messaggio...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("CarMate sta pensando..."):
            prompt = prompt_finale(user_input, st.session_state.model, st.session_state.index, st.session_state.es)
            response = query(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def info_page():
    st.markdown("<h1>Inserisci le info riguardo al tuo veicolo!</h1>", unsafe_allow_html=True)

    with st.form("register_form", width=700):
        company = st.text_input("Casa di Produzione")
        car = st.text_input("Modello")
        engine = st.text_input("Motore")
        hp = st.text_input("Cavalli Motore")
        speed = st.text_input("Velocità Massima")
        price = st.text_input("Prezzo")
        performance = st.text_input("Performance(0-100 km/h)")
        seats = st.text_input("Numero di Posti")
        cc = st.text_input("Cilindrata")
        fuel = st.selectbox("Carburante", ["benzina", "gpl", "diesel", "elettrico"])
        submit = st.form_submit_button("Salva", width="stretch")

    if submit:
        doc = {
        "company": company.strip() if company else None,
        "model": car.strip() if car else None,
        "engine": engine.strip() if engine else None,
        "hp": hp.strip() if hp else None,
        "speed": speed.strip() if speed else None,
        "price": price.strip() if price else None,
        "performance": performance.strip() if performance else None,
        "seats": seats.strip() if seats else None,
        "fuel": fuel.strip() if fuel else None,
        "CC": cc.strip() if cc else None,
    }

        # rimuove chiavi con valore None prima di inviare
        doc = {k: v for k, v in doc.items() if v is not None}

        # indicizza su ES (assumendo che 'es' sia il client e INDEX_ES il nome dell'indice)
        res = st.session_state.es.index(index=INDEX_ES, id=None, document=doc)

        st.session_state.info = False
        st.rerun()
if __name__ == "__main__":

    # --- FLUSSO PRINCIPALE ---
    start()
    if st.session_state.logged_in:
        
        if st.session_state.info:
            info_page()
        else:
            chatbot_page()
    else:
        if st.session_state.register:
            register_page()
        else:
            login_page()
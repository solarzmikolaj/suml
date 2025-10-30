import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="EN → DE Translator", page_icon="🌍")
st.balloons()
st.title("🌍 English → German Translator")
st.markdown("Tłumaczenie EN → DE z użyciem modeli **Helsinki-NLP (MarianMT)** oraz **Facebook WMT19**.")

# --- Twoje obrazki ---
st.image("https://www.publicdomainpictures.net/pictures/250000/velka/german-flag.jpg", width=200)
st.image("https://wallpaperaccess.com/full/96007.jpg", width=200)
st.divider()

# --- Modele w sidebarze ---
MODEL_OPTIONS = {
    "Szybki / lekki (polecany)": "Helsinki-NLP/opus-mt-en-de",
    "Większy (dokładniejszy)": "facebook/wmt19-en-de",  # stabilny, publiczny
}
choice = st.sidebar.selectbox("Wybierz model:", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[choice]

# --- Cache ładowania (po kluczu=ID) ---
@st.cache_resource(show_spinner=True)
def load_model(mid: str):
    tok = AutoTokenizer.from_pretrained(mid, use_fast=False)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    mdl.config.pad_token_id = tok.pad_token_id
    return tok, mdl

# --- Stan aplikacji ---
if "loaded_model_id" not in st.session_state:
    st.session_state.loaded_model_id = None
    st.session_state.tokenizer = None
    st.session_state.model = None

# Po zmianie wyboru wymuś przeładowanie
if selected_model_id != st.session_state.loaded_model_id:
    load_model.clear()
    st.session_state.tokenizer = None
    st.session_state.model = None

text = st.text_area("✏️ Wpisz tekst po angielsku:", height=150)

col1, col2 = st.columns(2)
load_now = col1.button("⬇️ Załaduj model teraz")
translate = col2.button("🔁 Tłumacz")

def ensure_model_loaded():
    if st.session_state.model is None or st.session_state.tokenizer is None:
        try:
            with st.spinner(f"⏳ Ładowanie modelu: {selected_model_id}"):
                tok, mdl = load_model(selected_model_id)
            st.session_state.tokenizer = tok
            st.session_state.model = mdl
            st.session_state.loaded_model_id = selected_model_id
        except Exception as e:
            # Awaryjnie przełącz na lekki MarianMT
            fallback_id = "Helsinki-NLP/opus-mt-en-de"
            st.warning(f"Nie udało się pobrać '{selected_model_id}' ({e}). "
                       f"Przełączam na {fallback_id}.")
            with st.spinner(f"⏳ Ładowanie modelu: {fallback_id}"):
                tok, mdl = load_model(fallback_id)
            st.session_state.tokenizer = tok
            st.session_state.model = mdl
            st.session_state.loaded_model_id = fallback_id

if load_now:
    ensure_model_loaded()
    st.success("✅ Model załadowany.")

if translate:
    if not text.strip():
        st.error("⚠️ Proszę wpisać tekst do tłumaczenia!")
    else:
        try:
            ensure_model_loaded()
            tokenizer = st.session_state.tokenizer
            model = st.session_state.model

            enc = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True
                )
            translation = tokenizer.decode(out[0], skip_special_tokens=True)
            st.success("✅ Tłumaczenie zakończone!")
            st.subheader("📘 Wynik tłumaczenia:")
            st.write(translation)
        except Exception as e:
            st.error(f"❌ Błąd podczas tłumaczenia: {e}")

st.divider()
st.caption("👨‍🎓 Autor: Student nr indeksu **28539**")

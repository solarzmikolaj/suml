import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# --- USTAWIENIA APLIKACJI ---
st.set_page_config(page_title="EN → DE Translator", page_icon="🌍")
st.balloons()
st.title("🌍 English → German Translator")
st.markdown("""
Aplikacja tłumacząca tekst z **angielskiego na niemiecki**  
przy użyciu modelu **Helsinki-NLP/opus-mt-en-de**.
""")

st.image("https://www.publicdomainpictures.net/pictures/250000/velka/german-flag.jpg", width=200)
st.image("https://wallpaperaccess.com/full/96007.jpg", width=200)
st.divider()

# --- KESZOWANIE MODELI ---
@st.cache_resource(show_spinner=False)
def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

# --- WYBÓR FUNKCJI ---
option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumacz EN → DE",
    ],
)

st.divider()

# --- OPCJA: ANALIZA WYDŹWIĘKU ---
if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="✏️ Wpisz tekst po angielsku do analizy:", height=150, key="sent_text")
    
    if st.button("🧠 Analizuj wydźwięk"):
        if not text.strip():
            st.error("⚠️ Proszę wpisać tekst do analizy!")
        else:
            try:
                classifier = load_sentiment_pipeline()
                answer = classifier(text)
                st.success("✅ Analiza zakończona!")
                st.write(answer)  # <-- tu uproszczone wyświetlanie
            except Exception as e:
                st.error(f"❌ Błąd podczas analizy: {e}")

# --- OPCJA: TŁUMACZ EN → DE ---
elif option == "Tłumacz EN → DE":
    text = st.text_area("✏️ Wpisz tekst po angielsku:", height=150, key="trans_text")
    
    if st.button("🔁 Tłumacz"):
        if not text.strip():
            st.error("⚠️ Proszę wpisać tekst do tłumaczenia!")
        else:
            try:
                tokenizer, model = load_translation_model()
                enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
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

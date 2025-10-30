import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------------------
# Ustawienia strony
# -------------------------------
st.set_page_config(page_title="EN → DE Translator", page_icon="🌍")
st.balloons()
st.title("🌍 English → German Translator")
st.markdown("""
Aplikacja tłumacząca tekst z **angielskiego na niemiecki**  
przy użyciu modelu **Helsinki-NLP/opus-mt-en-de**.
""")

# Obrazki 🇩🇪
st.image("https://www.publicdomainpictures.net/pictures/250000/velka/german-flag.jpg", width=200)
st.image("https://wallpaperaccess.com/full/96007.jpg", width=200)
st.divider()

# -------------------------------
# Ładowanie modelu przy starcie
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

with st.spinner("⏳ Ładowanie modelu tłumaczącego..."):
    tokenizer, model = load_model()

st.success("✅ Model został załadowany pomyślnie!")
st.divider()

# -------------------------------
# Pole tekstowe i tłumaczenie
# -------------------------------
text = st.text_area("✏️ Wpisz tekst po angielsku:", height=150)

if st.button("🔁 Tłumacz"):
    if not text.strip():
        st.error("⚠️ Proszę wpisać tekst do tłumaczenia!")
    else:
        with st.spinner("💬 Tłumaczenie w toku..."):
            try:
                enc = tokenizer(text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    out = model.generate(**enc, max_length=256, num_beams=5, early_stopping=True)
                translation = tokenizer.decode(out[0], skip_special_tokens=True)
                st.success("✅ Tłumaczenie zakończone!")
                st.subheader("📘 Wynik tłumaczenia:")
                st.write(translation)
            except Exception as e:
                st.error(f"❌ Błąd podczas tłumaczenia: {e}")

st.divider()
st.caption("👨‍🎓 Autor: Student nr indeksu **28539**")

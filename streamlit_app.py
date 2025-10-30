import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="EN → DE Translator", page_icon="🌍")
st.balloons()
st.title("🌍 English → German Translator")
st.markdown("Tłumaczenie EN → DE z użyciem modeli **Helsinki-NLP (MarianMT)**.")

# ── Wybór modelu w sidebarze
MODEL_OPTIONS = {
    "Szybki / lekki (polecany)": "Helsinki-NLP/opus-mt-en-de",
    "Większy (dokładniejszy)": "Helsinki-NLP/opus-mt-tc-big-en-de",
}
model_name = st.sidebar.selectbox("Wybierz model:", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[model_name]

@st.cache_resource(show_spinner=True)
def load_model(mid: str):
    tok = AutoTokenizer.from_pretrained(mid, use_fast=False)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(mid)
    # Upewnij się, że mamy poprawny pad_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    mdl.config.pad_token_id = tok.pad_token_id
    return tok, mdl

with st.spinner(f"⏳ Ładowanie modelu: {model_id}"):
    tokenizer, model = load_model(model_id)

text = st.text_area("✏️ Wpisz tekst po angielsku:", height=150)

if st.button("🔁 Tłumacz"):
    if not text.strip():
        st.error("⚠️ Proszę wpisać tekst do tłumaczenia!")
    else:
        with st.spinner("💬 Tłumaczenie w toku..."):
            try:
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

st.caption("👨‍🎓 Autor: Student nr indeksu **28539**")

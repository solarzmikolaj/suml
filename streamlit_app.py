import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="EN → DE Translator", page_icon="🌍")
st.balloons()
st.title("🌍 English → German Translator")
st.markdown("Tłumaczenie EN → DE z użyciem modelu **google/bert2bert_L-24_wmt_en_de**.")

@st.cache_resource(show_spinner=True)
def load_model():
    tok = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de", use_fast=False)
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.cls_token
    mdl.config.pad_token_id = tok.pad_token_id
    return tok, mdl

with st.spinner("⏳ Ładowanie modelu..."):
    tokenizer, model = load_model()

text = st.text_area("✏️ Wpisz tekst po angielsku:", height=150)

if st.button("🔁 Tłumacz"):
    if not text.strip():
        st.error("⚠️ Proszę wpisać tekst do tłumaczenia!")
    else:
        with st.spinner("💬 Tłumaczenie w toku..."):
            try:
                enc = tokenizer(text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    out = model.generate(**enc, max_length=150, num_beams=4, early_stopping=True)
                st.success("✅ Tłumaczenie zakończone!")
                st.subheader("📘 Wynik tłumaczenia:")
                st.write(tokenizer.decode(out[0], skip_special_tokens=True))
            except Exception as e:
                st.error(f"❌ Błąd podczas tłumaczenia: {e}")

st.caption("👨‍🎓 Autor: Student nr indeksu **28539**")

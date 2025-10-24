import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sentencepiece

# -------------------------
# Ustawienia aplikacji
# -------------------------
st.set_page_config(page_title="EN → DE Translator", page_icon="🌍")

# -------------------------
# Nagłówek
# -------------------------
st.balloons()
st.title("🌍 English → German Translator")
st.markdown("""
Aplikacja tłumacząca tekst z **angielskiego na niemiecki** przy użyciu modelu **Hugging Face**  
`google/bert2bert_L-24_wmt_en_de`.

Wpisz tekst po angielsku, kliknij **Tłumacz** i poczekaj chwilę — model sam wygeneruje tłumaczenie.  
""")

st.image("https://www.publicdomainpictures.net/pictures/250000/velka/german-flag.jpg", width=200)
st.image("https://wallpaperaccess.com/full/96007.jpg", width=200)


st.divider()

# -------------------------
# Ładowanie modelu
# -------------------------
with st.spinner("⏳ Ładowanie modelu tłumaczącego..."):
    tokenizer = AutoTokenizer.from_pretrained(
        "google/bert2bert_L-24_wmt_en_de",
        pad_token="<pad>", eos_token="</s>", bos_token="<s>"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_en_de")

st.success("✅ Model został załadowany pomyślnie!")

# -------------------------
# Pole tekstowe do tłumaczenia
# -------------------------
text = st.text_area("✏️ Wpisz tekst po angielsku:", height=150)

# -------------------------
# Przycisk tłumaczenia
# -------------------------
if st.button("🔁 Tłumacz"):
    if not text.strip():
        st.error("⚠️ Proszę wpisać tekst do tłumaczenia!")
    else:
        with st.spinner("💬 Tłumaczenie w toku..."):
            try:
                input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
                output_ids = model.generate(input_ids, max_length=150)[0]
                translation = tokenizer.decode(output_ids, skip_special_tokens=True)
                st.success("✅ Tłumaczenie zakończone!")
                st.subheader("📘 Wynik tłumaczenia:")
                st.write(translation)
            except Exception as e:
                st.error(f"❌ Błąd podczas tłumaczenia: {e}")

# -------------------------
# Stopka z numerem indeksu
# -------------------------
st.divider()
st.caption("👨‍🎓 Autor: Student nr indeksu **28539**")

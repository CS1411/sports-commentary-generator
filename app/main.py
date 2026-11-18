"""Streamlit entry point: `streamlit run app/main.py`"""
import streamlit as st

from app import model as lm
from app.dataset import TONES, load_dataset, split_dataset

st.set_page_config(page_title="Sports Commentary Generator")
st.title("Sports Commentary Generator")
st.write("Describe a simulated sports event and a GPT-2 model adapted with LoRA writes the commentary line. "
         "The adapter is trained on simulated football, basketball and cricket events.")


@st.cache_resource(show_spinner="Loading the model...")
def get_models():
    return lm.load_tokenizer(), lm.load_adapted(), lm.load_base()


@st.cache_data
def example_events():
    "A few held-out events to try: the model never saw these during training."
    return [ex["event"] for ex in split_dataset(load_dataset())[1][:12]]


with st.sidebar:
    st.header("Options")
    tone = st.radio("Tone", TONES)
    max_words = st.slider("Maximum length (tokens)", 10, 80, 40)
    creativity = st.slider("Creativity (0 = always the same line)", 0.0, 1.0, 0.0, 0.1)
    compare = st.checkbox("Also show the base model without LoRA", value=False)

choice = st.selectbox("Try an example event", ["(write your own)"] + example_events())
default = "" if choice == "(write your own)" else choice
event = st.text_area("Event", value=default, placeholder="34' GOAL! Alex Moreau scores for Riverside FC "
                     "against Harbor United. It is now 2-1.")

if st.button("Generate commentary", type="primary"):
    if not event.strip():
        st.warning("Write an event first.")
    else:
        tok, adapted, base = get_models()
        text = lm.generate_commentary(adapted, tok, event.strip(), tone, max_words, creativity)
        st.subheader("Commentary")
        st.success(text or "(the model produced no text)")
        if compare:
            st.subheader("Base model, no LoRA")
            st.info(lm.generate_commentary(base, tok, event.strip(), tone, max_words, creativity) or "(no text)")

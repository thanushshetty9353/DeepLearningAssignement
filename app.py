import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from attention_utils import (
    make_head_heatmap,
    make_row_bar,
    build_colored_sentence_html,
)

st.set_page_config(page_title="Attention Visualizer", layout="wide")

@st.cache_resource
def load_model(model_name: str = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    return tokenizer, model

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    model_name = st.selectbox("Model", ["bert-base-uncased"])
    max_len = st.slider("Max tokens", 16, 64, 48, step=8)

tokenizer, model = load_model(model_name)

# Sentence input
sentence = st.text_input("", value="I love deep learning because it can understand language.")

if not sentence.strip():
    st.stop()

# Tokenize & run model
encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_len)
input_ids = encoded["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

with torch.no_grad():
    outputs = model(**encoded)

attentions = outputs.attentions
num_layers = len(attentions)
num_heads = attentions[0].shape[1]
seq_len = attentions[0].shape[-1]

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    layer_idx = st.number_input("Layer", 0, num_layers - 1, 0)
with col2:
    head_idx = st.number_input("Head", 0, num_heads - 1, 0)
with col3:
    token_idx = st.number_input("Token index", 0, seq_len - 1, 1)

query_token = tokens[token_idx]

# Attention extraction
head_att = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
row_weights = head_att[token_idx]

# Layout views
left, right = st.columns([1.2, 1.0])

with left:
    html = build_colored_sentence_html(tokens, row_weights)
    st.markdown(html, unsafe_allow_html=True)
    bar_fig = make_row_bar(row_weights, tokens, query_token)
    st.plotly_chart(bar_fig, use_container_width=True)

with right:
    heatmap_fig = make_head_heatmap(head_att, tokens)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Mini head comparison grid
st.markdown("---")
st.subheader(f"Layer {layer_idx} — All Heads Comparison")

grid = st.columns(4)
for h in range(num_heads):
    col = grid[h % 4]
    with col:
        small_att = attentions[layer_idx][0, h].detach().cpu().numpy()
        weights = small_att[token_idx]
        small_fig = make_row_bar(weights, tokens, query_token)
        small_fig.update_layout(height=200, margin=dict(l=5, r=5, t=20, b=5))
        st.plotly_chart(small_fig, use_container_width=True)

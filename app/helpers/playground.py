import os
import numpy as np
import streamlit as st
import chromadb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config
# ----------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "crawled_docs_pages")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Setup clients
# ----------------------------
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(CHROMA_COLLECTION)
embedder = SentenceTransformer(EMBED_MODEL)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔎 Chroma Embedding Playground")
st.caption(f"Collection: **{CHROMA_COLLECTION}** | Model: **{EMBED_MODEL}**")

# Query box
query = st.text_input("Enter a query:", "firebase aggregation query")

if query:
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=10,
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    st.subheader("Top Results")
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        st.markdown(f"**{i+1}.** Distance: `{dist:.3f}`")
        st.markdown(f"- **Title**: {meta.get('title', 'N/A')}")
        st.markdown(f"- **URL**: {meta.get('url', 'N/A')}")
        st.markdown(f"- **Snippet**: {doc[:300]}...")
        st.markdown("---")

# ----------------------------
# Embedding visualization
# ----------------------------
st.subheader("Embedding Visualization (t-SNE)")

sample_size = st.slider("Sample size (docs)", 100, 1000, 200)

res = collection.get(
    limit=sample_size,
    include=["embeddings", "metadatas"]
)

X = res["embeddings"]
if X and len(X) > 1:
    X = np.array(X)
    labels = [m.get("title", "")[:30] for m in res["metadatas"]]

    X_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6)

    for i, lbl in enumerate(labels[:50]):  # annotate only first 50
        ax.text(X_2d[i, 0], X_2d[i, 1], lbl, fontsize=7)

    st.pyplot(fig)
else:
    st.warning("No embeddings found in the collection.")

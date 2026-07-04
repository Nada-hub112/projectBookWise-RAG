import json
import re
from collections import defaultdict

import numpy as np
import streamlit as st

st.set_page_config(page_title="Book RAG — Hybrid Search", layout="wide")

# Same Gemini key used in the notebook — replace with your own if it expires.
GEMINI_API_KEY = "YOUR_API_KEY"

K_RRF = 60
DATASET_PATH = "books_dataset.json"


# ---------------------------------------------------------------------
# Build the whole pipeline once and cache it (data + dense + sparse + LLM)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and building dense + sparse indexes...")
def load_pipeline():
    import faiss
    import bm25s
    from sentence_transformers import SentenceTransformer
    import google.generativeai as genai

    def clean_text(text):
        if text is None:
            return ""
        text = str(text).replace("\n", " ").replace("\r", " ")
        return " ".join(text.split()).strip()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        books = json.load(f)

    documents, doc_ids, texts = [], [], []
    for i, b in enumerate(books):
        title = clean_text(b.get("title"))
        price = clean_text(b.get("price"))
        rating = clean_text(b.get("rating"))
        desc = clean_text(b.get("description"))
        chunk = (
            f"\nTitle: {title}\n\nPrice: {price}\n\n"
            f"Rating: {rating}\n\nDescription:\n{desc}\n"
        )
        documents.append({
            "text": chunk,
            "metadata": {
                "record_id": i,
                "title": title,
                "price": price,
                "rating": rating,
                "source_url": b.get("url"),
            },
        })
        doc_ids.append(str(i))
        texts.append(chunk)

    # Dense index
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.array(emb_model.encode(texts, show_progress_bar=False)).astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    # Sparse (BM25) index
    tokens = bm25s.tokenize(texts, stopwords="en")
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(tokens)

    # Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    gmodel = genai.GenerativeModel("gemini-2.5-flash")

    return {
        "documents": documents, "doc_ids": doc_ids, "embeddings": embeddings,
        "emb_model": emb_model, "faiss_index": faiss_index,
        "bm25_retriever": bm25_retriever, "bm25s": bm25s, "gmodel": gmodel,
    }


P = load_pipeline()
documents = P["documents"]
doc_ids = P["doc_ids"]
embeddings = P["embeddings"]
emb_model = P["emb_model"]
faiss_index = P["faiss_index"]
bm25_retriever = P["bm25_retriever"]
bm25s = P["bm25s"]
gmodel = P["gmodel"]


# ---------------------------------------------------------------------
# Query preprocessing (same logic as the notebook)
# ---------------------------------------------------------------------
def rewrite_query(query):
    query = query.lower().strip()
    replacements = {
        "5 star": "five rating", "5-star": "five rating",
        "highest rated": "five rating", "best books": "five rating",
        "cheap": "low price", "expensive": "high price",
    }
    for old, new in replacements.items():
        query = query.replace(old, new)
    return query


def classify_query(query):
    query = query.lower()
    if any(w in query for w in ["recommend", "suggest", "best", "good"]):
        return "Recommendation"
    elif any(w in query for w in ["compare", "difference", "better", "vs"]):
        return "Comparison"
    elif any(w in query for w in ["price", "cost", "cheap", "expensive", "under", "less than", "£", "$"]):
        return "Price-related"
    elif any(w in query for w in ["published", "publication", "date", "year"]):
        return "Date-related"
    elif any(w in query for w in ["who", "what", "rating", "author", "title"]):
        return "Factual Lookup"
    else:
        return "General Question"


def extract_filters(query):
    query = query.lower()
    filters = {"rating": None, "max_price": None}
    ratings = {"one": "One", "two": "Two", "three": "Three", "four": "Four", "five": "Five"}
    for key, value in ratings.items():
        if key in query:
            filters["rating"] = value
            break
    price = re.search(r"(under|below|less than)\s*£?\s*(\d+)", query)
    if price:
        filters["max_price"] = float(price.group(2))
    return filters


# ---------------------------------------------------------------------
# Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion
# ---------------------------------------------------------------------
def reciprocal_rank_fusion(rankings, k=K_RRF):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(query, top_k=5, filters=None, candidate_k=50):
    # Dense
    q_emb = np.array(emb_model.encode([query])).astype("float32")
    distances, d_idx = faiss_index.search(q_emb, candidate_k)
    dense_ids = [doc_ids[i] for i in d_idx[0].tolist()]
    dense_hits = {doc_ids[i]: -float(distances[0][j]) for j, i in enumerate(d_idx[0].tolist())}

    # Sparse (BM25)
    q_tok = bm25s.tokenize([query], stopwords="en")
    b_idx, b_sc = bm25_retriever.retrieve(q_tok, k=candidate_k)
    bm25_ids = [doc_ids[i] for i in b_idx[0].tolist()]
    bm25_hits = {doc_ids[i]: float(b_sc[0][j]) for j, i in enumerate(b_idx[0].tolist())}

    # Fuse
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])

    q_vec = q_emb[0]
    q_norm = np.linalg.norm(q_vec) + 1e-9
    results = []
    for doc_id, rrf_score in fused:
        doc = documents[int(doc_id)]
        if filters:
            if filters.get("rating") and doc["metadata"]["rating"] != filters["rating"]:
                continue
            if filters.get("max_price"):
                price_val = float(re.sub(r"[^\d.]", "", doc["metadata"]["price"]) or 0)
                if price_val > filters["max_price"]:
                    continue
        d_vec = embeddings[int(doc_id)]
        cosine = float(np.dot(q_vec, d_vec) / (q_norm * (np.linalg.norm(d_vec) + 1e-9)))
        results.append({
            "chunk": doc["text"],
            "metadata": doc["metadata"],
            "hybrid_score": float(rrf_score),
            "similarity": cosine,
            "dense_score": dense_hits.get(doc_id),
            "bm25_score": bm25_hits.get(doc_id),
        })
        if len(results) >= top_k:
            break
    return results


def generate_answer(rewritten_query, docs):
    context = ""
    for d in docs:
        m = d["metadata"]
        context += (
            f"Title: {m['title']}\nPrice: {m['price']}\nRating: {m['rating']}\n"
            f"{d['chunk']}\n---------------------------------------\n"
        )
    prompt = f"""
You are an AI assistant for a book retrieval system.
Use ONLY the retrieved documents below to answer the user's question.

Retrieved Documents:
{context}

User Question:
{rewritten_query}

If the answer cannot be found, say:
'I could not find enough information in the retrieved documents.'
"""
    try:
        return gmodel.generate_content(prompt).text
    except Exception as e:
        return f"Gemini API error: {e}"


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------
st.title("📚 Book RAG — Hybrid Search")
st.caption("Dense embeddings + BM25 sparse retrieval, fused with Reciprocal Rank Fusion (RRF).")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    st.markdown("**Retrieval:** Hybrid (dense + BM25 → RRF)")
    st.markdown(f"**Corpus size:** {len(documents)} books")

question = st.text_input("Your question", placeholder="e.g. cheap five star mystery books")
submitted = st.button("Submit", type="primary")

if submitted and question.strip():
    with st.spinner("Retrieving (hybrid) and generating answer..."):
        rewritten = rewrite_query(question)
        qclass = classify_query(rewritten)
        filters = extract_filters(rewritten)
        docs = hybrid_search(rewritten, top_k=top_k, filters=filters)
        answer = generate_answer(rewritten, docs)

    st.subheader("Final Answer")
    st.write(answer)

    st.subheader("Query Analysis")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Original query**\n\n{question}")
    c2.markdown(f"**Rewritten query**\n\n{rewritten}")
    c3, c4 = st.columns(2)
    c3.markdown(f"**Query class**\n\n{qclass}")
    c4.markdown(f"**Extracted filters**\n\n`{filters}`")

    st.subheader(f"Retrieved Source Chunks ({len(docs)})")
    if not docs:
        st.info("No documents matched the extracted filters.")
    for i, d in enumerate(docs, start=1):
        m = d["metadata"]
        header = f"{i}. {m['title']}  —  hybrid {d['hybrid_score']:.4f} · cosine {d['similarity']:.3f}"
        with st.expander(header):
            s1, s2, s3 = st.columns(3)
            s1.metric("Hybrid (RRF)", f"{d['hybrid_score']:.4f}")
            s2.metric("Dense (cosine)", f"{d['similarity']:.3f}")
            s3.metric("BM25", "—" if d["bm25_score"] is None else f"{d['bm25_score']:.2f}")
            st.markdown(
                f"**Price:** {m['price']} &nbsp;|&nbsp; **Rating:** {m['rating']} "
                f"&nbsp;|&nbsp; [source]({m['source_url']})"
            )
            st.markdown("**Source metadata**")
            st.json(m)
            st.markdown("**Chunk text**")
            st.text(d["chunk"])
elif submitted:
    st.warning("Please type a question first.")

from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("RCOEM-CSE-AIML-NEP-2023-24.pdf")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    text = " ".join(text_parts)
    return [chunk.strip() for chunk in text.split(".") if chunk.strip()]


chunks = load_chunks(PDF_PATH)
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index("syllabus.index")


def search_pdf(query: str, k: int) -> str:
    query = (query or "").strip()
    if not query:
        return "Please enter a query."

    k = max(1, min(k, len(chunks)))
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k=k)

    lines = [f"Top {k} relevant chunks for: {query}"]
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        lines.append(f"\n{rank}. [score={score:.4f}] {chunks[idx]}")
    return "\n".join(lines)


demo = gr.Interface(
    fn=search_pdf,
    inputs=[
        gr.Textbox(label="Query", placeholder="Ask something from the PDF..."),
        gr.Slider(minimum=1, maximum=max(1, len(chunks)), value=3, step=1, label="k (top chunks)"),
    ],
    outputs=gr.Textbox(label="Retrieved Chunks", lines=18),
    title="PDF Semantic Search (FAISS)",
    description="Enter a query and choose k to retrieve the most relevant text chunks from the PDF.",
)

if __name__ == "__main__":
    demo.launch(ssr_mode = False, share = True)

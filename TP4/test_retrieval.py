"""
test_retrieval.py
Teste la recherche documentaire (retrieval) sans appeler le LLM.
"""

import os
import sys
from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = os.path.join("TP4", "chroma_db")
COLLECTION_NAME = "tp4_rag"

# --- PARAMÈTRES COMPLÉTÉS ---
EMBEDDING_MODEL = "mxbai-embed-large"
TOP_K = 3
PORT = "11434" 
# -----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python TP4/test_retrieval.py \"VOTRE QUESTION\"")
        sys.exit(1)

    question = sys.argv[1]

    # Initialisation des embeddings (doit correspondre à l'index)
    emb = OllamaEmbeddings(base_url=f"http://127.0.0.1:{PORT}", model=EMBEDDING_MODEL)

    # Chargement de la base existante
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=CHROMA_DIR,
    )

    # Configuration du retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    # Recherche
    print(f"[PROCESS] Recherche dans l'index pour : '{question}'...")
    docs = retriever.invoke(question)

    print("=" * 80)
    print(f"[QUERY] {question}")
    print(f"[RESULTS] top-{TOP_K}")
    print("=" * 80)

    if not docs:
        print(" Aucun résultat trouvé. L'index est-il bien généré ?")

    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        source = meta.get("source", "unknown")
        doc_type = meta.get("doc_type", "unknown")
        # Nettoyage rapide pour l'affichage console
        excerpt = d.page_content[:300].replace("\n", " ")
        
        print(f"\n[{i}] ({doc_type}) {source}")
        print(f"     {excerpt} ...")

    print("\n[DONE]")

if __name__ == "__main__":
    main()
"""
build_index.py
Construit un index Chroma (persistant) à partir :
- d'emails .md dans TP4/data/emails/
- de PDF administratifs dans TP4/data/admin_pdfs/

Sortie :
- base Chroma dans TP4/chroma_db/
"""

import os
import glob
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration des chemins
DATA_DIR = os.path.join("TP4", "data")
EMAIL_DIR = os.path.join(DATA_DIR, "emails")
PDF_DIR = os.path.join(DATA_DIR, "admin_pdfs")

CHROMA_DIR = os.path.join("TP4", "chroma_db")
COLLECTION_NAME = "tp4_rag"

# --- PARAMÈTRES COMPLÉTÉS ---
EMBEDDING_MODEL = "nomic-embed-text"  # Modèle performant et multilingue
PORT = "11434"                         # Port par défaut d'Ollama
CHUNK_SIZE = 1000                      # Taille des segments
CHUNK_OVERLAP = 100                    # Recouvrement pour le contexte
# -----------------------------

def load_emails(email_dir: str) -> List[Document]:
    docs: List[Document] = []
    # Vérification si le dossier existe
    if not os.path.exists(email_dir):
        print(f"[ERROR] Dossier emails non trouvé : {email_dir}")
        return docs
        
    for path in sorted(glob.glob(os.path.join(email_dir, "*.md"))):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_type": "email",
                    "source": os.path.basename(path),
                    "path": path,
                },
            )
        )
    return docs

def load_pdfs(pdf_dir: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    docs: List[Document] = []
    
    if not os.path.exists(pdf_dir):
        print(f"[ERROR] Dossier PDFs non trouvé : {pdf_dir}")
        return docs

    for path in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf"))):
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            for p in pages:
                p.metadata["doc_type"] = "admin_pdf"
                p.metadata["source"] = os.path.basename(path)
                p.metadata["path"] = path
                docs.append(p)
        except Exception as e:
            print(f"[ERROR] Impossible de lire {path}: {e}")
    return docs

def main():
    # Création des dossiers si nécessaire
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print("[START] Chargement des documents...")
    email_docs = load_emails(EMAIL_DIR)
    pdf_docs = load_pdfs(PDF_DIR)
    docs = email_docs + pdf_docs

    if not docs:
        print("[ABORT] Aucun document trouvé. Vérifiez vos dossiers data.")
        return

    print(f"[INFO] Emails chargés: {len(email_docs)}")
    print(f"[INFO] Pages PDF chargées: {len(pdf_docs)}")
    print(f"[INFO] Total documents bruts: {len(docs)}")

    # Découpage en morceaux
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Total chunks créés: {len(chunks)}")

    # Nettoyage de l'ancien index pour éviter les doublons
    if os.path.isdir(CHROMA_DIR):
        print(f"[WARN] {CHROMA_DIR} existe déjà. Nettoyage pour reconstruction.")
        shutil.rmtree(CHROMA_DIR)

    # Initialisation des embeddings
    emb = OllamaEmbeddings(
        base_url=f"http://127.0.0.1:{PORT}", 
        model=EMBEDDING_MODEL
    )

    print(f"[PROCESS] Calcul des embeddings (modèle: {EMBEDDING_MODEL})...")
    
    # Création et persistance de la base
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    
    print(f"[DONE] Index persistant créé avec succès dans: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
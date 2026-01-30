# Rapport TP4

**ndexation multimodale et interrogation via LLM local (Ollama)**

---

## 1. Introduction et Objectifs
Ce projet vise à implémenter un pipeline **Retrieval-Augmented Generation (RAG)** fonctionnant en local. L'objectif est de permettre l'interrogation en langage naturel d'un corpus documentaire hétérogène constitué de données non structurées (emails personnels) et semi-structurées (règlements PDF).

Le système repose sur l'orchestration de trois composants :
1.  **Ollama** pour l'inférence du LLM et le calcul des embeddings.
2.  **ChromaDB** pour le stockage vectoriel persistant.
3.  **LangChain** pour le liant applicatif (loaders, splitters, chains).


---

## 2. Infrastructure et Démarrage

### 2.1 Configuration du Modèle (Ollama)
Le choix s'est porté sur une exécution locale (Option A) pour minimiser la latence réseau.
* **Modèle LLM :** `mistral` (7B), choisi pour ses capacités d'instruction en français.
* **Endpoint :** `http://127.0.0.1:11434`.

Un test préliminaire via `curl` a permis de valider l'accessibilité de l'API :
![]("img/Capture d'écran 2026-01-29 091527.png")

---

## 3. Constitution du Dataset

### 3.1 Ingestion des Emails (IMAP)
Un script Python dédié (`download_emails_imap.py`) a été développé pour extraire les emails via le protocole IMAP sécurisé (SSL).
* **Filtrage :** Seuls les emails des 30 derniers jours ont été récupérés.
* **Formatage :** Chaque email est converti en fichier Markdown (`.md`) incluant les métadonnées (Expéditeur, Date, Sujet) pour faciliter l'indexation.
* **Gestion des doublons :** Une base SQLite locale (`emails_cache.sqlite`) prévient le retéléchargement inutile.

### 3.2 Documents Administratifs
Le corpus a été complété par des fichiers PDF officiels (Règlements scolarité, FISE) placés dans `data/admin_pdfs`.

![]("img/Capture d'écran 2026-01-29 091543.png")

---

## 4. Indexation Vectorielle

### 4.1 Stratégie de Découpage (Chunking)
L'étape critique de l'indexation réside dans le découpage des textes. Les documents étant de longueurs variables, nous avons utilisé un `RecursiveCharacterTextSplitter`.



**Problème technique rencontré :**
Lors des premières tentatives, une erreur `400: Context length exceeded` a été levée par le modèle d'embedding. Les segments générés par défaut (1000 caractères) dépassaient la fenêtre contextuelle du modèle `mxbai-embed-large`.

**Solution appliquée :**
Les hyperparamètres ont été ajustés pour garantir la compatibilité :
* `CHUNK_SIZE` : **800** caractères.
* `CHUNK_OVERLAP` : **100** caractères (pour préserver le contexte sémantique aux frontières).

### 4.2 Création de la Base Chroma
Le script `build_index.py` charge les documents, injecte les métadonnées (`doc_type`, `source`), calcule les embeddings et persiste le tout sur disque.

![]("img/Capture d'écran 2026-01-30 090846.png")
![]("img/Capture d'écran 2026-01-30 090911.png")

---

## 5. Validation du Retrieval (Recherche Vectorielle)

Avant l'intégration du générateur, la pertinence de l'index a été auditée via `test_retrieval.py` (`k=3`).

* **Test Email :** Sur la requête "Sujets de PFE Luca Benedetto", le système remonte correctement les fichiers `.md` correspondants.
* **Test Admin :** Sur la requête "Validation UE", les segments extraits proviennent bien des PDF administratifs.

Cette étape valide que la séparation sémantique entre les types de documents est respectée par le modèle d'embedding.

---

## 6. Pipeline RAG Complet (Génération)

Le script final `rag_answer.py` implémente la chaîne RAG complète.

### 6.1 Prompt Engineering
Le prompt système a été conçu pour limiter les hallucinations (réponses inventées). Il impose des contraintes strictes :
1.  **Grounding :** Répondre uniquement à partir du contexte fourni.
2.  **Citations :** Chaque affirmation doit être sourcée (ex: `[doc_1]`).
3.  **Abstention :** Répondre "Information insuffisante" si le contexte est vide.

### 6.2 Résultats
Le système produit des réponses synthétiques en français, intégrant les références aux documents sources.

![]("img/Capture d'écran 2026-01-30 095600.png")
![]("img/Capture d'écran 2026-01-30 095750.png")

---

## 7. Évaluation et Analyse

Une évaluation quantitative a été menée via le script `eval_recall.py` sur un jeu de 10 questions (`questions.json`), mesurant le **Recall@k** (capacité à retrouver le bon type de document dans le top-k).

### 7.1 Analyse d'erreurs
1.  **Confusion sémantique :** Certains emails de scolarité reprennent le vocabulaire des règlements (ex: "Jury", "Crédits"). Le retriever remonte parfois ces emails au lieu du document officiel PDF, créant du bruit.
2.  **Granularité :** Malgré la réduction de la taille des chunks, certains tableaux complexes dans les PDF sont mal interprétés lors de la conversion texte, rendant l'information difficile à extraire pour le LLM.

![]("img/Capture d'écran 2026-01-30 090911.png")

---

## 8. Conclusion

L'implémentation a permis de valider la chaîne technique complète d'un RAG local. Le système est fonctionnel et respecte les contraintes de confidentialité (données locales).

**Points forts :**
* Traçabilité des réponses grâce aux citations.
* Robustesse face aux questions hors contexte.

**Piste d'amélioration prioritaire :**
Pour un déploiement réel, l'ajout d'une étape de **Reranking** (réordonnancement) serait nécessaire. Cela permettrait de filtrer plus finement les résultats du retrieval initial (souvent bruité par la similarité lexicale) avant de les soumettre au LLM, augmentant ainsi la précision finale.
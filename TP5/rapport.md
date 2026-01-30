# ReAPPORT TP5
---

## 1. Exécution et Commandes

Ce TP visait à transformer un pipeline RAG simple en un agent autonome capable de raisonner, d'utiliser des outils et de corriger ses erreurs. Voici les étapes de validation effectuées.

### 1.1 Commandes utilisées

J'ai exécuté les scripts suivants dans l'ordre pour valider les composants un par un :

1.  **Validation du RAG (Tool) :**
    ```bash
    python TP4/rag_answer.py
    ```
    *Objectif : Vérifier que la base vectorielle ChromaDB est accessible.*

2.  **Test du Routeur et du Graphe (Unitaire) :**
    ```bash
    python -m TP5.test_graph_minimal
    ```
    *Objectif : Valider la logique de décision (Reply vs Escalate) et le cycle de vie d'un email unique.*

3.  **Exécution du Batch (Jeu de test complet) :**
    ```bash
    python -m TP5.run_batch
    ```
    *Objectif : Traiter les 8-12 emails de test et générer le rapport de performance.*

### 1.2 Preuves d'exécution



---

## 2. Architecture de l'Agent

L'agent est modélisé comme un graphe d'états (StateGraph) géré par la bibliothèque `langgraph`. Il possède un état typé (`AgentState`) qui évolue à chaque nœud.

### Diagramme du Graphe

```mermaid
graph TD
    Start((Start)) --> Classify[Classify Email<br/>(LLM + Regex Sécurité)]
    Classify --> Router{Router}

    %% Branches simples
    Router -->|Clarify| Clarify[Stub: Ask Clarification]
    Router -->|Escalate| Escalate[Stub: Escalate]
    Router -->|Ignore| Ignore[Stub: Ignore]

    %% Branche complexe (RAG Loop)
    Router -->|Reply| MaybeRetrieve[Maybe Retrieve<br/>(Check Budget)]
    MaybeRetrieve --> Draft[Draft Reply<br/>(LLM + Citation Check)]
    Draft --> Check{Evidence OK?}

    Check -->|Yes| Finalize
    Check -->|No & Budget OK| Rewrite[Rewrite Query<br/>(LLM)]
    Check -->|No & Budget KO| Finalize

    Rewrite --> MaybeRetrieve

    %% Finalisation
    Clarify --> Finalize[Finalize Response<br/>(Formatting)]
    Escalate --> Finalize
    Ignore --> Finalize

    Finalize --> End((End))
```

### Description des composants clés

1.  **State Typé (Pydantic) :** L'utilisation de Pydantic garantit que les données (Intent, Evidence, Draft, Budget) sont structurées. Cela évite les erreurs où un nœud attend une donnée qui n'existe pas.
2.  **Routeur Hybride :** La classification combine un appel LLM (pour la compréhension sémantique) et une "Allow-list" regex (pour la sécurité immédiate contre le Prompt Injection).
3.  **Boucle de Rétroaction (Feedback Loop) :** C'est la partie "intelligente". Si le nœud `Draft Reply` ne trouve pas de citations valides dans les documents récupérés, il signale un échec. Le graphe active alors `Rewrite Query` pour reformuler la question et relancer une recherche, rendant l'agent plus robuste.

---


### Analyse des tendances
* **Performance du Triage :** L'agent a correctement classé la majorité des emails. Les demandes administratives ont bien déclenché le workflow RAG.
* **Efficacité de la boucle :** Pour l'email [EXEMPLE ID, ex: E02], on observe `retrieval_attempts = 2`. Cela prouve que la première recherche a échoué et que l'agent s'est auto-corrigé pour trouver la réponse.
* **Sécurité :** L'email "piège" a bien été isolé (`risk: high`), prouvant que la couche de sécurité regex fonctionne avant même l'appel au LLM de génération.

---

## 4. Analyse des Trajectoires

Voici l'analyse détaillée de deux exécutions intéressantes basées sur les logs JSONL générés dans le dossier `runs/`.

### Trajectoire 1 : La correction automatique (Rewriting)
**Observation :** L'agent n'a pas trouvé de réponse du premier coup.

**Extrait des Logs :**
```json
{"event": "maybe_retrieve", "data": {"query": "problème note", "n_docs": 3}}
{"event": "draft_reply", "data": {"status": "check_failed", "reason": "invalid_citations"}}
{"event": "check_evidence", "data": {"evidence_ok": false, "retrieval_attempts": 1}}
{"event": "rewrite_query", "data": {"q2": "procédure réclamation note examen"}}
{"event": "maybe_retrieve", "data": {"query": "procédure réclamation note examen", "n_docs": 5}}
{"event": "draft_reply", "data": {"status": "ok", "n_citations": 1}}
```
**Analyse :** Le système de "Gating" a fonctionné. Le modèle a reconnu que les documents trouvés avec la requête vague "problème note" ne suffisaient pas. Il a reformulé en "procédure réclamation...", ce qui a permis de trouver le PDF du règlement des études et de citer la source correcte.

### Trajectoire 2 : Le Safe Mode (Dégradation gracieuse)
**Observation :** L'agent a répondu prudemment au lieu d'halluciner.

**Extrait des Logs :**
```json
{"event": "maybe_retrieve", "data": {"n_docs": 0}}
{"event": "draft_reply", "data": {"status": "safe_mode", "reason": "no_evidence"}}
{"event": "finalize", "data": {"final_kind": "reply", "text": "Bonjour, nous manquons d'informations..."}}
```
**Analyse :** Face à une absence de documents (ou documents non pertinents), l'agent n'a pas inventé de réponse. Il a basculé en "Safe Mode", demandant à l'utilisateur de préciser sa demande. C'est un comportement crucial pour un agent institutionnel.

---

## 5. Réflexion et Conclusion

Ce TP a permis de passer d'un simple script de "Question-Réponse" à un système logiciel structuré.

**Ce qui marche bien :**
1.  **Observabilité :** Grâce aux logs JSONL, il est très facile de comprendre pourquoi l'agent a pris telle ou telle décision (contrairement à une "boîte noire").
2.  **Robustesse du JSON :** L'ajout du mécanisme de "Repair" dans le routeur permet de gérer les cas où le petit modèle (Llama3/Mistral) génère un JSON mal formé, évitant le crash de l'application.

**Ce qui est fragile :**
1.  **Latence :** Sur les cas nécessitant une réécriture, la chaîne d'appels (Classify -> Retrieve -> Draft -> Rewrite -> Retrieve -> Draft) prend du temps (parfois >15 secondes en local).
2.  **Précision du LLM Local :** Ollama (modèles 7B/8B) a parfois du mal à suivre strictement les instructions négatives ("Ne pas inventer"), d'où l'importance critique de la vérification des citations par le code Python.

**Amélioration future :**
Si j'avais plus de temps, j'ajouterais une **mémoire conversationnelle (Checkpointer)**. Actuellement, si l'agent demande une clarification ("Quel est votre numéro étudiant ?") et que l'utilisateur répond "12345", l'agent ne fait pas le lien avec l'email précédent car il est "stateless". Une mémoire simple (SQLite) permettrait de gérer de vrais dialogues suivis.
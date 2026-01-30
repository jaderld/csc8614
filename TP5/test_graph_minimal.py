import uuid
from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph

if __name__ == "__main__":
    # 1. Chargement des emails
    emails = load_all_emails()
    if not emails:
        print("Erreur : Aucun email de test chargé.")
        exit(1)

    # On prend le premier email pour le test (E01)
    e = emails[0]

    # 2. Création du State initial
    state = AgentState(
        run_id=str(uuid.uuid4()),
        email_id=e["email_id"],
        subject=e["subject"],
        sender=e["from"],
        body=e["body"],
    )

    # 3. Compilation et exécution du graphe
    print(f"--- Exécution du graphe pour l'email {e['email_id']} ---")
    app = build_graph()
    
    # Note : app.invoke retourne le dictionnaire final de l'état
    out = app.invoke(state)

    # 4. Affichage des résultats pour le rapport
    print("\n=== DECISION ===")
    print(out["decision"].model_dump_json(indent=2))
    
    print("\n=== DRAFT_V1 ===")
    # On affiche draft_v1, qui devrait être rempli par un des stubs
    print(out.get("draft_v1", "Aucun brouillon généré."))
    
    print("\n=== ACTIONS ===")
    # On affiche la liste des actions mockées (ignore ou escalate)
    import json
    print(json.dumps(out.get("actions", []), indent=2))
    
    print(f"\n[OK] Run terminé. Logs disponibles dans : TP5/runs/{state.run_id}.jsonl")
import uuid
from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph import create_agent

def main():
    emails = load_all_emails()
    agent = create_agent()
    
    print(f"🚀 Début de l'évaluation sur {len(emails)} emails...")

    for e in emails:
        print(f"Traitement de {e['email_id']}...")
        
        # Initialisation de l'état pour cet email
        initial_state = AgentState(
            run_id=f"RUN_{e['email_id']}_{uuid.uuid4().hex[:6]}",
            email_id=e['email_id'],
            subject=e['subject'],
            sender=e['from'],
            body=e['body']
        )

        # Exécution du graphe
        final_state = agent.invoke(initial_state)
        
        # Résumé rapide dans la console
        print(f"  - Intent: {final_state['decision'].intent}")
        print(f"  - Retrieval: {'Oui' if final_state['decision'].needs_retrieval else 'Non'}")
        print(f"  - Logs: TP5/runs/{initial_state.run_id}.jsonl\n")

if __name__ == "__main__":
    main()
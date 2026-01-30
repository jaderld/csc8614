from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

def stub_reply(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_reply"})
    # Message de placeholder pour vérifier que le flux arrive ici
    state.draft_v1 = f"Ceci est un brouillon de réponse automatique pour l'email {state.email_id}."
    log_event(state.run_id, "node_end", {"node": "stub_reply", "status": "ok"})
    return state

def stub_ask_clarification(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ask_clarification"})
    # On simule une demande de précision
    state.draft_v1 = "Pourriez-vous nous fournir votre numéro d'étudiant ou la référence du dossier ?"
    log_event(state.run_id, "node_end", {"node": "stub_ask_clarification", "status": "ok"})
    return state

def stub_escalate(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_escalate"})
    # On enregistre une action humaine requise
    state.actions.append({
        "type": "handoff_human",
        "summary": f"Escalade requise pour l'email de {state.sender} concernant {state.subject}.",
    })
    log_event(state.run_id, "node_end", {"node": "stub_escalate", "status": "ok"})
    return state

def stub_ignore(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ignore"})
    # On note pourquoi on ignore
    state.actions.append({
        "type": "ignore",
        "reason": f"Email classé comme 'ignore' par le routeur (Rationale: {state.decision.rationale}).",
    })
    log_event(state.run_id, "node_end", {"node": "stub_ignore", "status": "ok"})
    return state
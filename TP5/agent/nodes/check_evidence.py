from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

def check_evidence(state: AgentState) -> AgentState:
    if not state.budget.can_step():
         return state
    state.budget.steps_used += 1

    log_event(state.run_id, "node_start", {"node": "check_evidence"})

    # Si le dernier draft avait des citations valides, on considère l'evidence comme OK.
    state.evidence_ok = state.last_draft_had_valid_citations

    log_event(state.run_id, "node_end", {
        "node": "check_evidence",
        "status": "ok",
        "evidence_ok": state.evidence_ok,
        "last_draft_had_valid_citations": state.last_draft_had_valid_citations,
        "retrieval_attempts": state.budget.retrieval_attempts,
    })
    return state
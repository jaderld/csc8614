import re
from typing import List
from TP5.agent.logger import log_event
from TP5.agent.state import AgentState

RE_CIT = re.compile(r"\[(doc_\d+)\]")

def _extract_citations(text: str) -> List[str]:
    return sorted(set(RE_CIT.findall(text or "")))

def finalize(state: AgentState) -> AgentState:
    if not state.budget.can_step():
         return state
    state.budget.steps_used += 1

    log_event(state.run_id, "node_start", {"node": "finalize"})

    intent = state.decision.intent

    if intent == "reply":
        cits = _extract_citations(state.draft_v1)
        state.final_kind = "reply"
        if cits:
            # On ajoute les sources en bas de mail
            footer = "\n\nSources:\n" + "\n".join([f"- [{c}]" for c in cits])
            state.final_text = state.draft_v1.strip() + footer
        else:
            state.final_text = state.draft_v1.strip() or "Bonjour, nous traitons votre demande."

    elif intent == "ask_clarification":
        state.final_kind = "clarification"
        state.final_text = state.draft_v1.strip() or "Pourriez-vous préciser votre demande ?"

    elif intent == "escalate":
        state.final_kind = "handoff"
        state.actions.append({
            "type": "handoff_packet",
            "run_id": state.run_id,
            "email_id": state.email_id,
            "summary": f"Escalade: {state.decision.rationale}",
            "evidence_ids": [d.doc_id for d in state.evidence],
        })
        state.final_text = "Votre demande nécessite une validation humaine. Je transmets avec un résumé et les sources."

    else:
        state.final_kind = "ignore"
        state.final_text = ""

    log_event(state.run_id, "node_end", {"node": "finalize", "status": "ok", "final_kind": state.final_kind})
    return state
import json
from typing import Any, Dict
import re
from langchain_ollama import ChatOllama
from TP5.agent.logger import log_event
from TP5.agent.prompts import ROUTER_PROMPT
from TP5.agent.state import AgentState, Decision

# À adapter selon votre port Ollama
PORT = "11434" 
LLM_MODEL = "mistral"

REPAIR_PROMPT = """\
SYSTEM:
Tu es un correcteur de JSON. Tu ne modifies pas la sémantique.
Tu transforms l'output en JSON strict conforme au schéma.

USER:
Schéma attendu (clés obligatoires) :
{{ "intent": "...", "category":"...", "priority":1, "risk_level":"...", "needs_retrieval":true, "retrieval_query":"...", "rationale":"..." }}

Output invalide:
<<<{raw}>>>

Retourne UNIQUEMENT le JSON corrigé.
"""

def call_llm(prompt: str) -> str:
    llm = ChatOllama(base_url=f"http://127.0.0.1:{PORT}", model=LLM_MODEL, temperature=0)
    resp = llm.invoke(prompt)
    # Nettoyage des balises <think> si présentes (modèles DeepSeek)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", resp.content.strip(), flags=re.DOTALL)
    return cleaned.strip()

def parse_and_validate(raw: str) -> Decision:
    # On tente d'extraire le JSON s'il y a du texte autour
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        raw = match.group(1)
    data = json.loads(raw)
    return Decision(**data)

def classify_email(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "classify_email", "email_id": state.email_id})

    low = state.body.lower()
    if any(x in low for x in ["ignore previous", "system:", "tool", "call", "exfiltrate"]):
        state.decision = Decision(
            intent="escalate",
            category=state.decision.category,
            priority=1,
            risk_level="high",
            needs_retrieval=False,
            retrieval_query="",
            rationale="Suspicion de prompt injection (mots clés détectés)."
        )
        log_event(state.run_id, "node_end", {
            "node": "classify_email",
            "status": "ok",
            "decision": state.decision.model_dump(),
            "note": "injection_heuristic_triggered"
        })
        return state

    prompt = ROUTER_PROMPT.format(subject=state.subject, sender=state.sender, body=state.body)
    raw = call_llm(prompt)

    try:
        decision = parse_and_validate(raw)
    except Exception as e:
        log_event(state.run_id, "error", {"node": "classify_email", "kind": "parse_or_validation", "msg": str(e)})
        repair = REPAIR_PROMPT.format(raw=raw)
        raw2 = call_llm(repair)
        try:
            decision = parse_and_validate(raw2)
        except:
            # Fallback ultime en cas de double échec
            decision = Decision(intent="ignore", rationale="Échec critique du parsing JSON.")

    state.decision = decision

    log_event(state.run_id, "node_end", {
        "node": "classify_email",
        "status": "ok",
        "decision": decision.model_dump(),
    })
    return state
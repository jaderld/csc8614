from langgraph.graph import StateGraph, END
from TP5.agent.state import AgentState
from TP5.agent.routing import route
from TP5.agent.nodes.classify_email import classify_email
from TP5.agent.nodes.stubs import (
    stub_reply,
    stub_ask_clarification,
    stub_escalate,
    stub_ignore,
)

def build_graph():
    # Initialisation du graphe avec le schéma de données AgentState
    g = StateGraph(AgentState)

    # Ajout des nœuds de traitement
    g.add_node("classify_email", classify_email)
    g.add_node("reply", stub_reply)
    g.add_node("ask_clarification", stub_ask_clarification)
    g.add_node("escalate", stub_escalate)
    g.add_node("ignore", stub_ignore)

    # Point d'entrée : On commence toujours par classer l'email
    g.set_entry_point("classify_email")

    # Routing conditionnel basé sur la décision du LLM
    g.add_conditional_edges(
        "classify_email",
        route,
        {
            "reply": "reply",
            "ask_clarification": "ask_clarification",
            "escalate": "escalate",
            "ignore": "ignore",
        },
    )

    # Chaque branche mène à la fin du graphe (pas de boucle pour le moment)
    g.add_edge("reply", END)
    g.add_edge("ask_clarification", END)
    g.add_edge("escalate", END)
    g.add_edge("ignore", END)

    return g.compile()
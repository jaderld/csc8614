from langgraph.graph import StateGraph, END
from TP5.agent.state import AgentState
from TP5.agent.nodes.classify_email import classify_email
from TP5.agent.nodes.retrieval import search_rag

def should_retrieve(state: AgentState):
    """Condition pour décider si on va vers le RAG ou si on termine."""
    if state.decision.needs_retrieval and state.budget.can_retrieve():
        return "retrieval"
    return END

def create_agent():
    # 1. Initialisation du graphe avec notre State typé
    workflow = StateGraph(AgentState)

    # 2. Ajout des nœuds
    workflow.add_node("classify", classify_email)
    workflow.add_node("retrieval", search_rag)

    # 3. Définition des liens
    workflow.set_entry_point("classify")
    
    # Transition conditionnelle après la classification
    workflow.add_conditional_edges(
        "classify",
        should_retrieve,
        {
            "retrieval": "retrieval",
            END: END
        }
    )

    # Après le RAG, on pourrait retourner à une classification/rédaction (boucle)
    workflow.add_edge("retrieval", END)

    return workflow.compile()
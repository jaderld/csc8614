from TP5.agent.state import AgentState, EvidenceDoc
from TP5.agent.logger import log_event
# Importe ici ta classe ou fonction RAG du TP4
# from TP4.rag_answer import MyRAG 

def search_rag(state: AgentState) -> AgentState:
    if not state.decision.needs_retrieval:
        return state

    log_event(state.run_id, "node_start", {"node": "retrieval", "query": state.decision.retrieval_query})
    
    state.budget.retrieval_attempts += 1
    
    # --- SIMULATION OU APPEL RAG REEL ---
    # Ici, tu appelles ton code du TP4 :
    # results = rag.search(state.decision.retrieval_query, k=4)
    
    # Exemple de structure pour remplir l'evidence :
    # for res in results:
    #    state.evidence.append(EvidenceDoc(
    #        doc_id=res.id, source=res.metadata['source'], 
    #        snippet=res.page_content, score=res.score
    #    ))
    
    log_event(state.run_id, "node_end", {"node": "retrieval", "found": len(state.evidence)})
    return state
import os
import uuid
from typing import List
from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph

OUT_MD = os.path.join("TP5", "batch_results.md")

def md_escape(s: str) -> str:
    return (str(s) or "").replace("|", "\\|").replace("\n", " ")

def main():
    emails = load_all_emails()
    app = build_graph()

    rows: List[str] = []
    rows.append("| email_id | subject | intent | category | risk | final_kind | tool_calls | retrieval_attempts | notes |")
    rows.append("|---|---|---|---|---|---|---:|---:|---|")

    print(f"Lancement du batch sur {len(emails)} emails...")

    for e in emails:
        run_id = str(uuid.uuid4())
        print(f"Processing {e['email_id']}...")
        
        state = AgentState(
            run_id=run_id,
            email_id=e["email_id"],
            subject=e["subject"],
            sender=e["from"],
            body=e["body"],
        )

        out = app.invoke(state)
        
        # Le résultat est un dict représentant le State final
        decision = out["decision"]
        budget = out["budget"]

        intent = decision.intent
        category = decision.category
        risk = decision.risk_level
        final_kind = out.get("final_kind", "unknown")
        tool_calls = budget.tool_calls_used
        retrieval_attempts = budget.retrieval_attempts

        notes = f"run={run_id}.jsonl"

        rows.append(
            "| "
            + " | ".join([
                md_escape(out["email_id"]),
                md_escape(out["subject"])[:40],
                intent,
                category,
                risk,
                final_kind,
                str(tool_calls),
                str(retrieval_attempts),
                md_escape(notes),
            ])
            + " |"
        )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {OUT_MD}")

if __name__ == "__main__":
    main()
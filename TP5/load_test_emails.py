import os
import re
from typing import Dict, List

EMAIL_DIR = os.path.join("TP5", "data", "test_emails")

RE_BODY = re.compile(r"CORPS:\s*<<<\s*(.*?)\s*>>>", re.DOTALL)
RE_ID = re.compile(r"email_id:\s*(\S+)")
RE_SUBJECT = re.compile(r"subject:\s*\"(.*)\"")
RE_FROM = re.compile(r"from:\s*\"(.*)\"")

def load_one_email(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    email_id_match = RE_ID.search(txt)
    subject_match = RE_SUBJECT.search(txt)
    from_match = RE_FROM.search(txt)
    body_match = RE_BODY.search(txt)

    # Extraction des données avec valeurs par défaut si non trouvé
    email_id = email_id_match.group(1) if email_id_match else "unknown"
    subject = subject_match.group(1) if subject_match else "No Subject"
    from_ = from_match.group(1) if from_match else "unknown@sender.com"
    body = body_match.group(1).strip() if body_match else ""

    return {
        "email_id": email_id,
        "subject": subject,
        "from": from_,
        "body": body,
        "path": path,
    }

def load_all_emails() -> List[Dict[str, str]]:
    if not os.path.exists(EMAIL_DIR):
        print(f"Erreur: Le dossier {EMAIL_DIR} n'existe pas.")
        return []
    
    files = []
    for fn in os.listdir(EMAIL_DIR):
        if fn.endswith(".md") or fn.endswith(".txt"):
            files.append(os.path.join(EMAIL_DIR, fn))

    # Tri pour garantir un ordre stable (E01, E02...)
    files.sort()

    emails = [load_one_email(p) for p in files]
    return emails

if __name__ == "__main__":
    emails = load_all_emails()
    print(f"Loaded {len(emails)} emails")
    for e in emails:
        print(f"- {e['email_id']}: {e['subject']} ({os.path.basename(e['path'])})")
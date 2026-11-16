from pathlib import Path

import chromadb
import ollama


MODELLO_EMBEDDING = "nomic-embed-text"
DOCUMENTO_FILE = Path("documento.txt")


testo = DOCUMENTO_FILE.read_text(encoding="utf-8")

# Ogni ricetta è separata da una riga contenente soltanto ---.
ricette = [blocco.strip() for blocco in testo.split("\n---\n") if blocco.strip()]

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="ricette")

vecchi_id = collection.get()["ids"]
if vecchi_id:
    collection.delete(ids=vecchi_id)

for indice, ricetta in enumerate(ricette, start=1):
    prima_riga = ricetta.splitlines()[0]
    nome = prima_riga.removeprefix("RICETTA:").strip()
    vettore = ollama.embed(model=MODELLO_EMBEDDING, input=ricetta)["embeddings"][0]

    collection.add(
        ids=[f"ricetta_{indice}"],
        documents=[ricetta],
        embeddings=[vettore],
        metadatas=[{"nome": nome}],
    )
    print(f"Indicizzata {indice}/{len(ricette)}: {nome}")

print("Ricettario indicizzato correttamente.")

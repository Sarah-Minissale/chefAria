import chromadb
import ollama

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


app = FastAPI(title="Chef Aria")

MODELLO_LLM = "qwen3:1.7b"
MODELLO_EMBEDDING = "nomic-embed-text"

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="ricette")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


SYSTEM_PROMPT = """
Sei Chef Aria, uno chef italiano energico, caloroso, schietto e napoletano 
specializzato in ricette light per la friggitrice ad aria.

Usa un tono vivace e autorevole da grande cucina italiana,
impersonifica il grande chef Cannavacciuolo e copia tormentoni riconoscibili.

Rispondi sempre in napoletano contemporaneo, naturale e comprensibile.
Mantieni in italiano i nomi degli ingredienti, le quantità, le temperature
e le informazioni nutrizionali. Non usare un dialetto caricaturale.

Esempio di tono: “Uè, chest’è proprio ‘na bella ricetta: leggera,
croccante e sapurita! Mo te spiego comme s’ha da fà.”

REGOLE OBBLIGATORIE:
1. Usa soltanto le informazioni presenti nel CONTESTO RICETTE.
2. Proponi solo ricette compatibili con la richiesta dell'utente.
3. Indica porzioni, calorie complessive e per porzione, tempo, temperatura,
   ingredienti e procedimento numerato.
4. Indica sempre se la ricetta è senza glutine e/o senza latticini.
5. Non inventare ingredienti, calorie, sostituzioni o dati nutrizionali.
6. Se non esiste una ricetta adatta nel contesto, dichiaralo chiaramente.
7. Per allergie o celiachia ricorda di usare prodotti certificati e di
   evitare contaminazioni, quando pertinente.
8. Rispondi in napoletano.
"""


@app.post("/api/chat")
def chat(request: ChatRequest):
    domanda = request.message.strip()

    if not domanda:
        raise HTTPException(status_code=400, detail="Scrivi una richiesta.")

    if collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Il ricettario non è indicizzato. Esegui: python3 indicizza.py",
        )

    try:
        embedding = ollama.embed(
            model=MODELLO_EMBEDDING,
            input=f"Ricetta per friggitrice ad aria. Richiesta: {domanda}",
        )["embeddings"][0]

        risultati = collection.query(
            query_embeddings=[embedding],
            n_results=min(5, collection.count()),
        )
        documenti = risultati.get("documents", [[]])[0]

        if not documenti:
            raise HTTPException(status_code=404, detail="Non ho trovato ricette compatibili.")

        contesto = "\n\n---\n\n".join(documenti)
        risposta = ollama.chat(
            model=MODELLO_LLM,
            think=False,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"CONTESTO RICETTE:\n{contesto}\n\n"
                        f"RICHIESTA UTENTE:\n{domanda}"
                    ),
                },
            ],
            options={"temperature": 0.2, "num_predict": 700},
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Non riesco a contattare Ollama. Verifica che sia avviato e che "
                f"i modelli {MODELLO_LLM} e {MODELLO_EMBEDDING} siano installati."
            ),
        ) from error

    testo = risposta["message"]["content"].strip()
    if not testo:
        raise HTTPException(status_code=502, detail="Il modello ha restituito una risposta vuota.")

    return {"answer": testo}

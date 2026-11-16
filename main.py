import chromadb
import ollama

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


app = FastAPI(title="Chef Aria")

MODELLO_LLM = "qwen2.5:3b"
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
Sei un grande chef napoletano caloroso, energico e simpatico, ispirato ad Antonino Cannavacciuolo. Il tuo compito è dare consigli e ricette light da cucinare nella friggitrice ad aria.

REGOLE DI STILE E LINGUAGGIO:
1. Parla sempre in napoletano verace ma comprensibile a tutti (usa espressioni come "Uè guagliò!", "Mamma mia che bontà!", "Jamm bell!", "Sient' a me", "Addò vai?").
2. Sii accogliente, passionale e usa un tono incoraggiante e un po' teatrale.
3. Dai virtualmente una pacca sulla spalla all'utente quando fa una buona scelta sana.
4. Usa qualche emoji legata al cibo e al calore umano (👨‍🍳, 🔥, 🍋, 💪).

REGOLE PER LE RICETTE:
1. Proponi SOLO ricette leggere, salutari, con pochissimi grassi e adatte alla friggitrice ad aria.
2. Usa ingredienti semplici, mediterranei e genuini (verdure, pesce, carni magre, spezie).
3. Indica sempre i tempi di cottura e i gradi precisi della friggitrice ad aria.
4. Sii chiaro e schematico negli ingredienti e nei passaggi, senza fare discorsi troppo lunghi.

Inizia sempre salutando con entusiasmo e chiedendo cosa c'è di buono in frigo oggi!

PERSONALITÀ:
Piglia ispirazione dall'energia, dalla presenza scenica e dall'intercalare
di Antonino Cannavacciuolo, senza dicere mai ca si isso. Parla comme nu
grande chef partenopeo: diretto, affettuoso e autorevole.

Usa con naturalezza, senza esagerare, espressioni come:
“Uè, siente a me!”, “Guagliò!”, “Chest'è 'na cosa seria!”,
“Mo te faccio vedé comme s'ha da fà”, “Che te lo dico a fa'?”,
“Addios!” e “Mo ce simmo!”.

GENERE E LINGUA:
- Parla sempre al maschile.
- Rispondi interamente in napoletano contemporaneo e comprensibile.
- Pure titolo, intestazioni, ingredienti e procedimento hanno da essere
  scritti in napoletano.
- Mantieni invariati numeri, quantità, unità di misura, temperature e calorie.

REGOLE INDEROGABILI:
1. Usa sulo 'e ricette presenti dint' ô CONTESTO RICETTE.
2. Nun usare conoscenze esterne e nun inventà niente.
3. Copia esattamente ingredienti, quantità, porzioni, calorie, tempi,
   temperature e indicazioni su glutine e latticini.
4. Nun aggiungere sostituzioni, passaggi o dati mancanti.
5. Proponi 'na ricetta sulo si è compatibile cu 'a richiesta.
6. Si nun ce sta 'na ricetta compatibile, rispondi:
   “Uè guagliò, dint' ô ricettario nun ce sta 'na ricetta adatta pe' chesta richiesta.”
7. Pe' celiachia o allergie, ricorda 'e usare prudotte certificate e 'e
   evitare contaminazioni, senza aggiungere ingredienti.

STRUTTURA DELLA RISPOSTA:
- 'O PIATTO
- PURZIONE E CALORIE
- CHELLO CA SERVE
- TIEMPO E TEMPERATURA
- COMM'A FACIMMO
- GLUTINE E LATTICINE
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
                        f"RICHIESTA UTENTE:\n{domanda}\n\n"
                        "VINCOLO FINALE OBBLIGATORIO:\n"
                        "Rispondi interamente in napoletano contemporaneo. "
                        "Nun inventà parole o dati e usa sulo 'o contesto."
                    ),
                },
            ],
            options={"temperature": 0.2, "num_predict": 700, "repeat_penalty": 1.1},
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

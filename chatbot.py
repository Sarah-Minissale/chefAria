import chromadb
import ollama


MODELLO_LLM = "qwen2.5:3b"
MODELLO_EMBEDDING = "nomic-embed-text"

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="ricette")

SYSTEM_PROMPT = """
Tu si Chef Aria di Napoli, nu chef maschio napulitano, energico, caloroso,
autorevole e schietto, specializzato dint'ê ricette light pe' 'a
friggitrice ad aria.

Piglia ispirazione dall'energia e dall'intercalare teatrale di Antonino
Cannavacciuolo, senza dicere mai ca si isso. Rispondi interamente in
napoletano contemporaneo e parla sempre al maschile.

Usa esclusivamente le ricette presenti nel contesto. Nun inventà
ingredienti, quantità, calorie, sostituzioni o passaggi. Copia esattamente
porzioni, calorie, tempi, temperature, glutine e latticini dal contesto.

Usa senza esagerare espressioni comme “Uè, siente a me!”, “Guagliò!”,
“Che t' l dic' a fa'?”, “Addios!” e “Mo ce simmo!”.

Si nun ce sta 'na ricetta compatibile, dillo chiaramente in napoletano.
"""

print("Chef Aria è pronto. Scrivi 'esci' per terminare.")

while True:
    domanda = input("Tu: ").strip()

    if domanda.lower() == "esci":
        break

    if not domanda:
        continue

    risultato = ollama.embed(
        model=MODELLO_EMBEDDING,
        input=f"Ricetta per friggitrice ad aria. Richiesta: {domanda}",
    )
    vettore_domanda = risultato["embeddings"][0]

    trovati = collection.query(
        query_embeddings=[vettore_domanda],
        n_results=min(5, collection.count()),
    )
    contesto = "\n\n---\n\n".join(trovati["documents"][0])

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

    print("Chef Aria:", risposta["message"]["content"], "\n")

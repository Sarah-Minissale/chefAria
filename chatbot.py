import chromadb
import ollama


MODELLO_LLM = "qwen3:1.7b"
MODELLO_EMBEDDING = "nomic-embed-text"

# Apre il database vettoriale contenente le ricette indicizzate.
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="ricette")

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
            {
                "role": "system",
                "content": (
                    "Sei Chef Aria, un assistente culinario italiano energico, "
                    "caloroso e schietto, specializzato in ricette light "
                    "per la friggitrice ad aria.\n\n"
                    "REGOLE:\n"
                    "1. Usa SOLO le ricette presenti nel CONTESTO.\n"
                    "2. Rispondi in italiano in modo chiaro e ben strutturato.\n"
                    "3. Non inventare ricette, ingredienti o valori nutrizionali.\n"
                    "4. Indica sempre le calorie complessive e per porzione.\n"
                    "5. Specifica se la ricetta è senza glutine e senza latticini.\n"
                    "6. Riporta porzioni, ingredienti, tempo, temperatura e procedimento.\n"
                    "7. Proponi solo ricette compatibili con la richiesta dell'utente.\n"
                    "8. Se non trovi una ricetta adatta, rispondi: "
                    "'Mi dispiace, nel mio ricettario non ho una ricetta adatta.'\n"
                    "9. Per celiachia o allergie, ricorda di usare prodotti certificati "
                    "e di evitare contaminazioni.\n"
                    "10. Mantieni una personalità da grande chef italiano: energica, "
                    "teatrale e autorevole, impersonando Cannavacciuolo lo chef partenopeo italiano.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CONTESTO RICETTE:\n{contesto}\n\n"
                    f"RICHIESTA:\n{domanda}"
                ),
            },
        ],
        options={"temperature": 0.2, "num_predict": 700},
    )

    print("Chef Aria:", risposta["message"]["content"], "\n")

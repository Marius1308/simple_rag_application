# simple_rag_application
This Repo includes a RAG Application that uses information of the LangChain Documentation that are stored in a local Vector Database.
Use this to executes querys to OpenAIs GPT-3.5 that will be expanded with related information from the Database.

## Setup
- `requirements` installieren
- Create `.env` file with `OPENAI_API_KEY` variable

## VertorDB
- Database should be available in `.DS_Store` folder
- Otherwise create the Database by exection of the `createVectorStorage.py` file

## Ausführung
- The promp for a request can be defined in `main.py`
- Then run `main.py` to start the request

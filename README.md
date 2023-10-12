# simple_rag_application
This Repo includes a RAG (Retrieval Augmented Generation) Application that uses information of the LangChain documentation that are stored in a local vector database.
Use this to executes querys to OpenAIs GPT-3.5 that will be expanded with related information from the database.

## Setup
- Install `requirements`
- Create `.env` file with `OPENAI_API_KEY` variable

## VertorDB
- Database should be available in `.DS_Store` folder
- Otherwise create the database by exection of the `createVectorStorage.py` file

## Usage
- The promp for a request can be defined in `main.py`
- Then run `main.py` to start the request

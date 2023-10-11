from __future__ import annotations
import os

from typing import Any, Dict, List, Optional, Sequence
from langchain import BasePromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessageChunk
from langchain.vectorstores import Chroma
from pydantic import BaseModel, Field, Extra

apiKey = os.getenv("OPENAI_API_KEY")


class KeyQuestions(BaseModel):
    """The Key Questions."""

    questions: List[str] = Field(..., description="the key Questions")


class contextChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel[BaseMessageChunk]
    iterations: int = 1
    output_key: str = "text"  #: :meta private:
    context: str = ""
    db: Chroma | None = None

    class Config(Chain.Config):
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs).to_string()

        self.db = Chroma(
            persist_directory="./langchainPages/db/chroma_db",
            embedding_function=OpenAIEmbeddings(openai_api_key=apiKey, client=None),
        )

        for _ in range(self.iterations):
            keyQuestions = self.getKeyQuestions(prompt_value, run_manager=run_manager)

            for question in keyQuestions.questions:
                self.query_database_and_add_to_context(question)

        response = self.get_answer_with_context(prompt_value, run_manager=run_manager)

        return {self.output_key: response}

    def getKeyQuestions(
        self,
        prompt_value: str,
        run_manager: Optional[CallbackManagerForChainRun],
    ):
        prompt_msgs: Sequence[SystemMessage | HumanMessagePromptTemplate] = []
        input = prompt_value
        input_variables = ["input"]
        prompt_msgs.append(
            SystemMessage(
                content="You are a System that can extract the key questions, that need to be answered for a detailed answer to a given question."
            )
        )
        if self.context != "":
            prompt_msgs.append(
                SystemMessage(
                    content="You have a context of already given information so the key questions should ask for missing information."
                )
            )
            prompt_msgs.append(
                HumanMessagePromptTemplate.from_template(
                    "This is your context: {context}"
                )
            )
            input_variables.append("context")
            input = {"input": prompt_value, "context": self.context}

        prompt_msgs.append(
            HumanMessagePromptTemplate.from_template(
                "Extract the 3 key question of following question: {input}"
            )
        )

        prompt = ChatPromptTemplate(
            messages=prompt_msgs, input_variables=input_variables
        )

        chain = create_structured_output_chain(
            output_schema=KeyQuestions,
            llm=self.llm,
            prompt=prompt,
            verbose=True,
        )

        keyQuestions: KeyQuestions = chain.run(
            input, callbacks=run_manager.get_child() if run_manager else None
        )

        return keyQuestions

    def query_database_and_add_to_context(self, input: str, answers: int = 2):
        if self.db is None:
            return
        docs = self.db.similarity_search(input, k=answers)
        for doc in docs:
            if doc.page_content not in self.context:
                self.context += input + "\n" + doc.page_content + "\n\n"

    def get_answer_with_context(
        self, prompt_value: str, run_manager: Optional[CallbackManagerForChainRun]
    ):
        prompt_msgs = [
            SystemMessage(
                content="You are a System that can answer questions based on a given context."
            ),
            HumanMessagePromptTemplate.from_template("This is your context: {context}"),
            HumanMessagePromptTemplate.from_template(
                "Now answer the following question: {input}"
            ),
        ]

        prompt = ChatPromptTemplate(
            messages=prompt_msgs, input_variables=["input", "context"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)

        return chain.run(
            {"input": prompt_value, "context": self.context},
            callbacks=run_manager.get_child() if run_manager else None,
        )

    @property
    def _chain_type(self) -> str:
        return "context_chain"

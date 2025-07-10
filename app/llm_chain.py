
# from langchain.chat_models import ChatOpenA
# from langchain.chains import RetrievalQA

# def create_qa_chain(vectorstore):
#    llm = ChatOpenAI(
#        temperature=0,
#        model_name="gpt-3.5-turbo"
#    )

#   retriever = vectorstore.as_retriever(search_type="similarity", search_k=4)

#    qa_chain = RetrievalQA.from_chain_type(
#        llm=llm,
#        retriever=retriever,
#        chain_type="stuff"
#   )

#   return qa_chain

from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
from langchain_core.runnables import Runnable
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import PrivateAttr
import os

# Load environment variables
load_dotenv()


class CustomHuggingFaceLLM(LLM, Runnable):
    repo_id: str
    api_token: str
    temperature: float = 0.2
    max_new_tokens: int = 512

    _client: InferenceClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = InferenceClient(
            self.repo_id,
            token=self.api_token
        )

    def invoke(self, prompt, **kwargs):
        response = self._client.text_generation(
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )
        return response.generated_text

    def _call(self, prompt: str, stop=None):
        return self.invoke(prompt)

    @property
    def _llm_type(self):
        return "custom_huggingface_llm"


def create_qa_chain(vectordb):
    llm = CustomHuggingFaceLLM(
        repo_id="google/flan-t5-xl",
        api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.2,
        max_new_tokens=512
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt_template = PromptTemplate(
        template="Answer the question based on the context:\n\n{context}\n\nQuestion: {question}",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )
    return qa_chain

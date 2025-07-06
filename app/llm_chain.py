
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

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()


def create_qa_chain(vectordb):
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-3n-E4B-it",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.2,
        max_length=512
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

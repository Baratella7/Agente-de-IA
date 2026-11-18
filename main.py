# pip install python-dotenv langchain langchain-openai langchain-community langchain-chroma chromadb openai pypdf
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB= "DB"

ChatPromptTemplate = """
responta a pergunta do usuario:
{pergunta}

 com base nesses informações:

 {base_conhecimento}

 Se voce nao encontrar a resposta para a pergunta do usuario nesses informações,
 responda nao sei te dizer isso"""

pergunta = input("Escreva sua pergunta: ")

# carregando o banco de dados
funcao_embedding = OpenAIEmbeddings()
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)


# comparando a pergunta do usuario (embedding) com o meu banco de dados
resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)
print(resultados)
print(len(resultados))


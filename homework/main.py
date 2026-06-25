from dotenv import load_dotenv
from openai import OpenAI

from rag_ingest import load_data, build_min_index, build_sqlite_index
from rag_helper import RAGBase, RAGAgent


def run_rag_base(index, openai_client, question):
    print("=" * 50)
    print("STEP 1: Traditional RAG")
    print("=" * 50)
    answer, usage, cost = RAGBase(index, openai_client).rag(question)
    print(f"\nAnswer:\n{answer}")
    print(f"\nUsage: {usage} | Cost: {cost}\n")


def run_rag_agent(index, question):
    print("=" * 50)
    print("STEP 2: Agentic RAG")
    print("=" * 50)
    messages, tokens, cost = RAGAgent(index).rag(question)
    print(f"\nFinal answer:\n{messages[-1].content[0].text}")
    print(f"\nTokens: {tokens} | Cost: {cost}\n")


def main():
    load_dotenv()

    print("=" * 50)
    print("Loading data and building index")
    print("=" * 50)
    
    # toggle between minsearch and sqlite index
    # index = build_min_index()
    index = build_sqlite_index(db_path="rag_index_text.db")

    question = "How does the agentic loop work, and how is it different from plain RAG?"
    openai_client = OpenAI()

    run_rag_base(index, openai_client, question)
    run_rag_agent(index, question)


if __name__ == "__main__":
    main()

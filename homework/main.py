from dotenv import load_dotenv
from openai import OpenAI

from rag_ingest import build_index
from rag_helper import RAGBase, agent_rag


def main():
    load_dotenv()

    print("=" * 50)
    print("Building search index")
    print("=" * 50)
    index = build_index()

    question = "How does the agentic loop work, and how is it different from plain RAG?"

    # Step 1: Traditional RAG
    print("=" * 50)
    print("STEP 1: Traditional RAG")
    print("=" * 50)

    openai_client = OpenAI()
    answer, usage = RAGBase(index, openai_client).rag(question)
    print(f"\nAnswer:\n{answer}")
    print(f"\nUsage: {usage}\n")

    # Step 2: Agentic RAG
    print("=" * 50)
    print("STEP 2: Agentic RAG")
    print("=" * 50)

    messages = agent_rag(index, question)
    print(f"\nFinal answer:\n{messages[-1].content[0].text}")


if __name__ == "__main__":
    main()

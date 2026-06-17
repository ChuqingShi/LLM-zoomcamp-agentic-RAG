from dotenv import load_dotenv
#from openai import OpenAI

from rag_ingest import index
#from rag_helper import RAGBase

from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback


def search(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """
    Search the index for the given query and return the results.
    """
    return index.search(query, num_results=num_results)


def agent_rag(instructions, question):
    agent_tools = Tools()
    agent_tools.add_tool(search)
    
    chat_interface = IPythonChatInterface()
    callback = DisplayingRunnerCallback(chat_interface)
    
    runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model='gpt-5.4-mini')
    )
    
    result = runner.loop(
        prompt=question,
        callback=callback,
    )
    return result.all_messages
    

def main():
    load_dotenv()
    # openai_client = OpenAI()
    
    # question = "How does the agentic loop keep calling the model until it stops?"
    # answer, usage = RAGBase(index, openai_client).rag(question)
    # print(answer)
    # print(usage)
    
    instructions = """
    You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering.
    """.strip()
    question = 'How does the agentic loop work, and how is it different from plain RAG?'
    
    messages = agent_rag(instructions, question)
    print(messages[-1].content[0].text)
    

if __name__ == "__main__":
    main()

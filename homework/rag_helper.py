from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat.interface import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback


INSTRUCTIONS_BASE = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()

INSTRUCTIONS_AGENT = '''
You're a course teaching assistant. Answer the student's question using the search tool. 

Make multiple searches with different keywords before answering.
'''


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS_BASE,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        print(f"[RAGBase] Searching for: '{query}' (top {num_results})")
        results = self.index.search(query, num_results=num_results)
        print(f"[RAGBase] Found {len(results)} results.")
        return results

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['filename'])
            lines.append(doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        print(f"[RAGBase] Built prompt context ({len(context)} chars).")
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        print(f"[RAGBase] Calling LLM (model={self.model})...")
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        print(f"[RAGBase] Response received. Tokens used: {response.usage}")
        return response

    def rag(self, query):
        print(f"[RAGBase] Query: '{query}'")
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        answer = response.output_text
        usage = response.usage
        print(f"[RAGBase] Answer received ({len(answer)} chars).")
        return answer, usage



def make_search_tool(index):
    def search(query: str, num_results: int = 5) -> list[dict[str, str]]:
        """Search the index for the given query and return the results."""
        print(f"[Agent] Tool call: search('{query}')")
        results = index.search(query, num_results=num_results)
        print(f"[Agent] Tool returned {len(results)} results.")
        return results

    return search


def agent_rag(index, question, instructions=INSTRUCTIONS_AGENT):
    search = make_search_tool(index)

    print("[Agent] Setting up agentic runner...")
    agent_tools = Tools()
    agent_tools.add_tool(search)

    chat_interface = IPythonChatInterface()
    callback = DisplayingRunnerCallback(chat_interface)
    runner = OpenAIResponsesRunner(
        tools=agent_tools,
        developer_prompt=instructions,
        chat_interface=chat_interface,
        llm_client=OpenAIClient(model='gpt-5.4-mini'),
    )

    print("[Agent] Running agentic loop...")
    result = runner.loop(prompt=question, callback=callback)
    print("[Agent] Loop complete.\n")
    return result.all_messages

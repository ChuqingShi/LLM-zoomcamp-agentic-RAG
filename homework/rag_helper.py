from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat.interface import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback
from toyaikit.pricing import PricingConfig

_pricing = (
    PricingConfig()
)  # Initialize at the module level, so it can be reused across all RAGBase and RAGAgent instances without needing to re-initialize it each time. This is more efficient and ensures consistent pricing calculations throughout the application.

INSTRUCTIONS_BASE = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

INSTRUCTIONS_AGENT = """
You're a course teaching assistant. Answer the student's question using the search tool. 

Make multiple searches with different keywords before answering.
"""


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS_BASE,
        prompt_template=PROMPT_TEMPLATE,
        model="gpt-5.4-mini",
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
            lines.append(doc["filename"])
            lines.append(doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        print(f"[RAGBase] Built prompt context ({len(context)} chars).")
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        print(f"[RAGBase] Calling LLM (model={self.model})...")
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model, input=input_messages
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
        cost = _pricing.calculate_cost(
            self.model, usage.input_tokens, usage.output_tokens
        )
        print(f"[RAGBase] Answer received ({len(answer)} chars). Cost: {cost}")
        return answer, usage, cost


class RAGAgent:

    def __init__(self, index, instructions=INSTRUCTIONS_AGENT, model="gpt-5.4-mini"):
        self.model = model
        self.instructions = instructions

        def search(query: str, num_results: int = 5) -> list[dict[str, str]]:
            """Search the index for the given query and return the results."""
            print(f"[RAGAgent] Tool call: search('{query}')")
            results = index.search(query, num_results=num_results)
            print(f"[RAGAgent] Tool returned {len(results)} results.")
            return results

        # self.index is not needed because the search closure already captures index directly from the __init__ parameter.
        #   The index is an implementation detail of the tool, nothing outside __init__ needs it.
        #   Storing it as self.index would expose it unnecessarily.
        #   Compare with RAGBase, where self.index is needed because search() is a regular method that accesses it via self.

        agent_tools = Tools()
        agent_tools.add_tool(search)
        # self.agent_tools is not needed since they're only used to initialize the runner, which is done immediately in __init__.

        self.chat_interface = IPythonChatInterface()
        self.runner = OpenAIResponsesRunner(
            tools=agent_tools,
            developer_prompt=instructions,
            chat_interface=self.chat_interface,
            llm_client=OpenAIClient(model=model),
        )

    def rag(self, query):
        print(f"[RAGAgent] Query: '{query}'")
        print("[RAGAgent] Running agentic loop...")
        callback = DisplayingRunnerCallback(self.chat_interface)
        result = self.runner.loop(prompt=query, callback=callback)
        messages = result.all_messages
        tokens = result.tokens
        cost = result.cost
        print(f"[RAGAgent] Loop complete. Cost: {cost}\n")
        return messages, tokens, cost

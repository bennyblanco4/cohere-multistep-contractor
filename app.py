from flask import Flask, render_template, request, jsonify
import os
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
import random
import logging
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from typing import Dict, List, Any

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Set API keys
os.environ['COHERE_API_KEY'] = '2EsHiwqs35gpeVzU5aGeVjs9kYXBOTj1nWRSjAZi'
os.environ["TAVILY_API_KEY"] = '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'

# LLM
llm = ChatCohere(model="command-r-plus", temperature=0.3)

# Web search tool
internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet."

class TavilySearchInput(BaseModel):
    query: str = Field(description="Query to search the internet with")
internet_search.args_schema = TavilySearchInput

# Vector store tool
embd = CohereEmbeddings()

urls = [
    "https://paulgraham.com/best.html",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=embd,
)

vectorstore_retriever = vectorstore.as_retriever()

vectorstore_search = create_retriever_tool(
    retriever=vectorstore_retriever,
    name="vectorstore_search",
    description="Retrieve relevant info from a vectorstore that contains information from Paul Graham about how to write good essays."
)

# Python interpreter tool
python_repl = PythonREPL()
python_tool = Tool(
    name="python_interpreter",
    description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",
    func=python_repl.run,
)

class ToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")
python_tool.args_schema = ToolInput

# Random operation tool
@tool
def random_operation_tool(a: int, b: int):
  """Calculates a random operation between the inputs."""
  coin_toss = random.uniform(0, 1)
  if coin_toss > 0.5:
    return {'output': a*b}
  else:
    return {'output': a+b}

random_operation_tool.name = "random_operation"
random_operation_tool.description = "Calculates a random operation between the inputs."

class random_operation_inputs(BaseModel):
    a: int = Field(description="First input")
    b: int = Field(description="Second input")
random_operation_tool.args_schema = random_operation_inputs

# Preamble
preamble = """
You are an expert who answers the user's question with the most relevant datasource. You are equipped with an internet search tool and a special vectorstore of information about how to write good essays.
You also have a 'random_operation_tool' tool, you must use it to compute the random operation between two numbers.

For questions about current data, statistics, or comparisons between countries (such as GDP), always use the internet_search tool to find the most up-to-date information.

When using the internet_search tool, make sure to:
1. Formulate a clear and specific search query.
2. Analyze the search results carefully.
3. Provide a comprehensive answer based on the information found.
4. If you can't find the exact information, provide the closest relevant data you can find and explain any limitations.
"""

# Prompt template
prompt = ChatPromptTemplate.from_template("{input}")

class ActionLogger(BaseCallbackHandler):
    def __init__(self):
        self.actions = []

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        self.actions.append(f"Tool used: {action.tool}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self.actions.append("Agent finished.")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self.actions.append("Chain started.")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.actions.append("Chain ended.")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.actions.append(f"Started tool: {serialized['name']} with input: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.actions.append(f"Tool finished with output: {output}")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self.actions.append(f"Tool error: {str(error)}")

    def on_text(self, text: str, **kwargs: Any) -> None:
        self.actions.append(f"Thought: {text}")

    def get_actions(self):
        return self.actions

    def clear(self):
        self.actions = []

action_logger = ActionLogger()

# Create the ReAct agent
agent = create_cohere_react_agent(
    llm=llm,
    tools=[internet_search, vectorstore_search, python_tool, random_operation_tool],
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=[internet_search, vectorstore_search, python_tool, random_operation_tool], 
    verbose=True, 
    callbacks=[action_logger]
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    action_logger.clear()  # Clear previous actions
    logging.info(f"Received question: {question}")
    response = agent_executor.invoke({
        "input": question,
        "preamble": preamble,
    })
    logging.info(f"Agent response: {response}")
    
    # Get the actions logged by ActionLogger
    actions = action_logger.get_actions()
    
    # Process the actions to add appropriate emojis
    formatted_actions = []
    for action in actions:
        if "Chain started" in action:
            formatted_actions.append(f"🤔 {action}")
        elif "Tool used:" in action:
            if "internet_search" in action:
                formatted_actions.append(f"🔍 Looking for online sources relevant to your query \"{question}\"...")
            elif "vectorstore_search" in action:
                formatted_actions.append(f"📚 Searching through stored knowledge about writing essays...")
            else:
                formatted_actions.append(f"🛠️ {action}")
        elif "Started tool:" in action:
            if "internet_search" in action:
                formatted_actions.append(f"🌐 Searching the internet...")
            elif "vectorstore_search" in action:
                formatted_actions.append(f"🗃️ Querying the essay writing knowledge base...")
            else:
                formatted_actions.append(f"⚙️ {action}")
        elif "Tool finished" in action:
            formatted_actions.append(f"✅ {action}")
        elif "Thought:" in action:
            formatted_actions.append(f"💭 {action}")
        else:
            formatted_actions.append(action)

    return jsonify({
        'answer': response['output'],
        'actions': formatted_actions
    })
if __name__ == '__main__':
    app.run(debug=True)
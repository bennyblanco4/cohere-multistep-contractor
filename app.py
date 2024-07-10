from flask import Flask, render_template, request, Response
import json
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
import matplotlib.pyplot as plt
import re
import urllib.parse
import traceback
from queue import Queue
from threading import Thread

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    "https://news.ycombinator.com/"
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
    description="Retrieve relevant info from a vectorstore that contains information of HVAC companies with their contact information"
)

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
    def __init__(self, queue):
        self.queue = queue

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        self.queue.put(f"🛠️ Tool used: {action.tool}")
        logger.debug(f"Agent action: {action.tool}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self.queue.put("✅ Agent finished")
        logger.debug("Agent finished")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self.queue.put("🔎 Searching")
        logger.debug("Chain started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.queue.put("✅ Done")
        logger.debug("Chain ended")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.queue.put(f"Started tool: {serialized['name']} with input: {input_str}")
        logger.debug(f"Started tool: {serialized['name']} with input: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.queue.put(f"Tool finished with output: {output}")
        logger.debug(f"Tool finished with output: {output}")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self.queue.put(f"Tool error: {str(error)}")
        logger.error(f"Tool error: {str(error)}")

    def on_text(self, text: str, **kwargs: Any) -> None:
        self.queue.put(f"Thought: {text}")
        logger.debug(f"Thought: {text}")

# Create the ReAct agent
agent = create_cohere_react_agent(
    llm=llm,
    tools=[internet_search, vectorstore_search],
    prompt=prompt,
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask')
def ask():
    question = request.args.get('question')
    logger.info(f"Received question: {question}")

    def generate():
        queue = Queue()
        action_logger = ActionLogger(queue)

        agent_executor = AgentExecutor(
            agent=agent, 
            tools=[internet_search, vectorstore_search], 
            verbose=True, 
            callbacks=[action_logger]
        )

        def run_agent():
            try:
                response = agent_executor.invoke({
                    "input": question,
                    "preamble": preamble,
                })
                queue.put(("answer", response['output']))
            except Exception as e:
                logger.error(f"Error in run_agent: {str(e)}")
                logger.error(traceback.format_exc())
                queue.put(("error", f"An error occurred: {str(e)}"))

        Thread(target=run_agent).start()

        yield f"data: {json.dumps({'type': 'start', 'content': 'Starting agent...'})}\n\n"

        while True:
            try:
                item = queue.get(timeout=1)
                if isinstance(item, tuple):
                    item_type, content = item
                    yield f"data: {json.dumps({'type': item_type, 'content': content})}\n\n"
                    if item_type in ('answer', 'error'):
                        break
                else:
                    yield f"data: {json.dumps({'type': 'action', 'content': item})}\n\n"
            except:
                continue

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
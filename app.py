from dotenv import load_dotenv
import os
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
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import re
import urllib.parse
import traceback
from queue import Queue
from threading import Thread

load_dotenv()


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set API keys
# Replace the existing API key assignments with:
cohere_api_key = os.getenv('COHERE_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

# Use the variables in your code
os.environ['COHERE_API_KEY'] = cohere_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# LLM
llm = ChatCohere(model="command-r-plus", temperature=0.3)

# Web search tool
internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet with contact info"

class TavilySearchInput(BaseModel):
    query: str = Field(description="Query to search the internet with")
internet_search.args_schema = TavilySearchInput

# Vector store tool
embd = CohereEmbeddings()

def parse_query_with_cohere(query):
    chat_model = ChatCohere(model="command", temperature=0)
    
    prompt = f"""
    Parse the following query into a service, location, and country for a Yellow Pages search including phone numbers.
    Query: {query}
    
    Respond in the following format:
    Service: [extracted service]
    Location: [extracted location]
    Country: [USA/Canada/Other]
    
    If a location isn't specified, use 'Toronto, ON' and 'Canada' as defaults.
    If a country isn't clearly specified, infer it from the location if possible.
    
    """
    
    messages = [HumanMessage(content=prompt)]
    response = chat_model.invoke(messages)
    
    # Parse the response
    lines = response.content.strip().split('\n')
    service = "General service"
    location = "Toronto, ON"
    country = "Canada"
    
    for line in lines:
        if line.startswith("Service:"):
            service = line.split(':', 1)[1].strip()
        elif line.startswith("Location:"):
            location = line.split(':', 1)[1].strip()
        elif line.startswith("Country:"):
            country = line.split(':', 1)[1].strip()
    
    return service, location, country

def generate_yellowpages_url(query):
    service, location, country = parse_query_with_cohere(query)

    if country.lower() == 'usa':
        service = urllib.parse.quote(service)
        location = urllib.parse.quote(location)
        url = f"https://www.yellowpages.com/search?search_terms={service}&geo_location_terms={location}"
    elif country.lower() == 'canada':
        service = urllib.parse.quote(service.replace(' ', '+'))
        location = urllib.parse.quote(location.replace(' ', '+'))
        url = f"https://www.yellowpages.ca/search/si/1/{service}/{location}"
    else:
        # For other countries, return None to indicate we should use internet search
        logger.info(f"Country {country} not supported, falling back to internet search")
        return None

    logger.info(f"Generated URL: {url}")
    return url

def create_yellowpages_tool(query):
    url = generate_yellowpages_url(query)
    if url is None:
        return None  # No Yellow Pages tool for this query
    
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=embd,
    )
    
    vectorstore_retriever = vectorstore.as_retriever()

    return create_retriever_tool(
        retriever=vectorstore_retriever,
        name="yellowpages_search",
        description=f"Retrieve relevant info about {query} from Yellow Pages"
    )

# Preamble
preamble = """
You are an expert who answers the user's question with the most relevant datasource. You are equipped with an internet search tool and, for queries about locations in the USA or Canada, a Yellow Pages search tool.

For questions about local businesses or services in the USA or Canada, use the yellowpages_search tool to find the most relevant information. The query has been intelligently parsed to extract the service and location.

For questions about locations outside the USA and Canada, or for any other type of query, use the internet_search tool.

When using the yellowpages_search tool:
1. Analyze the search results carefully.
2. Provide a comprehensive answer based on the information found, including business names and contact information when available.
3. If you can't find exact matches, provide the closest relevant data you can find and explain any limitations.
4. ALWAYS GET AND LIST PHONE NUMBERS OF BUSINESSES.
When using the internet_search tool:
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
        self.queue.put("✅ Agent finished.")
        logger.debug("Agent finished")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self.queue.put("🔎 Search started.")
        logger.debug("Chain started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self.queue.put("✅ Agent finished.")
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask')
def ask():
    question = request.args.get('question')
    logger.info(f"Received question: {question}")

    # Generate and log the URL
    url = generate_yellowpages_url(question)
    if url:
        logger.info(f"Generated Yellow Pages URL: {url}")
    else:
        logger.info("Using internet search instead of Yellow Pages")

    def generate():
        if url:
            yield f"data: {json.dumps({'type': 'url', 'content': url})}\n\n"
        
        queue = Queue()
        action_logger = ActionLogger(queue)

        yellowpages_tool = create_yellowpages_tool(question)
        tools = [internet_search]
        if yellowpages_tool:
            tools.append(yellowpages_tool)
        
        agent = create_cohere_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )

        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
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
import os
from dotenv import load_dotenv
load_dotenv()

#1 Setup API keys from Groq tavily and open ai

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

#2 set up and tools 
from langchain_groq import ChatGroq  # for not using a large model thats why
from langchain_openai import ChatOpenAI 
from langchain_community.tools.tavily_search import TavilySearchResults #for search other website on google
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

openai_llm=ChatOpenAI(model="gpt-40-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

#3 set up ai agent with search tool functionality

def get_response_from_ai_agent(llm_id,query,allow_search,system_prompt,provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id) 

    tools=[TavilySearchResults(max_results=2)] if allow_search else[]
    agent=create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message,AIMessage)]
    return ai_messages[-1]
   
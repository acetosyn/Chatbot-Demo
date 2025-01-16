import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import AnthropicLLM, ChatAnthropic
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.agents import initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from langchain.tools import tool
from datetime import datetime
import dotenv




st.title("ACE BOT")
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
prompt = hub.pull("hwchase17/openai-functions-agent")

@tool
def get_date_time(query:str)-> str:
    """
    A tool that tells today's date and current time. Use this when a user ask for the current date or time.
    """

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(" ")
    return f"The Current date is {current_time} and the current time is {current_time}"

tools = [search,wikipedia, get_date_time]

dotenv.load_dotenv()
api_key = os.environ.get("api_key")
api_key_claude = os.environ.get("api_key_claude")
groq_api_key  = os.environ.get("groq_api_key")
mistral_api = os.environ.get("groq_api_key")

if "memory" not in st.session_state:
    st.session_state.memory = []

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "OpenAI's GPT"

if st.button ("clear"):
    st.session_state.memory = []
    st.session_state.chat_memory = []


st.session_state.selected_model = st.selectbox("Select a Desired model", ["OpenAI's GPT", "Anthropic's Claude", "Meta's Llama", "Mistral's Mistral"])



if st.session_state.selected_model == "OpenAI's GPT":
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

elif st.session_state.selected_model == "Anthropic's Claude":
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key =api_key_claude)

elif st.session_state.selected_model == "Meta's Llama":
    model = ChatGroq(model="llama-3.1-70b-versatile", api_key =groq_api_key)


elif st.session_state.selected_model == "Mistral's Mistral":
    model = ChatGroq(model="mixtral-8x7b-32768", api_key = mistral_api)
   

model = model.bind_tools(tools)

chain = RunnablePassthrough.assign(
                agent_scratchpad = lambda x: format_to_tool_messages(x["intermediate_steps"])
                ) | prompt | model | ToolsAgentOutputParser()


agent_executor = AgentExecutor(agent=chain, tools=tools,
                                           handle_parsing_errors=True,
                                           return_intermediate_steps=True)
for chat in st.session_state.chat_memory:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])


if prompt:= st.chat_input("what's up gee?"):
    with st.chat_message("user"):
        st.session_state.memory.append(HumanMessage(content=prompt))
        st.markdown(prompt)
        st.session_state.chat_memory.append({"role":"user", "message":prompt})

    with st.chat_message("ai"):
        response = agent_executor.invoke ({"input" : prompt, 'chat_history': st.session_state.memory})["output"]
        if st.session_state.selected_model == "Anthropic's Claude":
            response = response[0]["text"].split("</thinking>")[-1]
        st.markdown(response)
        st.session_state.memory.append(AIMessage(content=response))
        st.session_state.chat_memory.append({"role":"ai", "message":response})

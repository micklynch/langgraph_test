import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

from langchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
openai_llm = ChatOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, model="deepseek/deepseek-chat")
weather = OpenWeatherMapAPIWrapper()

# Node to extract city from user input
def agent(input_1):
  res = openai_llm.invoke(f"""
  You are given one question and you have to extract city name from it
  Don't respond anything except the city name and don't reply anything if you can't find city name

  Here is the question:
  {input_1}
  """)
  return res.content
  
# Node to find weather information
def weather_tool(input_2):
  data = weather.run(input_2)
  return data

from langgraph.graph import Graph


workflow = Graph()

workflow.add_node("agent", agent)
workflow.add_node("weather", weather_tool)

# Connecting 2 nodes
workflow.add_edge('agent', 'weather')

workflow.set_entry_point("agent")
workflow.set_finish_point("weather")

app = workflow.compile()

app.invoke("what is the weather in Toronto?")

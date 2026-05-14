#sql_agent.py for studio
import pathlib
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model

# Load global .env
load_dotenv('/Users/sandeepvaid/Documents/Personal/agentic ai/.env')

from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize an LLM using Gemini
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# Connect to PostgreSQL database
db_uri = "postgresql+psycopg2://postgres:postgres@localhost:5432/test-1.6-aug"
db = SQLDatabase.from_uri(
    db_uri,
    engine_args={"pool_size": 500}
)

# print(f"Available tables: {db.get_usable_table_names()}")
# Create the tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

# for tool in tools:
#     print(f"{tool.name}: {tool.description}\n")

# Use create_agent
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)

question = "Total number of schools we have"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
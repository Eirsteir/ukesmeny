from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = "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"


if __name__ == "__main__":
    pg_uri = f"postgresql+psycopg2://admin:password@localhost:5432/db"
    db = SQLDatabase.from_uri(
        pg_uri, include_tables=["recipe", "recipe_ingredient", "ingredient"]
    )
    gpt = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo"
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=gpt)
    agent_executor = create_sql_agent(
        llm=gpt,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    question = "Recommend a weekly menu of 7 varied recipies using some of the same ingredients. Return all 7 recipies in a numbered list with all the ingredients needed aggregated to a list. I do not like Reinsdyrtaco i potetlefse med tyttebærrømme"
    agent_executor.run(question)

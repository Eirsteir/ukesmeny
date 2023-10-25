import logging

import chainlit as cl
import lancedb
import pandas as pd
from langchain import LLMChain
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage
from langchain.vectorstores import LanceDB

OPENAI_MODEL = "gpt-3.5-turbo-16k"  # "gpt-3.5-turbo"

logger = logging.getLogger(__name__)

recipes = pd.read_pickle("data/preprocessed/recipes.pkl")
recipes.drop_duplicates(subset=["id"], inplace=True)  # Must for this dataset
recipes.drop("target", axis=1, inplace=True)

uri = "dataset/chainlit-recipes-lancedb"
db = lancedb.connect(uri)
table = db.create_table("recipes", recipes, mode="overwrite")


@cl.on_chat_start
async def main():
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(LanceDB)(connection=table, embedding=embeddings)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    tool = create_retriever_tool(
        docsearch.as_retriever(search_kwargs={"k": 10}),  # kan kalle denne dynamisk for menyer
        "recommend_recipes_or_menus",
        "Recommends dinner recipes or menus based on user preferences. Invocations must be in norwegian.",
    )
    tools = [tool]
    system_message = SystemMessage(
        content=(
            """You are a recommender chatting with the user to provide dinner recipe recommendation. You must follow the instructions below during chat. You can recommend either a recipe plan for a week or single recipes. 
                If you do not have enough information about user preference, you should ask the user for his preference.
                If you have enough information about user preference, you can give recommendation. The recommendation list can contain items that the dialog mentioned before.
                Recommendations are given by using the tool recommend_recipes_or_menus with a query you think matches the conversation and user preferences. The query must be in norwegian."""
        )
    )

    qa = create_conversational_retrieval_agent(
        llm,
        tools,
        system_message=system_message,
        remember_intermediate_steps=True,
        # max_tokens_limit=4000,
        verbose=True,
    )

    # Store the chain in the user session
    cl.user_session.set("llm_chain", qa)


@cl.on_message
async def main(message: cl.Message):
    print(message)
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post-processing here
    await cl.Message(content=res["output"]).send()

# HOW TO RUN: chainlit run app.py -w

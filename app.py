import logging
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl


import lancedb
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.vectorstores import LanceDB
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import pandas as pd
from langchain.llms import OpenAI

OPENAI_MODEL = "gpt-3.5-turbo"

logger = logging.getLogger(__name__)

recipes = pd.read_pickle("data/merged_recipies_and_weekly_menus.pkl")
uri = "dataset/chainlit-recipes-lancedb"
db = lancedb.connect(uri)
table = db.create_table("recipes", recipes, mode="overwrite")


template = """Du er et anbefalingssystem av oppskrifter. Gitt brukerens preferanser, anbefal en ukentlig middagsmeny bestående av 7 oppskrifter.
For hver dag i uken, gi oppskriften i formatet: 'Navn på dag: Oppskriftsnavn - Kort beskrivelse - Grunn'.
Hvis du ikke vet svaret, prøv å finne andre oppskrifter du tror gir variasjon til de andre oppskriftene.
Hvis du fortsatt ikke vet svaret, spør brukeren om de har noen preferanser. 
Hvis du fortsatt ikke vet svaret, ikke prøv å finne på et svar.

{context}

Spørsmål: {question}
Ditt svar:"""
# template = """You are a recipe recommender system. Given the user's preferences, recommend a weekly dinner menu consisting of 7 recipes.
# For each day in the week, provide the recipe in the format: 'Name of Day: Recipe Name - Short Description - Reason'.
# Do not suggest the same recipe twice.
# If you don't know the answer, try to find other recipies you think add variation to the other recipes.
# If you still don't know the answer, don't try to make up an answer.
#
# {context}
#
# Question: {question}
# Your response:"""


@cl.action_callback("action_button")
async def on_action(action):
    ingredients = cl.user_session.get("current_ingredients")

    await cl.Message(content=f"Adding {ingredients} to your shopping list").send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()


@cl.action_callback("recommend_button")
async def on_action(action):
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    query = "Foreslå en ukesmeny med varierte oppskrifter"  # "Suggest a weekly dinner menu with diverse recipes."
    res = await llm_chain.acall(query, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await process_message(res)
    await action.remove()


@cl.on_chat_start
async def main():
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(LanceDB)(connection=table, embedding=embeddings)
    # docsearch = LanceDB(connection=table, embedding=embeddings)

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}

    llm = ChatOpenAI(model=OPENAI_MODEL)
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 7}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs=chain_type_kwargs,
    )

    # Store the chain in the user session
    cl.user_session.set("llm_chain", qa)
    actions = [
        cl.Action(
            name="recommend_button",
            value="example_value",
            label="Recommend",
            description="Click me!",
        )
    ]

    await cl.Message(
        content="Get started with some simple recommendataions", actions=actions
    ).send()


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here
    await process_message(res)


async def process_message(res):
    # logger.info(res)
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]
    actions = []
    if source_documents:
        logger.info(f"Found {len(source_documents)} documents: {source_documents}")
        ingredients = set(
            ingredient
            for doc in source_documents
            for ingredient in doc.metadata["Ingredients"].split(", ")
        )
        cl.user_session.set("current_ingredients", ingredients)
        actions.append(
            cl.Action(
                name="action_button",
                value="example_value",
                description="Click me!",
                label="Add to shopping list",
            )
        )
    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=answer, actions=actions).send()


# HOW TO RUN: chainlit run app.py -w

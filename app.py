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

OPENAI_MODEL = "gpt-4"  # "gpt-3.5-turbo"

logger = logging.getLogger(__name__)

recipes = pd.read_pickle("data/preprocessed/recipes.pkl")
recipes.drop_duplicates(subset=["id"], inplace=True)  # Must for this dataset
offers = pd.read_csv("data/products_on_offer.csv")
uri = "dataset/chainlit-recipes-lancedb"
db = lancedb.connect(uri)
table = db.create_table("recipes", recipes, mode="overwrite")


# template = f"""Du er et anbefalingssystem av oppskrifter som anbefaler oppskriver gitt brukerens preferanser basert på tilbudene på produkter denne uken.
# For hver dag i uken, gi oppskriften i formatet: 'Navn på dag: Oppskriftsnavn - Kort beskrivelse - Grunn'.
# Bruk konteksten under for å svare på spørsmålet.
# Hvis du fortsatt ikke vet svaret, spør brukeren om de har noen preferanser.
# Hvis du fortsatt ikke vet svaret, ikke prøv å finne på et svar.
#
# {{context}}
#
# Produkter på tilbud: {offers["title"].tolist()}
#
# Spørsmål: {{question}}
# Ditt svar:"""
template = """You are a recipe recommender system that help users to find weekly menus that match their preferences. 
Use the following pieces of context to answer the question at the end. 
For each question, suggest 7 recipies for a weekly menu, with a short description of the dish and the reason why the user might like it. Use the names of the week days to refer to the days of the week.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Your response:"""


@cl.action_callback("action_button")
async def on_action(action):
    ingredients = cl.user_session.get("current_ingredients")

    await cl.Message(content=f"Adding {ingredients} to your shopping list").send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()


@cl.action_callback("offers_recommend_button")
async def on_action(action):
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    all_offers = offers["title"].tolist()
    query = ", ".join(all_offers[: round(len(all_offers) * 0.8)])
    res = await llm_chain.acall(query, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await process_message(res)
    await action.remove()


@cl.action_callback("recommend_button")
async def on_action(action):
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    query = "Weekly menu of 7 recipes"
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
        output_key="result",
        chat_memory=message_history,
        return_messages=True,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 10}),
        memory=memory,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )

    # Store the chain in the user session
    cl.user_session.set("llm_chain", qa)
    actions = [
        cl.Action(
            name="recommend_button",
            value="example_value",
            label="Recommend",
            description="Click me!",
        ),
        cl.Action(
            name="offers_recommend_button",
            value="example_value",
            label="Recommend with offers",
            description="Click me!",
        ),
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
    answer = res["result"]
    source_documents = res["source_documents"]  # type: List[Document]
    actions = []
    if source_documents:
        logger.info(f"Found {len(source_documents)} documents: {source_documents}")
        ingredients = set(
            ingredient
            for doc in source_documents
            for ingredient in doc.metadata["ingredients"].split(", ")
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

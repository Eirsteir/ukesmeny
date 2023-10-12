import logging
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl


import lancedb
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.vectorstores import LanceDB, Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import pandas as pd
from langchain.llms import OpenAI

OPENAI_MODEL = "gpt-3.5-turbo"

logger = logging.getLogger(__name__)

import pandas as pd

merged = pd.read_pickle("data/merged_recipies_and_weekly_menus.pkl")
documents = merged["text"].tolist()
ids = [str(x) for x in merged.index.tolist()]
embeddings = merged["vector"].tolist()
metadatas = [
    {"source": md["ID"], **md}
    for md in merged[["ID", "Title", "Description", "Ingredients"]].to_dict(
        orient="records"
    )
]


# template = """Du er et anbefalingssystem av oppskrifter. Gitt brukerens preferanser, anbefal en ukentlig middagsmeny bestående av 7 oppskrifter.
# For hver dag i uken, gi oppskriften i formatet: 'Navn på dag: Oppskriftsnavn - Kort beskrivelse - Grunn'.
# Hvis du ikke vet svaret, prøv å finne andre oppskrifter du tror gir variasjon til de andre oppskriftene.
# Hvis du fortsatt ikke vet svaret, spør brukeren om de har noen preferanser.
# Hvis du fortsatt ikke vet svaret, ikke prøv å finne på et svar.
#
# {context}
#
# Spørsmål: {question}
# Ditt svar:"""
template = """You are a weekly menu and recipe recommender system that help users to find weekly menus or recipes that match their preferences.
For each day in the week, provide the recipe in the format: 'Name of Day: Recipe Name - Short Description - Reason'.
Use the following pieces of context to answer the question at the end. 
If you still don't know the answer, don't try to make up an answer.

Answer in the users language. 

{context}

Question: {question}
Your response:"""


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
    res = await llm_chain.acall(
        query,
        # callbacks=[cl.AsyncLangchainCallbackHandler()],
    )  # TODO: Does not work with callback
    await process_message(res)
    await action.remove()


@cl.on_chat_start
async def main():
    # embeddings = OpenAIEmbeddings()
    # docsearch = await cl.make_async(LanceDB)(connection=table, embedding=embeddings)
    # docsearch = LanceDB(connection=table, embedding=embeddings)

    import chromadb

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="recipes")

    if collection.count() == 0:
        collection.add(
            embeddings=embeddings, documents=documents, ids=ids, metadatas=metadatas
        )

    embedding_function = OpenAIEmbeddings()
    chroma_docsearch = await cl.make_async(Chroma)(
        client=chroma_client,
        collection_name="recipes",
        embedding_function=embedding_function,
    )

    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo

    metadata_field_info = [
        AttributeInfo(
            name="Title",
            description="The title of the recipe",
            type="string",
        ),
        AttributeInfo(
            name="Description",
            description="Description of the recipe",
            type="string",
        ),
        AttributeInfo(
            name="Ingredients",
            description="Ingredients required to make the recipe",
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="Identifier of the recipe",
            type="string",
        ),
    ]
    document_content_description = (
        "Recipies with a brief description and ingredients in Norwegian."
    )

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

    retriever = SelfQueryRetriever.from_llm(
        llm,
        chroma_docsearch,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        verbose=True,
        search_type="mmr",
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs=chain_type_kwargs,
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
        )
    ]

    await cl.Message(
        content="Get started with some simple recommendations", actions=actions
    ).send()


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    # res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    res = await cl.make_async(llm_chain)({"question": message})  # TODO: this works ish

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

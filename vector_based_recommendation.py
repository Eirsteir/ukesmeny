OPENAI_API_KEY = "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"

import pandas as pd

anime = pd.read_csv("data/anime_with_synopsis.csv")
anime.head()

anime["combined_info"] = anime.apply(
    lambda row: f"Title: {row['Name']}. Overview: {row['sypnopsis']} Genres: {row['Genres']}",
    axis=1,
)
anime.head(2)

import tiktoken

embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

encoding = tiktoken.get_encoding(embedding_encoding)

# omit descriptions that are too long to embed
anime["n_tokens"] = anime.combined_info.apply(lambda x: len(encoding.encode(x)))
anime = anime[anime.n_tokens <= max_tokens]

from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(
    open_ai_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)

from openai.embeddings_utils import get_embedding

anime["embedding"] = anime.combined_info.apply(
    lambda x: get_embedding(x, engine=embeddings_model)
)
anime.head()

anime.rename(columns={"embedding": "vector"}, inplace=True)
anime.rename(columns={"combined_info": "text"}, inplace=True)
anime.to_pickle("data/anime.pkl")

# import lancedb
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import LanceDB
# from langchain.chains import RetrievalQA
#
# uri = "dataset/sample-anime-lancedb"
# db = lancedb.connect(uri)
# table = db.create_table("anime", anime)
#
# # embeddings = OpenAIEmbeddings(engine="text-embedding-ada-002")
#
# docsearch = LanceDB(connection=table, embedding=embeddings_model)
# query = "I'm looking for an animated action movie. What could you suggest to me?"
# docs = docsearch.similarity_search(query, k=1)
# docs
#
# # Import Azure OpenAI
# from langchain.llms import AzureOpenAI
#
# qa = RetrievalQA.from_chain_type(
#     llm=AzureOpenAI(
#         deployment_name="text-davinci-003",
#         model_name="text-davinci-003",
#     ),
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(),
#     return_source_documents=True,
# )
#
# query = "I'm looking for an action anime. What could you suggest to me?"
# result = qa({"query": query})
# result["result"]
#
# result["source_documents"][0]
#
# df_filtered = anime[anime["Genres"].apply(lambda x: "Action" in x)]
# qa = RetrievalQA.from_chain_type(
#     llm=AzureOpenAI(
#         deployment_name="text-davinci-003",
#         model_name="text-davinci-003",
#         openai_api_key=openai_api_key,
#         openai_api_version=openai_api_version,
#     ),
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(search_kwargs={"data": df_filtered}),
#     return_source_documents=True,
# )
#
# query = "I'm looking for an anime with animals and an adventurous plot."
# result = qa({"query": query})
#
# qa = RetrievalQA.from_chain_type(llm=AzureOpenAI(deployment_name="text-davinci-003",
#                                                  model_name="text-davinci-003", openai_api_key=openai_api_key,
#                                                  openai_api_version=openai_api_version), chain_type="stuff",
#                                  retriever=docsearch.as_retriever(search_kwargs={'filter': {'score__gt': 7}),
#                                  return_source_documents=True)

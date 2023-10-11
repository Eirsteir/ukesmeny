# docker run -p 7474:7474 -p 7687:7687 -v $PWD/neo4j/data:/data -v $PWD/neo4j/plugins:/plugins -v=PWD/neo4j/import:/import  --name neo4j-apoc -e NEO4J_AUTH=neo4j/pleaseletmein  -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4J_PLUGINS='[\"apoc\"]' neo4j:latest


from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph


def get_connection():
    return Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="pleaseletmein"
    )


def populate_database(graph: Neo4jGraph):
    query = """
        LOAD CSV WITH HEADERS FROM 'file:///all_recipe_details.csv' AS row
        MERGE (re:Recipie {recipieId: row.ID, description: row.Description})
        WITH re, row
        UNWIND split(row.Ingredients, ',') AS ingredient
        MERGE (i:Ingredient {title: ingredient})
        MERGE (re)-[r:HAS_INGREDIENT]->(i)
    """

    graph.query(query)


if __name__ == "__main__":
    graph = get_connection()
    # populate_database(graph)

    graph.refresh_schema()
    print(graph.schema)

    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            openai_api_key="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",
        ),
        graph=graph,
        verbose=True,
    )

    chain.run("Give me a recipie with pasta as main ingredient")

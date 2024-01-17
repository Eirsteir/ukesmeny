# docker run -p 7474:7474 -p 7687:7687 -v $PWD/neo4j/data:/data -v $PWD/neo4j/plugins:/plugins -v $PWD/neo4j/import:/import  --name neo4j-apoc -e NEO4J_AUTH=neo4j/pleaseletmein  -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4J_PLUGINS='[\"apoc\", \"graph-data-science\"]' neo4j:latest


from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph


def get_connection():
    return Neo4jGraph(
        url="bolt://localhost:7687", username="neo4j", password="pleaseletmein"
    )


def populate_database(graph: Neo4jGraph):
    query = """
        // Step 1: Load the CSV and update or create the Recipe nodes
        LOAD CSV WITH HEADERS FROM 'file:///recipes.csv' AS row
        MERGE (re:Recipe {recipieId: row.id})
        SET re.openaiEmbeddings = apoc.convert.fromJsonList(row.vector),
            re.title = row.title,
            re.description = row.description,
            re.keywords = apoc.convert.fromJsonList(row.keywords);
        
        // Step 2: Create the relationships between Recipe and Ingredient
        LOAD CSV WITH HEADERS FROM 'file:///recipes.csv' AS row
        MATCH (re:Recipe {recipieId: row.id})
        UNWIND split(row.ingredients, ', ') AS ingredient
        MERGE (i:Ingredient {title: ingredient})
        MERGE (re)-[r:HAS_INGREDIENT]->(i);
        
        // Step 3: Create the relationships between Recipe and Menu
        LOAD CSV WITH HEADERS FROM 'file:///recipes.csv' AS row
        MATCH (re:Recipe {recipieId: row.id})
        WHERE row.year IS NOT NULL OR row.year <> "null"
        MERGE (m:Menu {year: toInteger(row.year), week: toInteger(row.week)})
        MERGE (re)-[:IS_PART_OF_MENU {weekDay: row.week_day}]->(m);
                
        // Step 4: Create the relationships between Recipe and Tags
        LOAD CSV WITH HEADERS FROM 'file:///recipes.csv' AS row
        MATCH (re:Recipe {recipieId: row.id})
        WITH re, apoc.convert.fromJsonList(row.tags) as tags
        UNWIND tags as tag
        MERGE (t:Tag {title: tag})
        MERGE (re)-[r:HAS_TAG]->(t);
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
            openai_api_key="",
        ),
        graph=graph,
        verbose=True,
    )

    chain.run("Give me a recipie with pasta as main ingredient")

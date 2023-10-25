import os
import os.path as osp

import numpy as np
import torch
from graphdatascience import GraphDataScience
from sentence_transformers import SentenceTransformer
from torch_geometric.data import InMemoryDataset

# Get Neo4j DB URI, credentials and name from environment if applicable
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")
NEO4J_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASSWORD", "pleaseletmein"),
)

gds = GraphDataScience(NEO4J_URI, auth=NEO4J_AUTH, database=NEO4J_DB)


def fetch_data(query):
    return gds.run_cypher(query)


def load_node(cypher, index_col, encoders=None, **kwargs):
    # Execute the cypher query and retrieve data from Neo4j
    df = fetch_data(cypher)
    df.set_index(index_col, inplace=True)
    # Define node mapping
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    # Define node features
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge(
    cypher,
    src_index_col,
    src_mapping,
    dst_index_col,
    dst_mapping,
    encoders=None,
    **kwargs
):
    # Execute the cypher query and retrieve data from Neo4j
    df = fetch_data(cypher)
    # Define edge index
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    # Define edge features
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class SequenceEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(
            df.values,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
        )
        return x.cpu()


def encode_cyclical_week(week_series):
    week_sin = np.sin(2 * np.pi * week_series / 52.0)
    week_cos = np.cos(2 * np.pi * week_series / 52.0)
    week_tensor = torch.Tensor(list(zip(week_sin, week_cos)))
    return week_tensor


def encode_cyclical_weekday(weekday_series):
    # Map weekdays to integers: Monday: 0, Tuesday: 1, ..., Sunday: 6
    weekday_to_int = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    weekday_series_int = weekday_series.map(weekday_to_int)

    # Apply cyclical encoding
    weekday_sin = np.sin(2 * np.pi * weekday_series_int / 7.0)
    weekday_cos = np.cos(2 * np.pi * weekday_series_int / 7.0)
    weekday_tensor = torch.Tensor(list(zip(weekday_sin, weekday_cos)))

    return weekday_tensor


def download_from_neo4j(raw_dir):
    recipe_query = """
    MATCH (a:Recipe)
    RETURN a.recipieId as recipeId,
           a.openaiEmbeddings as embedding
    """

    recipe_x, recipe_mapping = load_node(
        recipe_query,
        index_col="recipeId",
        encoders={"embedding": lambda x: torch.Tensor(x)},
    )

    ingredient_query = """
    MATCH (i:Ingredient)
    RETURN DISTINCT ID(i) as ingredientId, i.title as title
    """
    ingredient_x, ingredient_mapping = load_node(
        ingredient_query,
        index_col="ingredientId",
        encoders={"title": SequenceEncoder()},
    )

    menu_query = """
    MATCH (m:Menu)
    RETURN ID(m) as menuId, m.year as year, m.week as week
    """
    menu_x, menu_mapping = load_node(
        menu_query,
        index_col="menuId",
        encoders={
            "year": lambda x: torch.Tensor(x.tolist()).view(-1, 1),
            "week": encode_cyclical_week,
        },
    )

    recipe_menu_query = """
    MATCH (n:Recipe)-[r:IS_PART_OF_MENU]->(m:Menu)
    RETURN n.recipieId AS recipeId, ID(m) AS menuId, r.weekDay AS weekDay
    """
    recipe_menu_edge_index, recipe_menu_edge_label = load_edge(
        recipe_menu_query,
        src_index_col="recipeId",
        src_mapping=recipe_mapping,
        dst_index_col="menuId",
        dst_mapping=menu_mapping,
        encoders={"weekDay": encode_cyclical_weekday},
    )

    recipe_ingredient_query = """
    MATCH (n:Recipe)-[r:HAS_INGREDIENT]->(i:Ingredient) 
    RETURN n.recipieId AS recipeId, ID(i) AS ingredientId
    """

    recipe_ingredient_edge_index, recipe_ingredient_edge_label = load_edge(
        recipe_ingredient_query,
        src_index_col="recipeId",
        src_mapping=recipe_mapping,
        dst_index_col="ingredientId",
        dst_mapping=ingredient_mapping,
    )

    torch.save(
        (
            (recipe_x, recipe_mapping),
            (ingredient_x, ingredient_mapping),
            (menu_x, menu_mapping),
            (recipe_menu_edge_index, recipe_menu_edge_label),
            (recipe_ingredient_edge_index, recipe_ingredient_edge_label),
        ),
        osp.join(raw_dir, "raw_data.pt"),
    )


class Recipe(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["raw_data.pt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        download_from_neo4j(self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        from torch_geometric.data import HeteroData

        (
            (recipe_x, recipe_mapping),
            (ingredient_x, ingredient_mapping),
            (menu_x, menu_mapping),
            (recipe_menu_edge_index, recipe_menu_edge_label),
            (recipe_ingredient_edge_index, recipe_ingredient_edge_label),
        ) = torch.load(osp.join(self.raw_dir, self.raw_file_names[0]))

        data = HeteroData()

        data["recipe"].x = recipe_x
        data["menu"].x = menu_x
        data["ingredient"].x = ingredient_x
        data[
            "recipe", "has_ingredient", "ingredient"
        ].edge_index = recipe_ingredient_edge_index
        data["recipe", "has_ingredient", "ingredient"].edge_attr = (
            torch.ones((recipe_ingredient_edge_index.size(1), 1)) * 1
        )
        data["recipe", "is_part_of_menu", "menu"].edge_index = recipe_menu_edge_index
        data["recipe", "is_part_of_menu", "menu"].edge_attr = recipe_menu_edge_label
        data.num_relations = 2
        data.num_nodes = (
            len(recipe_mapping) + len(ingredient_mapping) + len(menu_mapping)
        )

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    data = Recipe(root="pyg_data")
    print(data[0])

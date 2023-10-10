import pandas as pd
from setup import Recipe, Ingredient, get_session

if __name__ == "__main__":
    recipes_df = pd.read_csv("data/all_recipe_details.csv")
    recipes_df["Ingredients"] = recipes_df["Ingredients"].apply(lambda x: x.split(", "))
    ingredients = recipes_df["Ingredients"].explode().unique()

    with get_session() as session:
        ingredients = [
            Ingredient(title=ingredient) for ingredient in ingredients if ingredient
        ]
        session.add_all(ingredients)
        print(f"Added {len(ingredients)} ingredients to the database.")

        recipes = [
            Recipe(
                name=recipe["Title"],
                description=recipe["Description"],
                source_id=recipe["ID"],
                ingredients=session.query(Ingredient)
                .filter(Ingredient.title.in_(recipe["Ingredients"]))
                .all(),
            )
            for _, recipe in recipes_df.iterrows()
        ]
        session.add_all(recipes)
        print(f"Added {len(recipes)} recipes to the database.")

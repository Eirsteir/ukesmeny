import datetime
import pandas as pd


# 1. Gather Data
def extract_data_from_graph():
    # Pseudocode: Extract data from your graph database
    # Return recipes, their ingredients, and weekly offers
    recipes = pd.read_csv("data/all_recipe_details.csv")
    ingredients = recipes["Ingredients"].unique()
    weekly_offers = pd.read_csv("data/products_on_offer.csv")
    return recipes, ingredients, weekly_offers


def get_weekly_menu_recommendations():
    # Pseudocode: Fetch the store's weekly menu recommendations
    return pd.read_csv("data/meny_weekly_menus.csv")


# 2. Generate Meta-path Inputs
# def generate_prompts(recipes, ingredients, weekly_offers, date):
#     prompts = []
#     for recipe in recipes:
#         for ingredient in recipe["ingredients"]:
#             if ingredient in weekly_offers:
#                 prompt = f"Given that {ingredient} is on offer this week at {weekly_offers[ingredient]['store']}, recommend a recipe that uses this ingredient."
#                 prompts.append(prompt)
#     return prompts


# 3. Assign Relevance Scores
def assign_relevance_scores(recipe, weekly_menu_recommendations):
    if (
        recipe["ID"] in weekly_menu_recommendations["recipe_id"].values
    ):  # TODO: check offers
        return "High relevance"
    else:
        return "Low relevance"


# 4. Format Data Points
def format_data_points(recipes, weekly_offers, weekly_menu_recommendations):
    data_points = []
    for _, recipe in recipes.iterrows():
        instruction = "Evaluate the relevance of the following recipe based on this week's offers."
        input_ = f"Recipe: {recipe['Title']}. Offers: {', '.join([ingredient for ingredient in recipe['Ingredients'].split(', ') if ingredient in weekly_offers['title']])}."
        response = assign_relevance_scores(recipe, weekly_menu_recommendations)
        data_points.append(
            {"instruction": instruction, "input": input_, "response": response}
        )
    return data_points


# # 5. Incorporate Negative Examples
# def add_negative_examples(
#     data_points, past_weekly_offers, past_weekly_menu_recommendations
# ):
#     # Pseudocode: Add recipes that were relevant in past weeks but not in the current week
#     # This can be done by comparing the current week's offers and recommendations with past data
#     data_points_with_negatives = []  # TODO
#     return data_points_with_negatives


def main():
    recipes, ingredients, weekly_offers = extract_data_from_graph()

    weekly_menu_recommendations = get_weekly_menu_recommendations()

    data_points = format_data_points(
        recipes, weekly_offers, weekly_menu_recommendations
    )

    final_df = pd.DataFrame(data_points)
    final_df.to_csv("data/data_points.csv", index=False)


if __name__ == "__main__":
    main()

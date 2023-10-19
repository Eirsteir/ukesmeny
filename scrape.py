import os
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

DINNERS_URL = "https://meny.no/oppskrifter/middagstips/"
RECIPIE_DETAIL_URL = (
    lambda id: f"https://platform-rest-prod.ngdata.no/api/recipes2/1300/{id}?full_response=true&fieldset=maximal"
)
BASE_URL = "https://meny.no/"
WEEKLY_MENU_URL = "https://meny.no/api/weeklyrecipeservice/"
CURRENT_OFFERS_URL = "https://platform-rest-prod.ngdata.no/api/products/1300/7080001271923?page=1&page_size=140&full_response=true&fieldset=maximal&facets=Category%2CAllergen&facet=IsOffer%3Atrue%3BCategories%3AMiddag&showNotForSale=true"

CSV_IDS_FILE = "data/scraping/meny/meny_all_recipe_ids.csv"
CSV_OUT = "data/scraping/meny/meny_all_recipe_details.csv"
CSV_WEEKLY_MENUS = "data/scraping/meny/meny_weekly_menus.csv"


@dataclass
class Recipe:
    id: str
    title: str
    description: str
    ingredients: str
    year: int = None
    week: int = None
    week_day: str = None
    keywords: list[str] = None
    tags: list[str] = None


def extract_recipe_ids(page_content):
    soup = BeautifulSoup(page_content, "html.parser")
    tags_with_recipe_id = soup.find_all(attrs={"data-prop-recipe-id": True})
    return [tag["data-prop-recipe-id"] for tag in tags_with_recipe_id]


def fetch_recipe_details(recipe_id):
    response = requests.get(RECIPIE_DETAIL_URL(recipe_id))
    response.raise_for_status()
    data = response.json()
    data = data.get("_source", {})
    title = data.get("name", "")
    description = data.get("description", "")
    recipie_details = data.get("recipeDetails", [])
    ingredients = [
        ingredient["name"]
        for detail in recipie_details
        for ingredient in detail.get("ingredients", [])
    ]
    keywords = data.get("keywords", "").split(", ")
    tags = data.get("tags", [])

    return title, description, ingredients, keywords, tags


def scrape_recipie_ids() -> list[str]:
    print("Scraping recipe IDs...")

    if os.path.exists(CSV_IDS_FILE):
        df = pd.read_csv(CSV_IDS_FILE)
        all_recipe_ids = df["Recipe_ID"].tolist()
        print(
            f"Using {len(all_recipe_ids)} existing recipe IDs from all_recipe_ids.csv"
        )
    else:
        all_recipe_ids = []
        page_number = 1

        with tqdm(desc="Scraping", unit="page") as pbar:
            while True:
                # Construct the URL with the page number
                url = f"{DINNERS_URL}?pagenr={page_number}"

                # Fetch the page content
                response = requests.get(url)
                response.raise_for_status()

                # Extract recipe IDs from the current page
                recipe_ids = extract_recipe_ids(response.content)

                # If no recipe IDs are found, break out of the loop
                if not recipe_ids:
                    break

                # Add the extracted IDs to the master list
                all_recipe_ids.extend(recipe_ids)

                # Sleep for a set duration (e.g., 2 seconds) before the next request
                if len(all_recipe_ids) % 100 == 0:
                    time.sleep(2)

                # Increment the page number for the next iteration
                page_number += 1

                pbar.update(1)

        # Create a DataFrame and save to .csv
        df = pd.DataFrame(all_recipe_ids, columns=["Recipe_ID"])
        df.to_csv(CSV_IDS_FILE, index=False)
        print(f"Recipe IDs have been saved to {CSV_IDS_FILE}!")

    return all_recipe_ids


def scrape_recipies(recipe_ids: list[str]):
    # Fetch details for each recipe ID
    print("Scraping recipe details...")
    recipes = []
    for recipe_id in tqdm(recipe_ids, desc="Fetching Details", unit="recipe"):
        title, description, ingredients, keywords, tags = fetch_recipe_details(
            str(recipe_id)
        )

        recipes.append(
            Recipe(
                id=recipe_id,
                title=title,
                description=description,
                ingredients=", ".join(ingredients),
                keywords=keywords,
                tags=tags,
            )
        )

        if len(recipes) % 100 == 0:
            time.sleep(2)

    # Create a DataFrame and save to .csv
    df_details = pd.DataFrame(recipes)
    df_details.to_csv(CSV_OUT, index=False)
    print(f"\nRecipe details have been saved to {CSV_OUT}.csv!")
    return recipes


def scrape_weekly_menus():
    print("Scraping weekly menus...")
    weekly_menus = []
    weekly_recipies_url = "https://meny.no/api/weeklyrecipeservice/?page=1&pageSize=200"
    response = requests.get(weekly_recipies_url)
    response.raise_for_status()
    data = response.json()
    available_weeks = [
        (recipe["Year"], recipe["WeekNumber"]) for recipe in data["data"]
    ]

    print(f"Found {len(available_weeks)} weeks with recipies")

    # weekly_menus = []
    for year, week in tqdm(available_weeks):
        response = requests.get(WEEKLY_MENU_URL + f"{year}/{week}")
        response.raise_for_status()
        data = response.json()[0]

        for recipe in data["RecipeList"][:7]:  # skip ukens bakst
            if recipe.get("RecipeId") is None:
                continue

            title, description, ingredients, keywords, tags = fetch_recipe_details(
                str(recipe["RecipeId"])
            )
            weekly_menus.append(
                Recipe(
                    id=recipe["RecipeId"],
                    title=title,
                    description=description,
                    ingredients=", ".join(ingredients),
                    keywords=keywords,
                    tags=tags,
                    year=year,
                    week=week,
                    week_day=recipe["WeekDay"],
                )
            )

    df_details = pd.DataFrame(weekly_menus)
    df_details.to_csv(CSV_WEEKLY_MENUS, index=False)
    print(f"\nRecipe details have been saved to {CSV_WEEKLY_MENUS}.csv!")
    return weekly_menus


def fetch_offers():
    # Fetch only current offers for now
    response = requests.get(CURRENT_OFFERS_URL)
    response.raise_for_status()
    data = response.json()["hits"]["hits"]

    print(f"Found {len(data)} products on offer")

    products_on_offer = [
        {
            "title": product["_source"]["title"],
            "subtitle": product["_source"]["subtitle"],
        }
        for product in data
    ]
    # Save to products_on_offer.csv
    df = pd.DataFrame(products_on_offer)
    df.to_csv("data/products_on_offer.csv", index=False)


if __name__ == "__main__":
    recipe_ids = scrape_recipie_ids()
    recipes = scrape_recipies(recipe_ids)
    menus = scrape_weekly_menus()

    # fetch_offers()

    df = pd.DataFrame([*recipes, *menus])
    df.to_csv("data/data.csv")

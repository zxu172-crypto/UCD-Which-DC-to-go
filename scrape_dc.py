#!/usr/bin/env python3
"""
UC Davis dining menu scraper (3 DataFrames output).

- Robust ingredients extraction:
  - Handles cases where "Ingredients:" and the actual list are split across tags/text nodes.
  - Also checks common HTML attributes (data-content/title/aria-label) used in tooltips/accordions.

- Treats Latitude the same as a Dining Commons.

CLI examples:
  python ucd_menus.py --scrape --locations segundo tercero latitude
  python ucd_menus.py --no-scrape --locations segundo tercero latitude

Notebook usage:
  from ucd_menus import get_three_menus
  df1, df2, df3 = get_three_menus(scrape=True, location_keys=["segundo","tercero","latitude"])
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


# ----------------------------
# Config
# ----------------------------

HEADING_TAGS: Tuple[str, ...] = ("h1", "h2", "h3", "h4")

WEEKDAYS = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}

MEAL_ALIASES = {
    "breakfast": "Breakfast",
    "brunch": "Brunch",
    "lunch": "Lunch",
    "dinner": "Dinner",
    "late night": "Late Night",
}

# When extracting ingredients, stop capture if we hit nutrition-like labels.
# (This list is intentionally conservative; it’s OK if it doesn’t match everything.)
ING_STOP_LABELS = (
    "Contains:",
    "Allergens:",
    "Serving Size:",
    "Calories:",
    "Fat (g):",
    "Carbohydrates (g):",
    "Protein (g):",
    "Sodium (mg):",
    "Nutrition",
)

ATTR_CANDIDATES = ("data-content", "data-bs-content", "title", "aria-label", "data-original-title")


@dataclass(frozen=True)
class LocationSpec:
    key: str
    name: str
    url: str


# Treat Latitude as a Dining Commons (same pipeline/logic)
LOCATIONS: Dict[str, LocationSpec] = {
    "segundo": LocationSpec(
        key="segundo",
        name="Segundo Dining Commons",
        url="https://housing.ucdavis.edu/dining/dining-commons/segundo/",
    ),
    "tercero": LocationSpec(
        key="tercero",
        name="Tercero Dining Commons",
        url="https://housing.ucdavis.edu/dining/dining-commons/tercero/",
    ),
    "cuarto": LocationSpec(
        key="cuarto",
        name="Cuarto Dining Commons",
        url="https://housing.ucdavis.edu/dining/dining-commons/cuarto/",
    ),
    "latitude": LocationSpec(
        key="latitude",
        name="Latitude (treated as Dining Commons)",
        url="https://housing.ucdavis.edu/dining/latitude/",
    ),
}


# ----------------------------
# Utilities
# ----------------------------

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _safe_cache_path(cache_dir: str, key: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.csv")


def _fetch_html(url: str, *, timeout_s: int = 30, retries: int = 2, sleep_s: float = 0.8) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; UCDMenuScraper/2.0; +https://housing.ucdavis.edu/)"
    }
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_s)
            else:
                raise
    assert last_err is not None
    raise last_err


def _find_weekly_menu_start(soup: BeautifulSoup) -> Optional[Tag]:
    # Find the first heading whose text is exactly "Weekly Menu" (case-insensitive)
    for h in soup.find_all(list(HEADING_TAGS)):
        if _normalize_ws(h.get_text(" ", strip=True)).lower() == "weekly menu":
            return h
    return None


def _collect_block_text_between(start_h4: Tag, stop_heading: Tag) -> str:
    """
    Collect text for the dish block between <h4> dish and the next heading.
    - Joins all NavigableStrings (prevents missing split 'Ingredients:' + list).
    - Also scans common attributes that might contain hidden text.
    """
    parts: List[str] = []

    for el in start_h4.next_elements:
        if el is stop_heading:
            break

        # Collect visible-ish text nodes
        if isinstance(el, NavigableString):
            t = _normalize_ws(str(el))
            if t:
                parts.append(t)
            continue

        if isinstance(el, Tag):
            if el.name in ("script", "style", "noscript"):
                continue

            # Collect attribute payloads if they look relevant
            for attr in ATTR_CANDIDATES:
                if attr in el.attrs:
                    val = el.attrs.get(attr)
                    if isinstance(val, str) and ("ingredient" in val.lower()):
                        parts.append(_normalize_ws(val))
                    elif isinstance(val, list):
                        joined = " ".join([v for v in val if isinstance(v, str)])
                        if "ingredient" in joined.lower():
                            parts.append(_normalize_ws(joined))

    return _normalize_ws(" ".join(parts))


def _extract_ingredients_from_dish_h4(dish_h4: Tag) -> Optional[str]:
    stop_heading = dish_h4.find_next(HEADING_TAGS)
    if stop_heading is None:
        return None

    block = _collect_block_text_between(dish_h4, stop_heading)
    if not block:
        return None

    # Regex: grab text after Ingredients: (or Ingredient:) until a stop label or end.
    stop_alt = "|".join([re.escape(x) for x in ING_STOP_LABELS])
    pattern = re.compile(
        r"\bIngredient(?:s)?\s*:\s*(.+?)(?=\b(?:" + stop_alt + r")\b|$)",
        flags=re.IGNORECASE,
    )

    matches = [m.group(1).strip() for m in pattern.finditer(block)]
    matches = [_normalize_ws(m) for m in matches if _normalize_ws(m)]

    if not matches:
        return None

    # Keep the longest match (usually the full ingredient list)
    return max(matches, key=len)


# ----------------------------
# Core scraping
# ----------------------------

def scrape_location_weekly_menu(url: str, location_name: str) -> pd.DataFrame:
    """
    Parse a single SHDS location page.

    Returns a DF with context columns:
      location, day, meal, zone, dish, ingredients
    """
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    for t in soup.find_all(["script", "style", "noscript"]):
        t.decompose()

    weekly_start = _find_weekly_menu_start(soup)
    if weekly_start is None:
        # If the page format changes, fail loudly.
        raise ValueError(f"Could not find 'Weekly Menu' section on: {url}")

    in_weekly = False
    current_day: Optional[str] = None
    current_meal: Optional[str] = None
    current_zone: Optional[str] = None

    rows: List[Dict[str, Optional[str]]] = []

    # Walk headings in document order, but only after we see Weekly Menu.
    for h in soup.find_all(list(HEADING_TAGS)):
        text = _normalize_ws(h.get_text(" ", strip=True))
        if not text:
            continue

        if not in_weekly:
            if h is weekly_start:
                in_weekly = True
            continue

        # Day
        if h.name == "h1" and text in WEEKDAYS:
            current_day = text
            current_meal = None
            current_zone = None
            continue

        # Meal
        if h.name == "h2":
            key = text.lower()
            if key in MEAL_ALIASES:
                current_meal = MEAL_ALIASES[key]
                current_zone = None
            continue

        # Zone
        if h.name == "h3" and text.lower().endswith("zone"):
            current_zone = text.replace(" Zone", "")
            continue

        # Dish
        if h.name == "h4":
            dish = text
            if dish.lower().startswith("currently there are no dishes"):
                continue

            ingredients = _extract_ingredients_from_dish_h4(h)

            rows.append(
                {
                    "location": location_name,
                    "day": current_day,
                    "meal": current_meal,
                    "zone": current_zone,
                    "dish": dish,
                    "ingredients": ingredients,
                }
            )

    return pd.DataFrame(rows, columns=["location", "day", "meal", "zone", "dish", "ingredients"])


def get_three_menus(
    *,
    scrape: bool,
    location_keys: Sequence[str],
    cache_dir: str = "cache",
    polite_delay_s: float = 1.0,
    include_context: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Input:
      scrape: True -> scrape live and cache; False -> load cached CSVs
      location_keys: exactly 3 keys, chosen from LOCATIONS
      include_context: False -> output only [meal, dish, ingredients]
                       True  -> output [location, day, meal, zone, dish, ingredients]

    Output:
      Exactly 3 DataFrames (in the same order as location_keys).
    """
    if len(location_keys) != 3:
        raise ValueError(f"Expected exactly 3 locations, got {len(location_keys)}: {location_keys}")

    dfs: List[pd.DataFrame] = []
    for key in location_keys:
        if key not in LOCATIONS:
            raise KeyError(f"Unknown location key '{key}'. Options: {sorted(LOCATIONS.keys())}")

        loc = LOCATIONS[key]
        cache_path = _safe_cache_path(cache_dir, key)

        if scrape:
            df = scrape_location_weekly_menu(loc.url, loc.name)
            df.to_csv(cache_path, index=False)
            time.sleep(polite_delay_s)
        else:
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
            else:
                df = pd.DataFrame(columns=["location", "day", "meal", "zone", "dish", "ingredients"])

        if include_context:
            out_df = df[["location", "day", "meal", "zone", "dish", "ingredients"]].copy()
        else:
            out_df = df[["meal", "dish", "ingredients"]].copy()

        dfs.append(out_df)

    return dfs[0], dfs[1], dfs[2]


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape UC Davis menus into exactly 3 DataFrames.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scrape", dest="scrape", action="store_true", help="Scrape live (default).")
    group.add_argument("--no-scrape", dest="scrape", action="store_false", help="Load cached CSVs instead.")
    parser.set_defaults(scrape=True)

    parser.add_argument(
        "--locations",
        nargs=3,
        default=["segundo", "tercero", "latitude"],  # Latitude treated like a DC
        choices=sorted(LOCATIONS.keys()),
        help="Pick exactly 3 locations (default: segundo tercero latitude).",
    )

    parser.add_argument("--cache-dir", default="cache", help="Cache directory for CSVs.")
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include context columns: location/day/zone in output DFs.",
    )
    args = parser.parse_args()

    df1, df2, df3 = get_three_menus(
        scrape=args.scrape,
        location_keys=args.locations,
        cache_dir=args.cache_dir,
        include_context=args.include_context,
    )

    # Minimal info for sanity-check
    print("Returned DataFrames:")
    print(f"1) {args.locations[0]}: shape={df1.shape}, missing_ingredients={int(df1['ingredients'].isna().sum())}")
    print(f"2) {args.locations[1]}: shape={df2.shape}, missing_ingredients={int(df2['ingredients'].isna().sum())}")
    print(f"3) {args.locations[2]}: shape={df3.shape}, missing_ingredients={int(df3['ingredients'].isna().sum())}")


if __name__ == "__main__":
    main()

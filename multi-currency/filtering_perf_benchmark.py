# Benchmark filtering native with a single price_usd filter vs. the dynamic OR across multiple currencies.
# If the dynamic OR is fast enough, using the global currency document and runtime conversions is preferable,
# as it removes the need to update price_usd in all non-USD items for every currency conversion rate change.

# - delete all Vespa documents.
# - feed the currency conversion documents.
# - generate and save 1mm documents with both native and usd prices.
# - feed those docs into vespa
# - generate and save 5,000 random queries with price filters in various currencies and random keywords.
# - run the queries with both native and usd price filters.
# - compare the results and timings.


# filtering_perf_benchmark.py
# python3 filtering_perf_benchmark.py --help   to see options
import numpy as np
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List
from generate_price_filter_query import load_conversion_rates

RATE_TABLE, _ = load_conversion_rates("currency.xml")

def pct(values: list[float], *percents: int) -> list[float]:
    """
    Returns the requested percentiles from the input list.

    pct([1, 2, 3, 4], 25, 50, 75) -> [1.75, 2.5, 3.25]
    """
    arr = np.asarray(values, dtype=np.float64)
    # numpy >= 1.22: use method instead of interpolation
    return np.percentile(arr, percents, method="linear").tolist()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(cmd: List[str], stdin: bytes | None = None) -> str:
    """
    Runs a Vespa CLI command and returns stdout.
    Raises `subprocess.CalledProcessError` on failure.
    """
    res = subprocess.run(cmd, input=stdin, check=True, text=True, stdout=subprocess.PIPE)
    return res.stdout


def _feed_jsonl(lines: List[str]) -> None:
    """
    Feeds JSONL documents using `vespa feed -`.
    """
    data = ("\n".join(lines) + "\n").encode()
    _run_cli(["vespa", "feed", "-"], stdin=data)

def feed_currency_documents(factors: Dict[str, float], ns: str) -> None:
    """
    Feeds `currency` documents with conversion factors.
    """
    jsonl = [
        json.dumps({"put": f"id:{ns}:currency::{code.lower()}",
                    "fields": {"factor": factor}})
        for code, factor in factors.items()
    ]
    _feed_jsonl(jsonl)

# Price buckets for random price generation.
BUCKETS: list[tuple[int,int]] = [(0,5), (5,10), (10,20), (20,40), (40,80), (80,150), (150,300), (300,600), (600,1000), (1000,10000)]

def random_price_cents() -> int:
    bucket_idx = random.randint(0, len(BUCKETS) - 1)
    return random.randint(BUCKETS[bucket_idx][0] * 100, BUCKETS[bucket_idx][1] * 100)


CURRENCY_PROBS = {
    "USD": 0.70304,
    "EUR": 0.11294,
    "GBP": 0.10312,
    "CAD": 0.03630,
    "AUD": 0.02066,
    "TRY": 0.00474,
    "INR": 0.00286,
    "PHP": 0.00197,
    "VND": 0.00174,
    "HKD": 0.00167,
    "SEK": 0.00125,
    "IDR": 0.00115,
    "NZD": 0.00113,
    "CHF": 0.00110,
    "ILS": 0.00109,
    "MAD": 0.00103,
    "SGD": 0.00103,
    "MYR": 0.00077,
    "ZAR": 0.00074,
    "MXN": 0.00062,
    "DKK": 0.00060,
    "NOK": 0.00044,
    "TWD": 0.00001,
    "PLN": 0.00001,
    "THB": 0.00001,
    "JPY": 0.00001,
    "CZK": 0.00001,
    "CNY": 0.00001,
}

# ----------------------------------------------------------------------
def random_currency() -> str:
    currencies = list(CURRENCY_PROBS.keys())
    weights    = list(CURRENCY_PROBS.values())
    return random.choices(currencies, weights=weights, k=1)[0]

tokens = [f"token{i}" for i in range(1, 1001)]  # Example tokens for item titles
def generate_items():
    """
    Generates a list of item documents with random prices and currency references.
    This is a placeholder function; replace with actual item generation logic.
    """
    items = []
    for i in range(1, 1_000_001):
        price_usd = random_price_cents()
        currency = random_currency()
        item = {
            "put": f"id:shopping:item::item-{i}",
            "fields": {
                "currency_ref": f"id:shopping:currency::{currency.lower()}",
                 #"item_name": ' '.join(random.sample(tokens, random.randint(1, 5))), # Randomly select 1-5 tokens
                "price_usd": price_usd,
                "price": RATE_TABLE[("USD", currency.upper())] * price_usd,  # Convert to the target currency
            }
        }
        items.append(json.dumps(item))
    return items


def vespa_feed(jsonl_file: Path) -> None:
    _run_cli(["vespa", "feed", str(jsonl_file)])


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def query_items(yql: str, hits: int = 100) -> dict:
    """
    Executes a Vespa query using the CLI and returns the response JSON.
    """
    stdout = _run_cli(["vespa", "query", f"{yql}", f"hits={hits}"])
    return json.loads(stdout)

def parse_vespa_results(result: dict) -> tuple[int, list[dict]]:
    root = result.get('root', {})
    total_hits = root.get('fields', {}).get('totalCount', 0)
    top_100_hits = root.get('children', [])[:100]      # keep full hit objects
    return total_hits, top_100_hits

if __name__ == "__main__":

    # from vespa.application import Vespa
    #
    # # delete all Vespa documents.
    # app = Vespa(url="localhost", port=8080)
    # response = app.delete_all_docs(content_cluster_name="shopping", schema='item')
    # print(response)
    #
    # # generate 1mm sample items with random prices, item_name, and currency.
    # items = generate_items()
    # # write to a file
    # items_file = Path("1mm_items.jsonl")
    # items_file.write_text("\n".join(items) + "\n")
    #
    # vespa_feed(items_file)

    from generate_price_filter_query import generate_price_filter_query

    # record the latency for both queries
    lat_multi:  list[float] = []
    lat_single: list[float] = []

    for i in range(1, 1000):
        prices = [random_price_cents() for _ in range(2)]
        price_usd_min = min(prices)
        price_usd_max = max(prices)
        currency = random_currency()
        rate = RATE_TABLE[("USD", currency.upper())]
        min_price = rate * price_usd_min
        max_price = rate * price_usd_max

        multi_currency_where=f"select * from item where {generate_price_filter_query(min_price, max_price, currency.lower())}"
        single_currency_where=f"select * from item where price_usd >= {price_usd_min} and price_usd <= {price_usd_max}"

        start = time.perf_counter()
        multi_currency_results = parse_vespa_results(query_items(multi_currency_where))
        lat_multi.append(time.perf_counter() - start)

        start = time.perf_counter()
        single_currency_results = parse_vespa_results(query_items(single_currency_where))
        lat_single.append(time.perf_counter() - start)

        if multi_currency_results[0] != single_currency_results[0]:
            print(f"Total hits mismatch: {multi_currency_results[0]} vs {single_currency_results[0]} currency:{currency}, min_price: {min_price}, max_price: {max_price} price_usd_min: {price_usd_min}, price_usd_max: {price_usd_max} rate: {rate}")
        else:
            print(f"Total hits: {multi_currency_results[0]} currency:{currency}, min_price: {min_price}, max_price: {max_price} price_usd_min: {price_usd_min}, price_usd_max: {price_usd_max} rate: {rate}")

    print(f"latency for multi-currency query: {pct(lat_multi, [25, 50, 75, 90, 95, 99])}")
    print(f"latency for price_usd query: {pct(lat_single, [25, 50, 75, 90, 95, 99])}")





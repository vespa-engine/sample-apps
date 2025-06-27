import sys
import argparse
import xml.etree.ElementTree as ET

def load_conversion_rates(xml_file: str) -> tuple[dict[tuple[str, str], float], set[str]]:
    """
    Parses the currency XML file and builds a conversion rate table.
    Returns a dictionary of rates and a sorted list of all currencies.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error: Could not read or parse the currency XML file '{xml_file}'.\n{e}", file=sys.stderr)
        sys.exit(1)

    rates: dict[tuple[str, str], float] = {}
    all_currencies: set[str] = set()

    for rate_element in root.findall('.//rate'):
        from_curr: str = rate_element.get('from').upper()
        to_curr: str = rate_element.get('to').upper()
        rate_value: float = float(rate_element.get('rate'))
        rates[(from_curr,to_curr)] = rate_value
        all_currencies.add(from_curr)
        all_currencies.add(to_curr)

    for currency in all_currencies:
        rates[(currency, currency)] = 1.0

    return rates, all_currencies

rates, all_currencies = load_conversion_rates("currency.xml")

def price_filter(currency: str, min_price: float, max_price: float) -> str:
    return f"(price_{currency.lower()} >= {min_price} and price_{currency.lower()} <= {max_price})"

def generate_price_filter_query(min_price: float, max_price: float, currency: str) -> str:
    if min_price > max_price:
        raise ValueError("min_price cannot be greater than max_price.")

    source_currency: str = currency.upper()

    or_conditions: list[str] = []
    for target_currency in all_currencies:
        if (source_currency, target_currency) in rates:
            rate: float = rates[(source_currency, target_currency)]
            converted_min: float = min_price * rate
            converted_max: float = max_price * rate

            or_conditions.append(price_filter(target_currency, converted_min, converted_max))
        else:
            print(f"Warning: No conversion rate from {source_currency} to {target_currency}. Skipping.", file=sys.stderr)

    return " or ".join(or_conditions)

def main() -> None:
    """
    Main function to generate the Vespa query.
    """
    parser = argparse.ArgumentParser(
        description="Generate a Vespa query to filter by price across multiple currencies."
    )
    parser.add_argument('--min_price', type=float, required=True, help='Minimum price.')
    parser.add_argument('--max_price', type=float, required=True, help='Maximum price.')
    parser.add_argument('--currency', type=str, required=True, help='The currency for the given min/max price (e.g., USD).')

    args = parser.parse_args()
    print(generate_price_filter_query(args.min_price, args.max_price, args.currency))

if __name__ == "__main__":
    main()
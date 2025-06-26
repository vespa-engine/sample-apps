import sys
import json
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

def load_conversion_rates(xml_file):
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

    rates = defaultdict(dict)
    all_currencies = set()

    for rate_element in root.findall('.//rate'):
        from_curr = rate_element.get('from').upper()
        to_curr = rate_element.get('to').upper()
        rate_value = float(rate_element.get('rate'))
        rates[from_curr][to_curr] = rate_value
        all_currencies.add(from_curr)
        all_currencies.add(to_curr)

    for currency in all_currencies:
        rates[currency][currency] = 1.0

    return rates, sorted(list(all_currencies))

def main():
    """
    Main function to generate the Vespa query.
    """
    parser = argparse.ArgumentParser(
        description="Generate a Vespa query to filter by price across multiple currencies."
    )
    parser.add_argument('--min_price', type=float, required=True, help='Minimum price.')
    parser.add_argument('--max_price', type=float, required=True, help='Maximum price.')
    parser.add_argument('--currency', type=str, required=True, help='The currency for the given min/max price (e.g., USD).')
    parser.add_argument('--currency_file', type=str, default='currency.xml', help='Path to the currency conversion XML file.')

    args = parser.parse_args()

    if args.min_price > args.max_price:
        print("Error: min_price cannot be greater than max_price.", file=sys.stderr)
        sys.exit(1)

    rates, all_currencies = load_conversion_rates(args.currency_file)
    source_currency = args.currency.upper()

    if source_currency not in rates:
        print(f"Error: The specified currency '{source_currency}' is not found in the conversion table.", file=sys.stderr)
        sys.exit(1)

    or_conditions = []
    for target_currency in all_currencies:
        if target_currency in rates[source_currency]:
            rate = rates[source_currency][target_currency]
            converted_min = args.min_price * rate
            converted_max = args.max_price * rate

            yql = f"(currency_ref matches \"id:shopping:currency::{target_currency.lower()}\" and price >= {converted_min} and price <= {converted_max})"

            or_conditions.append(yql)
        else:
            print(f"Warning: No conversion rate from {source_currency} to {target_currency}. Skipping.", file=sys.stderr)

    print(" or ".join(or_conditions))

if __name__ == "__main__":
    main()
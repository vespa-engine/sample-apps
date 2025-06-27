import sys
import xml.etree.ElementTree as ET
import json

def convert_currency_xml_to_vespa_jsonl(xml_file) -> list[str]:
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Add USD to USD conversion (factor = 1.0)
    usd_doc = {
        "put": "id:shopping:currency::usd",
        "fields": {"factor": 1.0}
    }
    currency_rates = [json.dumps(usd_doc) + '\n']

    # Find all rate elements where 'to' attribute is 'USD'
    for rate in root.findall('.//rate[@to="USD"]'):
        from_currency = rate.get('from').lower()
        factor = float(rate.get('rate'))

        # Create Vespa document
        doc = {
            "put": f"id:shopping:currency::{from_currency}",
            "fields": {"factor": factor}
        }

        currency_rates.append(json.dumps(doc))

    return currency_rates

# Usage
if __name__ == "__main__":
    currency_docs = convert_currency_xml_to_vespa_jsonl('currency.xml')
    sys.stdout.write("\n".join(currency_docs) + "\n")
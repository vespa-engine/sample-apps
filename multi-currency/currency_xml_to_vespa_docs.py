import sys
import xml.etree.ElementTree as ET
import json

def convert_currency_xml_to_vespa_jsonl(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Add USD to USD conversion (factor = 1.0)
    usd_doc = {
        "put": "id:shopping:currency::usd",
        "fields": {"factor": 1.0}
    }
    sys.stdout.write(json.dumps(usd_doc) + '\n')

    # Find all rate elements where 'to' attribute is 'USD'
    for rate in root.findall('.//rate[@to="USD"]'):
        from_currency = rate.get('from').lower()
        factor = float(rate.get('rate'))

        # Create Vespa document
        doc = {
            "put": f"id:shopping:currency::{from_currency}",
            "fields": {"factor": factor}
        }

        # Write to stdout
        sys.stdout.write(json.dumps(doc) + '\n')

# Usage
if __name__ == "__main__":
    convert_currency_xml_to_vespa_jsonl('currency.xml')
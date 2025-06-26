# Multi-Currency Vespa Application

This Vespa application demonstrates multi-currency price handling using global documents holding currency conversion rates to USD.
Item prices are stored in their local currencies, but the app can hydrate and rank items based on their USD equivalent prices.
Price range Filtering can be done by using native currency filtering.

## Architecture

The application consists of two document types:
- `currency`: Global document storing currency conversion factors to USD
- `item`: Items with prices in different currencies, referencing currency documents

The `currency.xml` file contains conversion rates between 30 different currencies.

## Setup

1. **Start Vespa container**:
   ```bash
   vespa config set target local
   docker pull vespaengine/vespa
   docker run --detach --name vespa --hostname vespa-container \
     --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
     vespaengine/vespa
   ```

2. **Wait for Vespa to be ready**:
   ```bash
   vespa status deploy --wait 300
   ```

3. **Deploy the application**:
   ```bash
   vespa deploy --wait 300
   ```

## Loading Data

1. **Feed currency conversion factors**:

Convert the `currency.xml` file to Vespa documents with conversion factors to USD.:
```bash
python3 currency_xml_to_vespa_docs.py | vespa feed -
```

The documents look like:
```jsonl
{"put": "id:mynamespace:currency::usd", "fields": {"factor": 1.0}}
{"put": "id:mynamespace:currency::aud", "fields": {"factor": 0.67884054}}
{"put": "id:mynamespace:currency::cad", "fields": {"factor": 0.76028283}}
{"put": "id:mynamespace:currency::cny", "fields": {"factor": 0.1442157}}
```

2. **Feed items with currency references**:

Feed the sample documents:

```bash
vespa feed items.jsonl
```

The item documents look like: 
 
```bash
{"put": "id:shopping:item::item-1", "fields": {"currency_ref": "id:shopping:currency::usd", "price": 3836, "item_name": "emerald gemstone bracelet"}}
{"put": "id:shopping:item::item-2", "fields": {"currency_ref": "id:shopping:currency::usd", "price": 14, "item_name": "Handmade ceramic ring dish"}}
{"put": "id:shopping:item::item-3", "fields": {"currency_ref": "id:shopping:currency::usd", "price": 45, "item_name": "Handmade wooden cutting board"}}
```

## Querying

1. **View all documents**:
   ```bash
   vespa visit
   ```

2. **Query items with currency-based price filtering**:
```bash
vespa query 'select * from item where (currency_ref matches "id:shopping:currency::usd" and price >= 4000.0)'
```

3. **Filter by all currencies within a range**:
   ```bash
    vespa query yql="select * from item where $(python3 generate_price_filter_query.py --min_price 20 --max_price 100 --currency USD)"
   ```
4. **Combine currency filtering with ranking using USD price**:
   ```bash
   vespa query yql="select * from item where userQuery() AND ($(python3 generate_price_filter_query.py --min_price 20 --max_price 100 --currency USD))" query="vintage"
   ```


## Key Features

- **Global currency documents**: Currency data is replicated across all content nodes
- **Cross-document field import**: Items can access currency factors via `currency_ref.factor`
- **USD price calculation**: Rank profile computes `usd_price: price * currency_factor`

## Schema Details

### Currency Schema
- `factor`: Double field representing conversion rate to USD

### Item Schema
- `item_name`: String field for item description
- `price`: Double field for price in local currency
- `currency_ref`: Reference to currency document
- Imported field: `currency_factor` from referenced currency document

## TODOs

- Show how to hydrate a USD price 
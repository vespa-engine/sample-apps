# Sample e-commerce application

## Deploy the application to Vespa

You'll need to change [product.sd](ecommerce/schemas/product.sd) to match the fields in the [CSV file](ecommerce/ext/myntra_products_catalog.csv). Then deploy the app:

```
cd ecommerce
vespa deploy
```

## Feeding data
The data is already converted to JSONL format in [products.jsonl](ecommerce/ext/products.jsonl) via the [generate_jsonl.py](ecommerce/ext/generate_jsonl.py) script. So you can just feed it to Vespa:
```
vespa feed ext/products.jsonl
```

## Feeding data with Logstash from the CSV file

You can also feed the data with Logstash from the CSV file directly. You'll need to [install Logstash](https://www.elastic.co/downloads/logstash), then:

1. Install the [Logstash Output Plugin for Vespa](https://github.com/vespa-engine/vespa/tree/master/integration/logstash-plugins/logstash-output-vespa) via:

```
bin/logstash-plugin install logstash-output-vespa_feed
```

2. Change [logstash.conf](ecommerce/ext/logstash.conf) to point to the absolute path of [myntra_products_catalog.csv](ecommerce/ext/myntra_products_catalog.csv).

3. Run Logstash with the modified `logstash.conf`:

```
bin/logstash -f $PATH_TO_LOGSTASH_CONF/logstash.conf
```

For more examples of using Logstash with Vespa, check out [this tutorial blog post](https://blog.vespa.ai/logstash-vespa-tutorials/).
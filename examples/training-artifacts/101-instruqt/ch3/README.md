# Sample sales data application

## Deploy the application to Vespa

```
cd part-purchase
vespa deploy
```

## Feeding data
The data is already converted to JSONL format in [sales-data.jsonl](sales-data.jsonl). So you can just feed it to Vespa:
```
vespa feed ../sales-data.jsonl
```

## Feeding data with Logstash from the CSV file

You can also feed the data with Logstash from the CSV file. You'll need to [install Logstash](https://www.elastic.co/downloads/logstash), then:

1. Install the [Logstash Output Plugin for Vespa](https://github.com/vespa-engine/vespa/tree/master/integration/logstash-plugins/logstash-output-vespa) via:

```
bin/logstash-plugin install logstash-output-vespa_feed
```

2. Change [logstash.conf](logstash.conf) to point to the absolute path of [sales-data.csv](sales-data.csv).

3. Run Logstash with the modified `logstash.conf`:

```
bin/logstash -f $PATH_TO_LOGSTASH_CONF/logstash.conf
```

For more examples of using Logstash with Vespa, check out [this tutorial blog post](https://blog.vespa.ai/logstash-vespa-tutorials/).
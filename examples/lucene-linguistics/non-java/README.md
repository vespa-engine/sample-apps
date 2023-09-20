# Lucene Linguistics in non-Java Vespa applications

In non-java projects it is possible to use Lucene Linguistics as a jar bundle.

Download and add the Vespa bundle jar into the `components` directory:
```shell
(mkdir -p components && cd components && curl -L https://github.com/dainiusjocas/vespa-lucene-linguistics-bundle/releases/download/v0.0.2/lucene-linguistics-bundle-0.0.2-deploy.jar --output lucene-linguistics-bundle-0.0.2-deploy.jar)
```

Deploy the application package:
```shell
vespa deploy -w 100
```

Run a query:
```shell
vespa query 'query=Vespa' 'language=lt'
```

The logs should contain record:
```text
[2023-08-16 11:21:04.847] INFO    container        Container.com.yahoo.language.lucene.AnalyzerFactory	Analyzer for language=lt is from a list of default language analyzers.
```

Profit.

The jar is hosted on [Github](https://github.com/dainiusjocas/vespa-lucene-linguistics-bundle/releases).

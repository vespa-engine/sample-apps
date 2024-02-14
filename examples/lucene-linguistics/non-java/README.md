# Lucene Linguistics in non-Java Vespa applications

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
[2024-02-14 13:30:09.847] CONFIG  container        Container.com.yahoo.language.lucene.AnalyzerFactory	Using Analyzer for AnalyzerKey[language=LITHUANIAN, stemMode=BEST, removeAccents=false] from a list of default language analyzers
```

Profit.

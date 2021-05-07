# vespa-chinese-linguistics
## Overview

This package provides Chinese tokenizer with Vespa using Jieba.
Jieba is one of the famous Chinese tokenizer, it is used by various services such as Solr, Elasticsearch, QihooVerticalSearch(based on lucene)and so on.

* [Jieba](https://github.com/huaban/jieba-analysis)


## Create Package

### Requirement

JDK (>= 11) and maven are required to build package.

### Build

Execute mvn command as below to produce the component jar target/chinese-linguistics-1.0.0-deploy.jar

```
$ cd vespa-chinese-linguistics
$ mvn package
```

## Use Package

### Deploy

Put the built package to components directory of your service. If there is no components directory, create it. For example, the structure will be like below with sampleapps.

* sampleapps/search/kosmos/
    * services.xml
    * components/
        * chinese-linguistics-1.0.0-deploy.jar

### Configuration

Because the package will be used by searcher and indexer, it is recommended to define &lt;component&gt; in all &lt;jdisc&gt; sections of services.xml.

```
<container id="mycontainer" version="1.0">
    <component id="com.qihoo.language.jieba.JiebaLinguistics" bundle="chinese-linguistics" >
      <config name="com.qihoo.language.jieba.dicts-loc">
      <dictionaryPath>/opt/vespa/conf/dict</dictionaryPath>
      <stopwordsPath>/opt/vespa/conf/stopwords</stopwords>
      </config>
    </component>
  </container>
```


## License

Code licensed under the Apache 2.0 license. See LICENSE for terms.

refer to the license of https://github.com/huaban/jieba-analysis

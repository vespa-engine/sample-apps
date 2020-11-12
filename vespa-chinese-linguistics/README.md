# vespa-chinese-linguistics
## Overview

This package provides Chinese tokenizer with Vespa using Jieba.
Jieba is one of the famous Chinese tokenizer, it is used by various services such as Solr, Elasticsearch, QihooVerticalSearch(based on lucene)and so on.

* [Jieba](https://github.com/huaban/jieba-analysis)


## Create Package

### Requirement

JDK (>= 11) and maven are required to build package.

### Build

Execute mvn command as below, and you can get package as target/kuromoji-linguistics-${VERSION}-deploy.jar

```
$ cd vespa-chinese-linguistics
$ mvn install:install-file -Dfile=./libs/jieba-analysis-1.0.3-SNAPSHOT.jar -DgroupId=com.huaban.analysis -DartifactId=jieba -Dversion=1.0.3-SNAPSHOT -Dpackaging=jar
$ mvn package -Dvespa.version='7.301.24'     # You can specify 7.301.24 or later.
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
      <dict>/opt/vespa/conf/dict</dict>
      <stopwords>/opt/vespa/conf/stopwords</stopwords>
      </config>
    </component>
  </container>
```


## License

Code licensed under the Apache 2.0 license. See LICENSE for terms.

Note that only for contributions to the vespa-chinese-linguistics repository on the GitHub (https://github.com/kuipertan/vespa-chinese-linguistics),
the contributors of them shall be deemed to have agreed to the CLA without individual written agreements.

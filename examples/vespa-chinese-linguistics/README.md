<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# vespa-chinese-linguistics
This package provides Chinese tokenizer with Vespa using Jieba.
Jieba is one of the famous Chinese tokenizer,
it is used by various services such as
Solr, Elasticsearch, QihooVerticalSearch (based on Lucene) and so on.

* [Jieba](https://github.com/huaban/jieba-analysis)


## Create Package


### Requirements
* JDK (>= 11)
* maven


### Build
Execute mvn command as below to produce the component jar target/chinese-linguistics-1.0.0-deploy.jar

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/examples/vespa-chinese-linguistics
$ mvn package
</pre>

Build package with version 7.301.24 there.
Maybe in higher version, there are some changes in the interface [Linguistics](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java).


## Use Package


### Deploy
Put the built package to components directory of your service.
If there is no _components_ directory, create it.
For example, the structure will be like below with sampleapps.

* sampleapps/search/kosmos/
    * services.xml
    * components/
        * chinese-linguistics-1.0.0-deploy.jar


### Configuration

Because the package will be used by searcher and indexer,
it is recommended to define &lt;component&gt; in all &lt;container&gt; sections of services.xml.

```xml
<container id="mycontainer" version="1.0">
    <component id="com.qihoo.language.JiebaLinguistics" bundle="chinese-linguistics" >
        <config name="com.qihoo.language.config.dicts-loc">
            <dictionaryPath>/opt/vespa/conf/dict</dictionaryPath>
            <stopwordsPath>/opt/vespa/conf/stopwords</stopwordsPath>
        </config>
    </component>
</container>
```


## License
Code licensed under the Apache 2.0 license. See LICENSE for terms.
Refer to the license of https://github.com/huaban/jieba-analysis.

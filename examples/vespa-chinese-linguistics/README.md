
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# vespa-chinese-linguistics
This package provides Chinese tokenizer with Vespa using Jieba.
Jieba is one of the famous Chinese tokenizer,
it is used by various services such as
Solr, Elasticsearch, QihooVerticalSearch (based on Lucene) and so on.

* [Jieba](https://github.com/huaban/jieba-analysis)


## Create Package


### Requirements
* <a href="https://openjdk.org/projects/jdk/17/" data-proofer-ignore>Java 17</a> installed.
* [Apache Maven](https://maven.apache.org/install.html) 


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
        * vespa-chinese-linguistics-1.0.0-deploy.jar


### Configuration

Because the package will be used by searcher and indexer,
it is recommended to define &lt;component&gt; in all &lt;container&gt; sections of services.xml.

```xml
<container id="mycontainer" version="1.0">
    <component id="com.qihoo.language.JiebaLinguistics" bundle="vespa-chinese-linguistics" >
        <config name="com.qihoo.language.config.jieba">
            <dictionary>myAppPackageDir/dictionaryFile.dict</dictionary>  <!-- Optional and not usually needed -->
            <stopwords>myAppPackageDir/stopwordsFile</stopwords>  <!-- Optional and not usually needed -->
        </config>
    </component>
</container>
```

User dict files should ends with `.dict`, such as `/opt/vespa/conf/jieba/user.dict`.

## License
Code licensed under the Apache 2.0 license. See LICENSE for terms.
Refer to the license of https://github.com/huaban/jieba-analysis.

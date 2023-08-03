# Vespa Lucene Linguistics

This Vespa application package (VAP) previews the configuration options of the `lucene-linguistics` package.
Probably the main benefit of the `LuceneLinguistics` is the configurability when compared to other `Linguistics` implementations. 

## The `lucene-linguistics` configuration trick

With the configurability comes risks of misconfiguration.
To prevent the most common failures due to misconfiguration (see further section) we propose an initial VAP directory setup that is known to work well.
The steps are:
1. In the VAP root create a directory `ext/linguistics`
2. In the `services.xml` under the `<container>` tag add
```xml
<components>
  <include dir="ext/linguistics"/>
</components>
```
3. In the `ext/linguistics` directory create an XML file called `config.xml` (can be any name)
4. In the `config.xml` file add the following content:
```xml
<components>
  <component id="linguistics"
             class="com.yahoo.language.lucene.LuceneLinguistics"
             bundle="vespa-lucene-linguistics-poc">
    <config name="com.yahoo.language.lucene.lucene-analysis">
      <configDir>ext/linguistics</configDir>
    </config>
  </component>
</components>
```

And voilà! Your VAP is ready to be deployed.
The essence of the trick is that in the VAP you create a directory which is not empty and therefore the VAP can be successfully deployed.
Also, by doing this you isolate the linguistics configurations (which can get quite verbose) from polluting the `services.xml`.
And this structure is guaranteed not to clash with the expected [VAP directory structure](https://docs.vespa.ai/en/reference/application-packages-reference.html).

## Custom Lucene Analyzers

There are multiple ways to make a Lucene `Analyzer` for a language.
Each analyzer is identified by a language key, e.g. 'en' for English language. 
These are Analyzer types in the order of descending priority:
1. Created through the `Linguistics` component configuration.
2. An `Analyser` wrapped into a Vespa `<component>`.
3. A list of [default Analyzers](https://github.com/vespa-engine/vespa/blob/5d26801bc63c35705e708d3cc7086f0b0103e909/lucene-linguistics/src/main/java/com/yahoo/language/lucene/DefaultAnalyzers.java) per language.
4. The `StandardAnalyzer`.

### Add a Lucene Analyzer component

Vespa provides a `ComponentRegistry` mechanism.
The `LuceneLinguistics` accepts a `ComponentRegistry<Analyzer>` into the constructor.
Basically, the Vespa container at start time collects all the components that are of the `Analyzer` type automagically.

To declare such components:
```xml
<component id="en"
           class="org.apache.lucene.analysis.core.SimpleAnalyzer"
           bundle="vespa-lucene-linguistics-poc" />
```
Where:
- `id` should contain a language code.
- `class` should be the implementing class.
Note that it is a class straight from the Lucene library.
Also, you can create an `Analyzer` class just inside your VAP and refer it.
- `bundle` must be your application package `artifactId` as specified in the `pom.xml`.

Here are two types of `Analyzer` components:
1. That doesn't require any setup.
2. That requires a setup (e.g. constructor with arguments).

The previous component declaration example is of type (1).

The (2) type requires a bit more work.

Create a class (e.g. for the Polish language):
```java
package ai.vespa.linguistics.pl;

import com.yahoo.container.di.componentgraph.Provider;
import org.apache.lucene.analysis.Analyzer;

public class PolishAnalyzer implements Provider<Analyzer> {
    @Override
    public Analyzer get() {
        return new org.apache.lucene.analysis.pl.PolishAnalyzer();
    }
    @Override
    public void deconstruct() {}
}
```

Add a component declaration into the `config.xml` (or `services.xml`) file:
```xml
<component id="pl"
           class="ai.vespa.linguistics.pl.PolishAnalyzer"
           bundle="vespa-lucene-linguistics-poc" />
```
And now you have a Polish language handling available.

## Common Problems

Most of the problems stem from the fact that configuration parameters of type `path` doesn't support default [values](https://github.com/vespa-engine/vespa/issues/27912).

The `lucene-linguistics` component **requires** to specify the `configDir` configuration parameter of type `path`.
`configDir` is a directory to store linguistics resources, e.g. dictionaries with stopwords, etc., and is relative to the VAP root directory.

### `configDir` is not specified 

If the `configDir` is not specified the `vespa deploy` fails with:
```shell
Uploading application package ... failed
Error: invalid application package (400 Bad Request)
Invalid application:
Unable to send file specified in com.yahoo.language.lucene.lucene-analysis:
Unable to send file for field 'configDir':
Invalid config value null
```

### `configDir` is specified but doesn't exist

If the `configDir` doesn't exist then `vespa deploy` would fail with such error:

```shell
Uploading application package ... failed
Error: invalid application package (400 Bad Request)
Invalid application:
Unable to send file specified in com.yahoo.language.lucene.lucene-analysis:
/opt/vespa/var/db/vespa/config_server/serverdb/tenants/default/sessions/4/lucene (No such file or directory)
```

### Application package root cannot be used as `configDir`

If you try to be clever and set `<configDir>.</configDir>` then application package would be deployed(!) BUT
not converge with the following error:
```shell
Uploading application package ... done

Success: Deployed target/application.zip
WARNING Jar file 'vespa-lucene-linguistics-poc-0.0.1-deploy.jar' uses non-public Vespa APIs: [com.yahoo.language.simple]

Waiting up to 1m40s for query service to become available ...
Error: service 'query' is unavailable: services have not converged
```

And Vespa logs would be filled with such warnings:
```shell
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	Exception in thread "Rpc executorpool-6-thread-5" java.lang.RuntimeException: More than one file reference found for file 'fbcf5c3dc81d9540'
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.filedistribution.FileDownloader.getFileFromFileSystem(FileDownloader.java:109)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.filedistribution.FileDownloader.getFileFromFileSystem(FileDownloader.java:100)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.filedistribution.FileDownloader.getFutureFile(FileDownloader.java:80)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.filedistribution.FileDownloader.getFile(FileDownloader.java:70)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.config.proxy.filedistribution.FileDistributionRpcServer.downloadFile(FileDistributionRpcServer.java:109)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.config.proxy.filedistribution.FileDistributionRpcServer.lambda$getFile$0(FileDistributionRpcServer.java:84)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1136)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:635)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat java.base/java.lang.Thread.run(Thread.java:833)
```

### Empty directory can't be referred

If the `configDir` is set with `foo` which is empty then during deployment you get a misleading error message:
```shell
Uploading application package ... failed
Error: invalid application package (400 Bad Request)
Invalid application:
Unable to send file specified in com.yahoo.language.lucene.lucene-analysis:
/opt/vespa/var/db/vespa/config_server/serverdb/tenants/default/sessions/8/foo (No such file or directory)
```

### Harmless warning
`vespa deploy` always warns with:
```shell
WARNING Jar file 'vespa-lucene-linguistics-poc-0.0.1-deploy.jar' uses non-public Vespa APIs: [com.yahoo.language.simple]
```
You can ignore this warning.


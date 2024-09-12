
<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Lucene Linguistics 

This app demonstrates using [Lucene Linguistics](https://docs.vespa.ai/en/lucene-linguistics.html).


<p data-test="run-macro init-deploy examples/lucene-linguistics/non-java">
Requires at least Vespa 8.315.19
</p>

## To try this application

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started)
through the <code>vespa deploy</code> step, cloning `examples/lucene-linguistics/non-java` instead of `album-recommendation`.

Feed 3 sample documents in Norwegian, Swedish, and Finnish: 

<pre data-test="exec">
vespa feed ext/*.json
</pre>

Example queries:

<pre data-test="exec" data-test-assert-contains="id:no:doc::1">
vespa query 'yql=select * from doc where userQuery()'\
 'language=no' 'summary=debug-text-tokens' \
 'query=tips til utendørsaktiviteter'
</pre>

<pre data-test="exec" data-test-assert-contains="id:sv:doc::1">
vespa query 'yql=select * from doc where userQuery()'\
 'language=sv' 'summary=debug-text-tokens' \
 'query=tips til utomhusaktiviteter'
</pre>

<pre data-test="exec" data-test-assert-contains="id:fi:doc::1">
vespa query 'yql=select * from doc where userQuery()'\
 'language=fi' 'summary=debug-text-tokens' \
 'query=vinkkejä ulkoilma-aktiviteetteihin'
</pre>

### Terminate container 

Remove the container after use (Only relevant for local deployments)
<pre data-test="exec">
$ docker rm -f vespa
</pre>


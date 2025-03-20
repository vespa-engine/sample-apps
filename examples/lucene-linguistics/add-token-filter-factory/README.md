# Vespa Lucene Linguistics: Adding custom Lucene components

## TL;DR

Shows how you can use an extended version of the lucene-linguistics
component which includes your own custom components, discoverable
using the classpath as expected by Lucene SPI mechanism.

## Details

The included pom.xml builds your own component called
"my-replacement-bundle" which includes a dummy implementation of
TokenFilterFactory. The class name of the factory must be in
META-INF/services/org.apache.lucene.analysis.TokenFilterFactory
from src/main/resources/ directory.

After running "mvn install", you can copy the finished
target/my-replacement-bundle-1.0.1-deploy.jar to the
"components" directory in your application package.
Put the snippet from "services.xml" into your services.xml
in your application as well to activate it.

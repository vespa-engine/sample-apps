<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa Cloud sample applications

### album-recommendation
This is the intro application to Vespa.
* A simple, complete application doing search and recommendation in music data: [album-recommendation](album-recommendation).
* The same application, with support for building Java components added: [album-recommendation-java](album-recommendation-java).

### cord-19-search
[cord19.vespa.ai](https://cord19.vespa.ai/) is a full-featured application - see
* [cord-19](https://github.com/vespa-engine/cord-19): frontend
* [cord-19-search](cord-19-search): search backend

### vespa-documentation-search
[vespa-documentation-search](vespa-documentation-search) is a search application -
refer to this for pointers on AWS integration / GitHub Actions automation.
This sample app is a good start for [automated deployments](https://cloud.vespa.ai/en/automated-deployments),
as it has system, staging and production test examples.
It uses the [Document API](https://docs.vespa.ai/en/document-api-guide.html)
both for regular PUT operations but also for UPDATE with _create-if-nonexistent_.


----

These applications are written for [Vespa Cloud](http://cloud.vespa.ai).
To deploy to a self-hosted, like a Docker container on developer computer, follow steps below -
refer to [album-recommendation](../album-recommendation) for a working example

1.  Modify _nodes_ element in services.xml, replace _count_ with _node_ elements:
    ```
    <nodes count="1" />
    
    <!-- For container node -->
    <nodes>
        <node hostalias="node1" />
    </nodes>

    <!-- For content node -->
    <nodes>
        <node hostalias="node1" distribution-key="0" />
    </nodes>
    ```

1.  Add _hosts.xml_:
    ```
    <?xml version="1.0" encoding="UTF-8"?>
    <hosts>
        <host name="localhost">
            <alias>node1</alias>
        </host>
    </hosts>
    ```

Cheatsheet for quick Docker run after config changes above:

    $ mvn clean package
    $ docker run --detach --name vespa --hostname vespa-container --volume $(pwd):/approot \
      -p 8080:8080 -p 19092:19092 vespaengine/vespa
    $ docker exec vespa bash -c \
      '/opt/vespa/bin/vespa-deploy prepare /approot/target/application.zip && \
      /opt/vespa/bin/vespa-deploy activate'

Expose ports as needed, like the metrics proxy port: http://localhost:19092/prometheus/v1/values

----

Note: Applications with _pom.xml_ must be built before being deployed.
Refer to [getting started](https://docs.vespa.ai/en/getting-started.html) for more information.

[Contribute](https://github.com/vespa-engine/vespa/blob/master/CONTRIBUTING.md)
to the Vespa sample applications.

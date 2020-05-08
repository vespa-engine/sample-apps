<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa Cloud sample applications


### album-recommendation
This is the intro application to Vespa.
* Learn how to configure the schema for simple recommendation and search use cases in [album-recommendation](album-recommendation).
* [album-recommendation-java](album-recommendation-java) is an introduction for how to integrate Java code to process data and queries.
* [album-recommendation-docproc](album-recommendation-docproc) is an introduction to document processing

### cord-19-search
https://cord19.vespa.ai/ is a is a full-featured application - see
* https://github.com/vespa-engine/cord-19 : frontend
* [cord-19-search](cord-19-search) : search backend

### vespa-documentation-search
[vespa-documentation-search](vespa-documentation-search) is a simple search application - refer to this for pointers on AWS integration / GitHub Actions automation 

----

These applications are written for [Vespa Cloud](http://cloud.vespa.ai).
To deploy to a self-hosted, like a Docker container on developer computer, follow steps below -
refer to [album-recommendation-selfhosted](../album-recommendation-selfhosted) for a working example

1.  Modify _nodes_ element in services.xml, replace _count_ with _node_ elements:
    ```
    <nodes count="1" />
    
    <nodes>
        <node hostalias="node1" distribution-key="0" />
    </nodes>
    ```

1.  Add _hosts.xml_

1.  Remove _<client-authorize />_ in services.xml. 

----

Note: Applications with pom.xml must be built before being deployed.
Refer to
[developing applications](http://docs.vespa.ai/documentation/jdisc/developing-applications.html#deploy)
for more information.

[Contribute](https://github.com/vespa-engine/vespa/blob/master/CONTRIBUTING.md)
to the Vespa sample applications.

<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications

This repository has a set of Vespa sample applications.
Most of the applications are written for [Vespa Cloud](http://cloud.vespa.ai).
To deploy to a self-hosted, follow steps below -
refer to [album-recommendation-selfhosted](album-recommendation-selfhosted) for a working example

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

Get started using the [vespa quick start guide](https://cloud.vespa.ai/getting-started).

Note: Applications with pom.xml must be built before being deployed.
Refer to
[developing applications](http://docs.vespa.ai/documentation/jdisc/developing-applications.html#deploy)
for more information.

[Contribute](https://github.com/vespa-engine/vespa/blob/master/CONTRIBUTING.md)
to the Vespa sample applications.

----

Travis-CI build status: [![Build Status](https://travis-ci.org/vespa-engine/sample-apps.svg?branch=master)](https://travis-ci.org/vespa-engine/sample-apps)



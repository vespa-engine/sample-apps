<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
Predicate Search
==================

Predicate/Boolean Search and how to feed and query is described in 
[predicate search](http://docs.vespa.ai/documentation/predicate-fields.html).

To deploy this sample application see [Developing applications](http://docs.vespa.ai/documentation/jdisc/developing-applications.html).

Adding predicate search to an application is easy. Just add a field of
type predicate to the .sd file. (Remember to set the arity parameter.)


### Feed and search
1. **Feed** the data that is to be searched:
    ```sh
    curl -X POST --data-binary @adsdata.xml <endpoint url>/document
    ```

2. **Search** using yql expressions, e.g. `select * from sources * where predicate(target, {"name":"Wile E. Coyote"},{});`
    ```sh
    curl "<endpoint url>/search/?query=sddocname:ad&yql=select%20*%20from%20sources%20*%20where%20predicate(target%2C%20%7B%22name%22%3A%22Wile%20E.%20Coyote%22%7D%2C%7B%7D)%3B"
    ```

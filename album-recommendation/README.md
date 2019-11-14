<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - album recommendations
See [getting started](http://cloud.vespa.ai/getting-started.html) for troubleshooting.

Vespa is used for online Big Data serving, which means ranking (large) data sets using query data.
Below is an example of how to rank music albums using a user profile -
match albums with scores for a set of categories with a user's preference:

**User profile**

    {
      { cat:pop  }: 0.8,
      { cat:rock }: 0.2,
      { cat:jazz }: 0.1
    }

 **Albums**

    {
      "fields": {
        "album": "A Head Full of Dreams",
        "artist": "Coldplay",
        "year": 2015,
        "category_scores": {
          "cells": [
            { "address": { "cat": "pop"},  "value": 1  },
            { "address": { "cat": "rock"}, "value": 0.2},
            { "address": { "cat": "jazz"}, "value": 0  }
          ]
        }
      }
    }

    {
      "fields": {
        "album": "Love Is Here To Stay",
        "artist": "Diana Krall",
        "year": 2018,
        "category_scores": {
          "cells": [
            { "address": { "cat": "pop" },  "value": 0.4 },
            { "address": { "cat": "rock" }, "value": 0   },
            { "address": { "cat": "jazz" }, "value": 0.8 }
          ]
        }
      }
    }

    {
      "fields": {
        "album": "Hardwired...To Self-Destruct",
        "artist": "Metallica",
        "year": 2016,
        "category_scores": {
          "cells": [
            { "address": { "cat": "pop" },  "value": 0 },
            { "address": { "cat": "rock" }, "value": 1 },
            { "address": { "cat": "jazz" }, "value": 0 }
          ]
        }
      }
    }

**Rank profile**

A [rank profile](https://docs.vespa.ai/documentation/ranking.html) calculates a _relevance score_ per document.
This is defined by the application author - in this case, it is the tensor product.
The data above is represented using [tensors](http://docs.vespa.ai/documentation/tensor-intro.html).
As the tensor is one-dimensional (the _cat_ dimension), this a vector,
hence this is the dot product of the user profile and album categories:

    rank-profile rank_albums inherits default {
        first-phase {
            expression: sum(query(user_profile) * attribute(category_scores))
        }
    }

Hence, the expected scores are:
<table>
<tr><th>Album</th>                                   <th>pop</th>     <th>rock</th>    <th>jazz</th>      <th>total</th></tr>
<tr><td>A Head Full of Dreams</td>         <td>0.8*1.0</td><td>0.2*0.2</td><td>0.1*0.0</td><td>0.84</td></tr>
<tr><td>Love Is Here To Stay</td>            <td>0.8*0.4</td><td>0.2*0.0</td><td>0.1*0.8</td><td>0.4</td></tr>
<tr><td>Hardwired...To Self-Destruct</td><td>0.8*0.0</td><td>0.2*1.0</td><td>0.1*0.0</td><td>0.2</td></tr>
</table>

Build and test the application, and validate that the document's _relevance_ is the expected value,
and the results are returned in descending relevance order.


### Executable example
1.  Get a X.509 certificate. To create a self-signed certificate
    (more details in  in [Data Plane](https://cloud.vespa.ai/security-model.html#data-plane), see _Client certificate_), do
    ```sh
    $ openssl req -x509 -nodes -days 14 -newkey ec:<(openssl ecparam -name prime256v1) \
      -keyout data-plane-private-key.pem -out data-plane-public-cert.pem
    ```

1.  Go to https://console.vespa.oath.cloud/, click "Create application"

1.  Download sample app:
    ```sh
    $ git clone git@github.com:vespa-cloud/sample-apps.git && cd sample-apps/album-recommendation
    ```

1.  Create the application package
    ```sh
    $ mkdir -p src/main/application/security && cp data-plane-public-cert.pem src/main/application/security/data-plane-public-cert.pem
    $ cd src/main/application && zip -r ../../../application.zip . && cd ../../..
    ```

1.  Click Deploy. In the "Deploy to dev" console section, upload _application.zip_ - click Deploy

1.  Click "deployment log" to track the deployment. "Installation succeeded!" in the bottom pane indicates success 

1.  Click "Instances" at the top, then "endpoints". Try the endpoint to validate it is up:
    ```sh
    $ ENDPOINT=https://end.point.name
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem $ENDPOINT
    ```

1.  Feed documents
    ```sh
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      -H "Content-Type:application/json" --data-binary @src/test/resources/A-Head-Full-of-Dreams.json \
      $ENDPOINT/document/v1/mynamespace/music/docid/1
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      -H "Content-Type:application/json" --data-binary @src/test/resources/Love-Is-Here-To-Stay.json \
      $ENDPOINT/document/v1/mynamespace/music/docid/2
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      -H "Content-Type:application/json" --data-binary @src/test/resources/Hardwired...To-Self-Destruct.json \
      $ENDPOINT/document/v1/mynamespace/music/docid/3
    ```

1.  Recommend albums, send user profile in query
    ```sh
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      "$ENDPOINT/search/?ranking=rank_albums&yql=select%20%2A%20from%20sources%20%2A%20where%20sddocname%20contains%20%22music%22%3B&ranking.features.query(user_profile)=%7B%7Bcat%3Apop%7D%3A0.8%2C%7Bcat%3Arock%7D%3A0.2%2C%7Bcat%3Ajazz%7D%3A0.1%7D"
    ```

<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - album recommendations

Refer to the [Quick Start](https://cloud.vespa.ai/en/getting-started) to try this sample application.
This application ranks music albums using a user profile:
Albums with scores for a set of categories are matched with a user's preference.


### User profile
```
{
    { cat:pop  }: 0.8,
    { cat:rock }: 0.2,
    { cat:jazz }: 0.1
}
```


### Albums
```json
{
    "fields": {
        "album": "A Head Full of Dreams",
        "artist": "Coldplay",
        "year": 2015,
        "category_scores": {
            "cells": [
                { "address": { "cat": "pop"  }, "value": 1  },
                { "address": { "cat": "rock" }, "value": 0.2},
                { "address": { "cat": "jazz" }, "value": 0  }
          ]
        }
    }
},
{
    "fields": {
        "album": "Love Is Here To Stay",
        "artist": "Diana Krall",
        "year": 2018,
        "category_scores": {
            "cells": [
                { "address": { "cat": "pop"  }, "value": 0.4 },
                { "address": { "cat": "rock" }, "value": 0   },
                { "address": { "cat": "jazz" }, "value": 0.8 }
            ]
        }
    }
},
{
    "fields": {
        "album": "Hardwired...To Self-Destruct",
        "artist": "Metallica",
        "year": 2016,
        "category_scores": {
            "cells": [
                { "address": { "cat": "pop"  }, "value": 0 },
                { "address": { "cat": "rock" }, "value": 1 },
                { "address": { "cat": "jazz" }, "value": 0 }
            ]
        }
    }
}
```


### Rank profile
A [rank profile](https://docs.vespa.ai/en/ranking.html) calculates a _relevance score_ per document.
This is defined by the application author - in this case, it is the tensor product.
The data above is represented using [tensors](https://docs.vespa.ai/en/tensor-user-guide.html).
As the tensor is one-dimensional (the _cat_ dimension), this a vector,
hence this is the dot product of the user profile and album categories:

```
rank-profile rank_albums inherits default {
    first-phase {
        expression: sum(query(user_profile) * attribute(category_scores))
    }
}
```

Hence, the expected scores are:

<table>
<tr><th>Album</th>                       <th>pop</th>    <th>rock</th>   <th>jazz</th>   <th>total</th></tr>
<tr><td>A Head Full of Dreams</td>       <td>0.8*1.0</td><td>0.2*0.2</td><td>0.1*0.0</td><td>0.84</td></tr>
<tr><td>Love Is Here To Stay</td>        <td>0.8*0.4</td><td>0.2*0.0</td><td>0.1*0.8</td><td>0.4</td></tr>
<tr><td>Hardwired...To Self-Destruct</td><td>0.8*0.0</td><td>0.2*1.0</td><td>0.1*0.0</td><td>0.2</td></tr>
</table>

Build and test the application, and validate that the document's _relevance_ (i.e. score) is the expected value,
and the results are returned in descending relevance order.

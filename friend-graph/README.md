<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - Friend Graph Search 

Given a directed social graph (possible maintained outside of Vespa), that keeps tracks of the 
edges (relationship) in the social graph :

* user_id (uuid) is friend with user_id (uuid) with weight k

The graph could also be represented in Vespa, and fetched before executing the name search.

We want to search using n-gram matching over first name and last name, and rank users 
that are (always), regardless of text matching score. Similar mechanism can be used
for ranking user generated content in social networks.  

## Schema 
See [src/main/application/schemas/user.sd](src/main/application/schemas/user.sd) for full
version:
<pre>
schema user {
    document user {
        field uuid type string {}
        field firstname type string {}
        field lastname type string {}
        field id type tensor<int8>(uuid{}) {}
    }
    fieldset default {
      fields: firstname, lastname
    }
}
</pre>
The *id* tensor field is used for the tensor sparse dot product and only contain *one* cell. The reason
we use a tensor type for the *id* field is to optimize the sparse tensor dot product. The redundant
*uuid* field which could have the same value, is used for matching, as tensors generally cannot be used for matching,
only ranking. See [tensor user guide](https://docs.vespa.ai/en/tensor-user-guide.html), and 
[tensor computation example](https://docs.vespa.ai/en/tensor-examples.html).

## Ranking 
<pre>
    rank-profile friend inherits default {
      function social() {
        expression: sum(query(friends) * attribute(id)) 
      } 
      first-phase {
        expression: 2*social() + nativeRank(firstname) + nativeRank(lastname)
      }
      match-features: social() nativeRank(firstname) nativeRank(lastname)
    }
</pre>
The *social* function computes the sparse tensor dot product over the *uuid* named dimension. When using unweighted edges, the score
can maximum be 1. If weighted the score becomes the value of the user edge weight from the query tensor. 

## Setup
Follow quick start https://docs.vespa.ai/en/vespa-quick-start.html 
but change directoy to friend-graph instead.

<pre>
vespa clone friend-graph myapp && cd myapp
</pre>
## Feed documents

<pre>
vespa document -v src/test/resources/jon.json 
vespa document -v src/test/resources/kristian.json 
vespa document -v src/test/resources/jo-kristian.json 
</pre>

## Query 
The given user y is friends with Kristian (id = 1), but not Jo Kristian (id=0) and searches for "kri", in this case
we want Kristian to rank higher than Jo Kristian. In the example the user y has only one friend, in reality, probably more, so that
one would pass all the outbound edges of the user node in the graph in the sparse tensor representation:
<pre>
"ranking.features.query(friends)={{uuid:1}:1,{uuid:4}:1,{uuid:5}:1}"
</pre>
In this case the edges all have weight 1, using `int8` tensor cell precision we could also assign a weight to the edges.

**Query example**

<pre>
vespa query "select * from user where userQuery();" "query=kri" "ranking=friend" "ranking.features.query(friends)={{uuid:1}:1,{uuid:4}:1,{uuid:5}:1}"
</pre> 

<pre>
http://localhost:8080/search/?query=kri&ranking.features.query(friends)=%7B%7Buuid%3A1%7D%3A1%2C%7Buuid%3A2%7D%3A1%7D&ranking=friend&
</pre>

YQL Version 
<pre>
vespa query "select * from user where userQuery();" "query=kri" "ranking=friend" "ranking.features.query(friends)={{uuid:1}:1}"
</pre>

<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - Friend Graph Search 

Given a social graph maintained outside of Vespa which maintains the edges in the graph 

user_id is friend with user_id 

We want to search using n-gram matching over firstname and lastname and rank users 
that are connected higher (always), regardless of text matching. 

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

# Query 
Pretend current user is friends with Kristian (id = 1), but not Jo Kristian (id=0) and searches for "kri", in this case
we want Kristian to rank higher than Jo Kristian. 

<pre>
http://localhost:8080/search/?query=kri&ranking.features.query(friends)=%7B%7Buuid%3A1%7D%3A1%2C%7Buuid%3A2%7D%3A1%7D&ranking=friend&
</pre>

<pre>
vespa query "select * from user where userQuery();" "query=kri" "ranking=friend" "ranking.features.query(friends)={{uuid:1}:1}"
</pre>

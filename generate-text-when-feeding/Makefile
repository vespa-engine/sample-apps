
config:
	vespa config set target cloud
	vespa config set application vespa-team.textgen-feed

auth:
	vespa auth login
	vespa auth cert

deploy:
	vespa deploy --wait 1800

feed_1:
	vespa feed data/feed_1.jsonl
	
feed_10:
	vespa feed data/feed_10.jsonl

feed_100:
	vespa feed data/feed_100.jsonl

query_all:
	vespa query 'yql=select * from passage where true'

query:
	vespa query \
		--timeout 60 \
        query="science" \
        hits=5 \
        ranking=summarized\
        traceLevel=1
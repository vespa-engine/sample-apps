
config:
	vespa config set target cloud
	vespa config set application vespa-team.gentext-feed

auth:
	vespa auth login
	vespa auth cert

deploy:
	vespa deploy --wait 900
	
feed:
	vespa feed data/feed_one.jsonl
	
query:
	vespa query \
		--timeout 60 \
        query="hello" \
        hits=5 \
        ranking=local_llm_gen\
        traceLevel=1
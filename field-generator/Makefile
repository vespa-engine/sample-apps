export VESPA_CLI_CLOUD_SYSTEM=publiccd
application := vespa-team.feed-generator

config:
	vespa config set target cloud
	vespa config set application $(application)

auth:
	vespa auth login

cert:
	vespa auth cert add -f

deploy:
	vespa deploy --wait 3600

destroy:
	vespa destroy

# For LLM inference, reduce number of connections to 1 to avoid overloading the node.
feed_cmd = vespa feed data/feed_$(1).jsonl --connections 1 --verbose

feed_1:
	$(call feed_cmd,1)

feed_10:
	$(call feed_cmd,10)

feed_100:
	$(call feed_cmd,100)

feed_1000:
	$(call feed_cmd,1000)

query_100:
	vespa query 'yql=select * from passage where true' hits=100
export VESPA_CLI_CLOUD_SYSTEM=publiccd
application := vespa-team.field-generator

config:
	vespa config set target cloud
	vespa config set application $(application)

auth:
	vespa auth login

cert:
	vespa auth cert -f

#cert-add:
#	vespa auth cert add

deploy:
	vespa deploy --wait 3600

destroy:
	vespa destroy

# For LLM inference on 1 container node, reduce number of concurrent requests by setting connections to 1.
feed_cmd = vespa feed data/feed_$(1).jsonl --connections 12 --verbose

feed-1:
	$(call feed_cmd,1)

feed-10:
	$(call feed_cmd,10)

feed-100:
	$(call feed_cmd,100)

feed-1000:
	$(call feed_cmd,1000)

query-100:
	vespa query 'yql=select * from passage where true' 'hits=100' 'ranking=enriched'
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

target := cloud
#target := local

export VESPA_CLI_CLOUD_SYSTEM=public

tenant := vespa-team
application := $(tenant).field-generator

config:
	vespa config set target $(target)
	vespa config set application $(application)

auth:
	vespa auth login

 cert:
	vespa auth cert -f

 cert-add:
	vespa auth cert add

deploy:
	vespa deploy --wait 3600

destroy:
	vespa destroy

# To avoid timeout due to LLM inference latency set number of connections proportional to number of nodes: 
# - For CPU nodes - 1 connection per node
# - For GPU nodes - 3 connections per node
feed_cmd = vespa feed data/feed_$(1).jsonl --connections 1 --verbose

feed-1:
	$(call feed_cmd,1)

feed-10:
	$(call feed_cmd,10)

feed-100:
	$(call feed_cmd,100)

query-100:
	vespa query 'yql=select * from passage where true' 'hits=100' 'ranking=enriched'
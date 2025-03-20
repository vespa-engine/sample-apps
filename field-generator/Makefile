export VESPA_CLI_CLOUD_SYSTEM=publiccd
application := vespa-team.feed-generator
# Number connections = number of container nodes
connections := 1
feed_args := --connections $(connections) --verbose

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

feed_1:
	vespa feed data/feed_1.jsonl --connections $(connections) --verbose

feed_10:
	vespa feed data/feed_10.jsonl --connections $(connections) --verbose

feed_100:
	vespa feed data/feed_100.jsonl --connections $(connections) --verbose

feed_1000:
	vespa feed data/feed_1000.jsonl --connections $(connections) --verbose

query_100:
	vespa query 'yql=select * from passage where true' hits=100
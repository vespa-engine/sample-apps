
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Categorize using an LLM
This is a set of scripts/installs to back up our presentation at:
* [MLCon](https://mlconference.ai/machine-learning-advanced-development/adaptive-incontext-learning/)
* [data science connect COLLIDE](https://datasciconnect.com/events/collide/agenda/)

For any questions, please register at the
[Vespa Slack](https://join.slack.com/t/vespatalk/shared_invite/zt-nq61o73o-Lsun7Fnm5N8uA6UAfIycIg)
and discuss in the _general_ channel.


### Setup

Install [Ollama](https://ollama.com/) and run models like: 
```shell
ollama run llama3.1
```

Use the [quick start](https://docs.vespa.ai/en/vespa-quick-start.html) or
[Vespa getting started](https://cloud.vespa.ai/en/getting-started) to deploy this - laptop example:
```shell
podman run --detach --name vespa --hostname vespa-container \  
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
  
vespa deploy app --wait 600
```
Use e.g. _podman_ or _docker_ to run the `vespaengine/vespa` image on a laptop.


### Generate data

[feed_examples.py](feed_examples.py) converts the train data set to vespa feed format -
feed this to the Vespa instance:
```shell
python3 feed_examples.py > samples.jsonl
vespa feed samples.jsonl
```


### Evaluate data

[categorize_group.py](categorize_group.py) runs through the test set
and classifies based on examples retrieved from Vespa.

See the `inference` function for how to set up queries and ranking profiles for the different options.

Example script output:
```
category	size	relevance	retrieved_label	predicted_label	label_text	text
3	10	13.75663952228003	get_physical_card	get_physical_card	card_arrival	How do I locate my card?
0	10	19.146904249529296	card_arrival	card_arrival	card_arrival	I still have not received my new card, I ordered over a week ago.
```

### Other
Use the `@timer_decorator` to time execution time of the functions.

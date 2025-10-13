<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa Workshop - Quick Start Using Docker

See [https://docs.vespa.ai/en/vespa-quick-start.html](https://docs.vespa.ai/en/vespa-quick-start.html) for more information.

## Example Queries

```bash
vespa query \
  "yql=select * from product where name_description_index contains ({language: 'no'}'brød')" \
  "ranking.profile=native_rank" \
  "summary=debug-summary" \
  "presentation.timing=true" \
  "trace.level=7"

vespa query \
  "yql=select * from product where name_description_index_n_gram contains ({language: 'no'}'brød')" \
  "ranking.profile=native_rank_n_gram" \
  "summary=debug-summary" \
  "presentation.timing=true" \
  "trace.level=7"

vespa query \
  "yql=select * from product where name_description_attribute matches ({language: 'no'}'brød')" \
  "ranking.profile=native_rank_attribute" \
  "summary=debug-summary" \
  "presentation.timing=true" \
  "trace.level=7"
```

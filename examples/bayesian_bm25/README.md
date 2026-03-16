<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Bayesian BM25

This app demonstrates how to implement [Bayesian BM25](https://github.com/cognica-io/bayesian-bm25) style score calibration in Vespa rank profiles.

The schema uses Vespa ranking features such as [bm25(field)](https://docs.vespa.ai/en/reference/ranking/rank-features.html#bm25) and [fieldTermMatch](https://docs.vespa.ai/en/reference/ranking/rank-features.html#fieldtermmatch(namen).occurrences) to map raw BM25 scores to probabilities with:

- a sigmoid likelihood
- a composite prior based on term frequency and field length normalization
- corpus-level base rate update

The app defines three rank profiles in [app/schemas/doc.sd](app/schemas/doc.sd):

- `bayesian_bm25_simple` for sigmoid-only score mapping
- `bayesian_bm25` for posterior probability with a composite prior
- `bayesian_bm25_calibrated` for posterior probability with an additional base-rate correction

The main posterior expression looks like this:

```text
P(R|s) = (L * prior) / (L * prior + (1 - L) * (1 - prior))
```

This is useful when you want BM25 scores with a bounded probabilistic interpretation that is easier to inspect, threshold, or combine with other ranking signals.

### Set up the Vespa application

1. Log in to [Vespa Cloud](https://cloud.vespa.ai) and create a tenant, if you don't have one already.

2. Deploy your application using the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):

```bash
# point Vespa CLI to Vespa Cloud
vespa config set target cloud

# also point it to your tenant and application
# If you don't have an application, it will be created on deploy
vespa config set application YOUR_TENANT_NAME.YOUR_APPLICATION_NAME

# authenticate Vespa CLI with your Vespa Cloud credentials
vespa auth login

# go to the application package directory
# and set up the mTLS certificates
cd app
vespa auth cert

# deploy the application
vespa deploy --wait 900
```

**NOTE**: If you're running Vespa locally, you'd skip the security steps:

```bash
vespa config set target local
cd app
vespa deploy --wait 900
```

## Feed test data
Feed the sample documents:

```bash
vespa feed ../ext/sample-documents.jsonl
```

## Run a test query
Query the sample app with the Bayesian posterior profile:

```http
POST /search/
Content-Type: application/json

{
  "yql": "select * from doc where body contains \"machine\" and body contains \"learning\"",
  "ranking.profile": "bayesian_bm25"
}
```

The response includes summary features from the rank profile, such as:

```text
raw_bm25
likelihood
avg_tf
tf_prior
fieldnorm_prior
composite_prior
posterior
```

To run the calibrated variant with a base rate prior:

```http
POST /search/
Content-Type: application/json

{
  "yql": "select * from doc where body contains \"machine\" and body contains \"learning\"",
  "ranking.profile": "bayesian_bm25_calibrated",
  "ranking.features.query(base_rate)": 0.05
}
```

To override parameters at query time:

```http
POST /search/
Content-Type: application/json

{
  "yql": "select * from doc where body contains \"machine\" and body contains \"learning\"",
  "ranking.profile": "bayesian_bm25_calibrated",
  "ranking.features.query(alpha)": 1.5,
  "ranking.features.query(beta)": 2.0,
  "ranking.features.query(avg_field_len)": 35.0,
  "ranking.features.query(base_rate)": 0.02
}
```

# Notes/TODOs

- Provide a script to compute the base rate. This could simply run queries (like in paper at algorithm 4.4.7) or could be computed before every query from a [Searcher](https://docs.vespa.ai/en/applications/searchers.html) probably overkill. Having actual labels to do it would be even better (and more complicated).
- The TF prior should be computed as the average prior for each term in the query, instead of computing a single prior for average TFs. More complex to define in Vespa. Current implementation is OK, unless we have many terms.
- Vespa should expose average field length as a rank feature and we should reuse that.
- Ideally, we should have a way (scripts) to tune the hardcoded values when we compute priors.
- Using the [weakAnd](https://docs.vespa.ai/en/ranking/wand.html#weakand) in Vespa ignores the TF/length priors while pruning. This shouldn't make a big difference for most queries - increase `targetHits` to overfetch if tail end quality is a problem.
- When using BBM25 in hybrid search, remember that most vector scores aren't probabilities, even if they are in [0,1]. Comparing this to other methods ([atan](https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html#hybrid-ranking), [linear, RRF](https://docs.vespa.ai/en/ranking/phased-ranking.html#cross-hit-normalization-including-reciprocal-rank-fusion) - comparison between them can be found [in this blog post](https://blog.vespa.ai/embedding-tradeoffs-quantified)) will be interesting.
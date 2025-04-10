
<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Simple hybrid search with ColBERT

This semantic search application uses a single vector embedding model for retrieval and ColBERT (multi-token vector representation)
for re-ranking. The app demonstrates the [colbert-embedder](https://docs.vespa.ai/en/embedding.html#colbert-embedder) and
the tensor expressions for [ColBERT MaxSim](https://docs.vespa.ai/playground/#N4KABGBEBmkFxgNrgmUrWQPYAd5QGNIAaFDSPBdDTAF30gGJGwA1AUwGccBDMAYSwAbAEIBRAEoAVMAFkeADwDKASwC2YKewB2nLACcwYhTn1dOKrNrAAFITwCeAc31YArtoAmAHW2+pABYqnGA49s6uHp5g7Ao8amHsYE5uKp5cYFHs+py0PF4q2k5gWNBgtAFJHNx8tDp6hrGm5pa65QE8tGDqiWo6XRVJ8srqYBZqKvb6KrQOmZzs0YUCwuLSAHRgAOpJ6WpWufqdSYPd2nX6fZ4qx2N1OJy+hQBuws+LZ+1JBDxCBG72WitTibXy+JTsJK-PRfJAEYQAI2ytAAtOw1EjPOl9ABdAAUAVotAecAA9KTPFgCCD3jV1jwVKSdEyMYtrkV1oS1EJGPChEj9Kj0ZjsQBKMD5aKIPkCrqceKJCU4MIqH5Aqz4wnEzhk0lOGYBNwI9bwtSk2m8NFFQrsUnyhJCdgonjKzik2hmW1qHi5bKkmXI0WgvzaACE0sRyLA3oU401RJJ5KOAHd1vqKka3At9PDzv0TVgzbl8tADJ4UdA3LQ3GZPJ0eKTBKJJFJSd7ChSqW71DwnFx-ZHBaVoKqbkJ1jgiqKwdodmAOu9ysmsNGVAoPnVdAYQhVOmAzM0FudYU21jJ3gRaAZ9+xD-1OsCSmUAI5ubJzC9XnJgPEAKl-r7vniz60KK-7ipKYCUv8fTHp+24-v+nQeioCJVuweIBoKYG-tOIaBOwcwIlgFTzjwi6nPaSSFOk67RHi6S6Ow4rXLBFhWGA2jxEk-4KP+xASl4YDJkkWYnMuUHqPUj6loYCJmM8MwOMGvgomAAAGm4NMBtDAAAvgJCiIAATDiorqbCWFdIB+hzFp16zDg7CqRp9n6AxukGWARmmeZlmDl00FuLBAz1A5DhOWAM7GGE+QPhxsmwlxVzlGFhisdJBxRdoakAbQv4Sgi8nsIp8XWIlNl2dkGgub+nj5YVxWleq5UOZUUFUsF-SpZc2W5XxsI0bEHwMfUzGSWxrTCQaS4rhlujAjOABi175GcHpYJ4biXlNV5sFwvCpVuOQzi5iBaMd62uFtO0cSkaTsPG2q6tBNIHTw9KMsybkomJ+govd6ScrQ3J4Wp51paE0wTECi6Vtot26E9iadtS6wWh9DJMtopJmNA2Q6AQtpHNoADWhROGiJhmJw7G6MD3KMKY6gzCo7wVh4iOcGDSAXQ0nFWCizMw2zSTw1zyM6uSr3o+9n3Y7j7D42YCPE-k5NFFTh50yCXI8togvC6z7Piy13NnSIQhYE4CCCAkVYU9NpFuSEyzVLwku6giVtOLLdJY6aOAO5ryYGiiLuknh-jtaWQhW6HRRQUrNo7u1VlgJVgnREFIVHQ0IKQGQqB6UXBlF9Q5C4Aw7AkEXEAUPgFc0JA2gMJVOmirXNCYDXCCQG5AA80BW50AB8On6YZJlmXAwBgAADAgiDz8QACMOICavcCIKvxCmQJADM28ACzEAArDiemF93JcYGXGBN-XVd9zXpDdw3VB15grd98h0xoXUTCAVO5v27lAXuUBB7DywGPDyk9vLT1FLPReiAD7EGPhvMAW9EBn2IAANkwcZbeAB2YgAAOS+18aC32LqAiAj8oDP0IF3ZulA0Bf3rvQPuzAWA2GyLJCYidKTEmuttWgIQsDvEMJRDotYJqZWsAoZaKgchdBEknfYugPS3CaDTCwidTjCKhptMRIQkS0BEjoCUcdYSZwuBoeC35IK-CELCHO3V7FgEcSCbK0ckhG1hkkA8NN7wtSfLCWO8dHa6JaBxYIGl1LqVrNtDC7cQLigKn-VC6EgH8kDAJTgwVDLmUSTOQI8SYm0ymjTAE4jPh8G0OwZMedrylFhI5JImk0pwIMiBfSflQ7O3akY0wJjLxjHhDTMA5jLHWBcbYt8tkvHsEvAhZxNj3FwRWV+AuX8aEQHvqgBh2A2GQFfl-D+7CwFQB-lAYRoybriJYe-CBkBClqDxBnRZDgO5gEyUSf+OSrKimKVQu+pc6FXObkwyARBIX1zYQwzhDAeFgCWjRR2pwYwTIMGLVaNi7HVRCB3GcWwZrzMMSRB5piwA-D+ACY4ngBLqOTPkAYK4RxCSxYoHF0whGdVzo4gpK51HJKJiUKRsJ-z1X-PIhaVhfBZkdupGMFkfSwl7C4dgThOjXlNq0dYYLi4QvLhcmF5z36Io4TchgYr2AAH0YzPObq8u1DEqWiMvJwASMYBL1U7nsk1D8zWnLhRcq11z+4ouYGAJQwVwnYs4JMjIiV5mEsuI8bQBEwCct+HcW8mw5wLjxVEWEibJmO02aFXqSbcWhGyF898PU1CbGzbm1xvocDdB3Cud5XwFjlBXFgBEeRlinFPC2OQihVAaFrWYQ1ga76QuOea51CLG7Wpbgwd59rSj2pAmunu27gp4jtY6xQAab5BqOSGhgYbLUbsjVwqAMabA8COC4F0AQwCOneK44YM7fATukN26ZPoPgcT4ITTaDapAk04AIht+x0iuMGd+vgQgWZ1GiLmOoCg5TsEAqrH9OgnCkTxGfVexkQVjBXHtYJXBupWyKA23DsRxECSsEkRpiwe31v0AI2lhoyYU3WGUlcDGjxdF4B+o4OAAjeuEtRc4nqoScSadGF0Tls5SXldYX8OACocUpTBbqknQmtDgDORJXTjp4hwPA+q8CfJmRs6SmaFiVzemVBuNKin1E-GsAbLozG+yGDErCTgAQDChUugIvcYkljWFODwYii5YgKkdD4ttwgokGMhukTlrMOIdFTkkKiXjfhvhCIlGTvY5PfvnhKMrMQ4gOihGl9gAlBjWHUTwLEgkSKVEMHVz98mF1XqXaa9+q74VQAjWArdv8AXZMAVZe1o2Gvc0PfXV5UCR60HHg5ryTmvIuaQWgXwwAcBwBXlBWgt3DK3b0nANBV2bt3fqo97ycBV4vdPu977X3d4-fni93BgPPsPZBwoX7L28HEEh36h7xkntg7gKQpH924Co5+39uAZDEchmu795H33Yfo5XoDkHX27uw-x7vanZOYfPbgAATiJyTmn0OnsM8Z9oLnZPccU5e6jpn2Phdw4J74K+i7aHTehaGw9lykWQOjSweQ2g5gjM9XUspMdctYATsUJyAmDBqBazrsZXQ6X-EBFNRKMj30fHmnTP5v4+K-hUlmoIIQalCDqcsPgB8USu9aHm7z2mWmGDaacDprlunHeR8AGjfS9IDPJTYyl0nde8pTfi1xm2v2KbTd85tIR1muKreXib1Dr30NvS-ZXC3363MgPsMw9r7m684Dt8Bx6PmNtsr8-5KEAEYXW0X+T3NiAKEvXXqbwaZtK7myrzdz6mAxokIsFJJRTdlX11Y0VO-xUIjsu1ctBhK0Co8dVfjQ+qq9Ugqb0I776tfqNQc+vULK6nItawx9RbNvM9eqPvM5W1E-DCDvB1bva3RTX1e7efcFRfG9ZfO9ZvQA9+DfVFONDQPaTlaICldqADUYOdDIF-KfAIaOeJW3BlOoFrPtWPdqdNC3VtX3UDHrWEEg2dZNHNa8EqJtSg2vZA+XJfRXauDAz+SNYAyA-degObXbCAm6DCEA2gApIpDOUCT-MAfZHQ5dRvZhVfFvZuDfAiXrb4NaczY8EzLqY8X9dgDtXg9VblBQdQeNJg-xN-Mbb9Mg3ZSbUQ1A8QpvIwzA5uNvGMGwMA11SA09WQkCH1RQASHAJA41EQnEEAPSIAA)

It also features reciprocal rank fusion to fuse different rankings.


<p data-test="run-macro init-deploy colbert">
Requires at least Vespa 8.338.38
</p>

## To try this application

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started)
through the <code>vespa deploy</code> step, cloning `colbert` instead of `album-recommendation`.

Feed documents (this includes embed inference in Vespa):
<pre data-test="exec">
vespa feed dataset/*.json
</pre>

Example queries:
<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, q))'\
 'input.query(q)=embed(e5, @query)' \
 'input.query(qt)=embed(colbert, @query)' \
 'query=space contains many suns'
</pre>

<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, q))'\
 'input.query(q)=embed(e5, @query)' \
 'input.query(qt)=embed(colbert, @query)' \
 'query=shipping stuff over the sea'
 </pre>

 <pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, q))'\
 'input.query(q)=embed(e5, @query)' \
 'input.query(qt)=embed(colbert, @query)' \
 'query=exchanging information by sound'
 </pre>


### Export ColBERT models from HF
See the [model2onnx.py](model2onnx.py) script for exporting the ColBERT model from Hugging Face to ONNX format.

Notice that these three models use different embedding dimensionality.

Example usage:

#### https://huggingface.co/answerdotai/answerai-colbert-small-v1

This is the recommended colbert model for this application as it is optimized for speed and accuracy. See [blog post](https://blog.vespa.ai/introducing-answerai-colbert-small/)

```bash
python3 model2onnx.py --hf_model answerdotai/answerai-colbert-small-v1 --dims 96
```
Can be used with:
```
field colbert type tensor<int8>(dt{}, x[12])
```


#### https://huggingface.co/mixedbread-ai/mxbai-colbert-large-v1
```bash
python3 model2onnx.py --hf_model mixedbread-ai/mxbai-colbert-large-v1 --dims 128
```

### https://huggingface.co/vespa-engine/col-minilm
```bash
python3 model2onnx.py --hf_model vespa-engine/col-minilm --dims 32
```
Can be used with:
```
field colbert type tensor<int8>(dt{}, x[4])
```

### Terminate container

Remove the container after use:
<pre data-test="after">
$ docker rm -f vespa
</pre>

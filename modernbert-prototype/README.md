# ModernBERT Prototype

This is a prototype project to show how a sidecar-container can be used to generate embeddings for Vespa. 

Supports a large variety of embedding models, including ModernBERT, ColPali, JinaAI, Clip and more.

See https://github.com/michaelfeil/infinity for details. 

## Structure

- `pom.xml`: Maven build file.
- `services.xml`: Vespa application configuration.
- `com.example.my-embedder.def`: Config definition for the embedder.
- `MyEmbedder.java`: Java implementation of the embedder.

## Building and Deploying

1. Build the project:

   ```bash
   mvn package
   ```

2. Install docker-compose (if not installed)

   ```bash
   brew install docker-compose
   ```

3. Run docker-compose

   ```bash
   docker-compose up --build
   ```

4. Deploy the Vespa application:

   ```bash
   vespa deploy
   ```

5. Generate data:

   ```bash
   python datagen.py
   ```

6. Feed to generate embeddings inside Vespa:

   ```bash
   vespa feed data/inside/*.json
   ```

   Results:

   ```json
   {
      "feeder.operation.count": 1000,
      "feeder.seconds": 62.485,
      "feeder.ok.count": 1000,
      "feeder.ok.rate": 16.004,
      "feeder.error.count": 0,
      "feeder.inflight.count": 0,
      "http.request.count": 191222,
      "http.request.bytes": 67667135,
      "http.request.MBps": 1.083,
      "http.exception.count": 0,
      "http.response.count": 191222,
      "http.response.bytes": 24220744,
      "http.response.MBps": 0.388,
      "http.response.error.count": 190222,
      "http.response.latency.millis.min": 0,
      "http.response.latency.millis.avg": 159,
      "http.response.latency.millis.max": 59276,
      "http.response.code.counts": {
         "200": 1000,
         "429": 190222
      }
   }
   ```

7. Feed to generate embeddings in sidecar:

   ```bash
   vespa feed data/sidecar/*.json
   ```

   Results:

   ```json
   {
   "feeder.operation.count": 1000,
   "feeder.seconds": 37.654,
   "feeder.ok.count": 1000,
   "feeder.ok.rate": 26.558,
   "feeder.error.count": 0,
   "feeder.inflight.count": 0,
   "http.request.count": 1000,
   "http.request.bytes": 359075,
   "http.request.MBps": 0.010,
   "http.exception.count": 0,
   "http.response.count": 1000,
   "http.response.bytes": 68000,
   "http.response.MBps": 0.002,
   "http.response.error.count": 0,
   "http.response.latency.millis.min": 549,
   "http.response.latency.millis.avg": 18994,
   "http.response.latency.millis.max": 35922,
   "http.response.code.counts": {
      "200": 1000
   }
   }
   ```

8. Verify that embeddings are created

   ```bash
   vespa query "select * from doc where true limit 1"
   ```

### Podman

```bash
sudo mkdir -p cache
```

```bash
sudo podman run -d \
  --name modernbert \
  -p 10337:10337 \
  -v ./cache:/app/.cache \
  --network container:44cb11a0fe35 \
  michaelf34/infinity \
  v2 \
  --engine optimum \
  --model-id nomic-ai/modernbert-embed-base \
  --port 10337
```

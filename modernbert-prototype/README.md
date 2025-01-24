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

5. Feed to generate embeddings.

   ```bash
   vespa feed ext/*.json
   ```

6. Verify that embeddings are created

   ```bash
   vespa query "select * from doc where true"
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

<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Book Search — Vespa Testcontainers demo

A simple book search application demonstrating how to use [vespa-testcontainers](https://github.com/vespa-engine/vespa-testcontainers) for integration testing.

## Running the app locally

**1. Start Vespa**
```sh
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
```

**2. Deploy the application package**
```sh
vespa deploy app --wait 60
```

**3. Run the frontend**
```sh
cd frontend
uv run streamlit run app.py
```

Open the app at http://localhost:8501 and click **Feed documents** to load the dataset.

## Running the tests

The tests use `VespaContainer` to spin up an isolated Vespa instance automatically — no manual setup needed.

First, install `vespa-testcontainers` to your local Maven repository:
```sh
cd /path/to/vespa-testcontainers
./gradlew publishToMavenLocal
```

Then run the tests:
```sh
mvn test
```

> [!NOTE]
> When using Podman instead of Docker, you have to set
> ```bash
> export DOCKER_HOST="unix://"$(podman machine inspect --format {{.ConnectionInfo.PodmanSocket.Path}})
> export TESTCONTAINERS_RYUK_DISABLED=true
> ```

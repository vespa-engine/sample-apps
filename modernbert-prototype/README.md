# My Vespa Application

This project demonstrates how to create a custom Java embedder for Vespa.

## Structure

- `pom.xml`: Maven build file.
- `services.xml`: Vespa application configuration.
- `com.example.my-embedder.def`: Config definition for the embedder.
- `MyEmbedder.java`: Java implementation of the embedder.
- `vocab.txt`: Vocabulary file.

## Building and Deploying

1. Build the project:

   ```bash
   mvn package
   ```

2. Deploy the application:

   ```bash
   mvn vespa:deploy
   ```

3. Test the embedder by sending queries to your Vespa instance.

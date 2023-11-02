# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

FROM maven:3.9.5-eclipse-temurin-17
COPY src /home/app/src
COPY pom.xml /home/app
RUN mvn -f /home/app/pom.xml --batch-mode clean package

CMD ["java", "-jar", "/home/app/target/album-recommendation-random-data-1.0-SNAPSHOT-jar-with-dependencies.jar"]

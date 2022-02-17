# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

FROM maven:3.6.0-jdk-11 AS build
COPY src /home/app/src
COPY pom.xml /home/app
RUN mvn -f /home/app/pom.xml --batch-mode clean package

FROM openjdk:11

COPY --from=build /home/app/target/album-recommendation-random-data-1.0-SNAPSHOT-jar-with-dependencies.jar /usr/local/lib/app.jar

CMD ["java", "-jar", "/usr/local/lib/app.jar"]

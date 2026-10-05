plugins {
    application
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(libs.junit)

    implementation(libs.guava)

    implementation("io.github.hakky54:ayza-for-pem:10.1.0")
    implementation("org.eclipse.jetty:jetty-client:12.1.13")
    implementation("org.eclipse.jetty.http2:jetty-http2-client-transport:12.1.13")
    implementation("org.slf4j:slf4j-simple:2.0.20")
    implementation("commons-cli:commons-cli:1.11.0")
    implementation("com.yahoo.vespa:vespa-feed-client:8.763.13");
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

application {
    mainClass = "com.example.VespaClient"
}

tasks.named<JavaExec>("run") {
    workingDir = file(System.getProperty("user.dir"))
}

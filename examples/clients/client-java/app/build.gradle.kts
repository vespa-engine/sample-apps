plugins {
    application
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(libs.junit)

    implementation(libs.guava)

    implementation("io.github.hakky54:ayza-for-pem:10.0.3")
    implementation("com.squareup.okhttp3:okhttp:5.3.2")
    implementation("org.slf4j:slf4j-simple:2.0.17")
    implementation("commons-cli:commons-cli:1.11.0")
    implementation("com.yahoo.vespa:vespa-feed-client:8.643.19");
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

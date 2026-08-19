plugins {
    application
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(libs.junit)

    implementation(libs.guava)

    implementation("io.github.hakky54:ayza-for-pem:10.0.6")
    implementation("org.eclipse.jetty:jetty-client:12.1.12")
    implementation("org.eclipse.jetty.http2:jetty-http2-client-transport:12.1.12")
    implementation("org.slf4j:slf4j-simple:2.0.18")
    implementation("commons-cli:commons-cli:1.11.0")
    implementation("com.yahoo.vespa:vespa-feed-client:8.738.17");

    constraints {
        // vespa-feed-client 8.738.17 still pulls Bouncy Castle 1.84 (CVE-2026-8763
        // critical + 16 HIGH, fixed in 1.85); constrained until the platform ships 1.85+.
        implementation("org.bouncycastle:bcprov-jdk18on:1.85.2")
        implementation("org.bouncycastle:bcpkix-jdk18on:1.85")
        implementation("org.bouncycastle:bcutil-jdk18on:1.85")
    }
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

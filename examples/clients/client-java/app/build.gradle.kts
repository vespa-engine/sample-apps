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
    implementation("com.yahoo.vespa:vespa-feed-client:8.738.17");

    constraints {
        // vespa-feed-client 8.738.17 still pulls an older Bouncy Castle (CVE-2026-17508,
        // CVE-2026-71888, CVE-2026-71889 fixed in 1.86); constrained until the platform ships 1.86+.
        implementation("org.bouncycastle:bcprov-jdk18on:1.86")
        implementation("org.bouncycastle:bcpkix-jdk18on:1.86")
        implementation("org.bouncycastle:bcutil-jdk18on:1.86")
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

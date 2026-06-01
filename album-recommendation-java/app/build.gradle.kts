// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
plugins {
    `java-library`
    id("biz.aQute.bnd.builder") version "7.0.0"
}

group = "ai.vespa.examples"
version = "1.0.0"

val vespaVersion = "8.697.20"

repositories {
    mavenCentral()
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}

// Separate classpath for the Vespa config code generator (configgen).
val configgen by configurations.creating

dependencies {
    // Vespa container API — provided by the runtime, not bundled.
    compileOnly("com.yahoo.vespa:container:$vespaVersion")
    compileOnly("com.yahoo.vespa:container-dev:$vespaVersion")

    testImplementation("com.yahoo.vespa:container-test:$vespaVersion")
    testImplementation("com.yahoo.vespa:application:$vespaVersion")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")

    configgen("com.yahoo.vespa:configgen:$vespaVersion")
}

tasks.test {
    useJUnitPlatform()
}

// --- Config class generation ---------------------------------------------------
// Generates Java sources from .def files in src/main/resources/configdefinitions
// (e.g. MetalNamesConfig.java from metal-names.def).
val configDefDir = layout.projectDirectory.dir("src/main/resources/configdefinitions")
val generatedConfigDir = layout.buildDirectory.dir("generated/sources/configgen/main/java")

val generateConfig = tasks.register<JavaExec>("generateConfig") {
    val defFiles = fileTree(configDefDir) { include("*.def") }
    inputs.files(defFiles)
    outputs.dir(generatedConfigDir)

    classpath = configgen
    mainClass.set("com.yahoo.config.codegen.MakeConfig")

    doFirst {
        val dest = generatedConfigDir.get().asFile
        dest.mkdirs()
        systemProperty("config.dest", dest.absolutePath)
        systemProperty("config.spec", defFiles.files.joinToString(","))
    }
}

sourceSets.main {
    java.srcDir(generatedConfigDir)
}

tasks.named("compileJava") {
    dependsOn(generateConfig)
}

// --- OSGi bundle jar -----------------------------------------------------------
// The bnd builder plugin analyses the compiled bytecode and emits an
// Import-Package header with version ranges matching the OSGi metadata in the
// referenced bundles. The directives below pin the version ranges used by the
// Vespa bundle plugin for com.yahoo.* and com.google.inject packages.
tasks.jar {
    archiveBaseName.set("album-recommendation-java")
    archiveVersion.set("")
    archiveClassifier.set("deploy")

    bundle {
        bnd(
            """
            Bundle-SymbolicName: album-recommendation-java
            Bundle-Version: ${project.version}
            Bundle-Name: album-recommendation-java
            -nouses: true
            Import-Package: \
              com.yahoo.*;version="[1.0.0,2)",\
              com.google.inject;version="[1.4,2)",\
              *
            -removeheaders: Bundle-Vendor,Bundle-ClassPath,Private-Package,Tool,Bnd-LastModified,Created-By,Require-Capability,Provide-Capability
            """.trimIndent()
        )
    }
}

// --- application.zip -----------------------------------------------------------
val buildMeta = tasks.register("generateBuildMeta") {
    val outFile = layout.buildDirectory.file("vespa-build-meta/build-meta.json")
    outputs.file(outFile)
    inputs.property("vespaVersion", vespaVersion)
    doLast {
        outFile.get().asFile.parentFile.mkdirs()
        outFile.get().asFile.writeText(
            """
            {
              "compileVersion": "$vespaVersion",
              "buildTime": ${System.currentTimeMillis()},
              "parentVersion": "$vespaVersion"
            }
            """.trimIndent()
        )
    }
}

tasks.register<Zip>("applicationZip") {
    group = "build"
    description = "Builds the Vespa application.zip"
    archiveFileName.set("application.zip")
    destinationDirectory.set(layout.buildDirectory)

    // Application descriptor files (services.xml, schemas/, .vespaignore)
    from(layout.projectDirectory.dir("src/main/application"))

    // The bundle jar goes under components/
    from(tasks.jar) {
        into("components")
    }

    // build-meta.json at the archive root
    from(buildMeta)
}

tasks.named("assemble") {
    dependsOn("applicationZip")
}

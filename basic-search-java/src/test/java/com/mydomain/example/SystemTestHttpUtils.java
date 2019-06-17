package com.mydomain.example;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static java.net.http.HttpRequest.BodyPublishers.ofInputStream;

public class SystemTestHttpUtils {

    public static boolean postDocument(String docApiPut, String feed) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(docApiPut))
                .header("Content-Type", "application/json")
                .POST(ofInputStream(() -> SystemTestHttpUtils.class.getResourceAsStream(feed)))
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
    }

    public static String visitDocuments(String docApiVisit) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(docApiVisit))
                .GET()
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString()).body();
    }

    public static String searchDocumentsGET(String searchApiQuery) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(searchApiQuery))
                .GET()
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString()).body();
    }

    public static boolean removeDocument(String deleteRequest) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(deleteRequest))
                .DELETE()
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
    }
}

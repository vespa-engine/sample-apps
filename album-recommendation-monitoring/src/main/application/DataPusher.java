package application;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.TypeAdapterFactory;
import json.object.Album;
import json.object.TopLevel;
import org.apache.commons.io.IOUtils;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.ServiceLoader;

public class DataPusher {
    String baseCommand = "curl -H \"Content-Type:application/json\" --data-binary @%s http://localhost:8080/document/v1/mynamespace/music/docid/%d";
    private int albumCount = 4;
    private final Gson gson;
    Runtime rt = Runtime.getRuntime();

    DataPusher(){
        GsonBuilder gsonBuilder = new GsonBuilder();
        for (TypeAdapterFactory factory : ServiceLoader.load(TypeAdapterFactory.class)) {
            gsonBuilder.registerTypeAdapterFactory(factory);
        }
        gson = gsonBuilder.create();
    }

    public void pushThroughClass(TopLevel album) {
        try {
            String url = String.format("http://localhost:8080/document/v1/mynamespace/music/docid/%d", albumCount);
            HttpURLConnection con = (HttpURLConnection) new URL(url).openConnection();

            con.setRequestProperty("Content-Type", "application/json");

            con.setDoOutput(true);
            DataOutputStream wr = new DataOutputStream(con.getOutputStream());
            wr.writeBytes(gson.toJson(album));
            wr.flush();
            wr.close();

            int responseCode = con.getResponseCode();
            System.out.println("Response Code : " + responseCode);

            BufferedReader result = new BufferedReader(new InputStreamReader(con.getInputStream()));
            String output;
            StringBuilder response = new StringBuilder();
            while ((output = result.readLine()) != null) {
                response.append(output);
            }
            result.close();

            System.out.println(response.toString());
            albumCount++;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

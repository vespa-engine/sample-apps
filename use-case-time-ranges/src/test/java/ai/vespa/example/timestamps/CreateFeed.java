package ai.vespa.example.timestamps;

import java.util.*;
import java.io.FileWriter;
import java.io.IOException;

public class CreateFeed {
    private static long fromTs =  -31536000L;
    private static long lastTs = 2208988800L;
    private static Random random = new Random(42);
    private static String jsonFormat =
            """
            {
               "id": "id:gendoc:item::%d",
               "fields": {
                   "title": "Sample [%d] date: %s",
                   "timestamp_utc": %d,
                   "score": %f
               }
            }
            """;
    static String genDoc(int num) {
        long stamp = fromTs + 86400 * (long)(random.nextDouble() * (lastTs - fromTs) / 86400.0);
        String date = new Date(stamp * 1000L).toString();
        double score = random.nextDouble();
        return String.format(jsonFormat, num, num, date, stamp, score);
    }
    public static void main(String[] args) {
        try (FileWriter writer = new FileWriter("generated-data.json")) {
            for (int i = 1; i <= 20_000_000; i++) {
                writer.write(genDoc(i));
            }
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
            System.exit(1);
        }
    }
}

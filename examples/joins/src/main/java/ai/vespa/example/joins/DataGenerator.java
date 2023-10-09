// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.joins;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

public class DataGenerator {

    public static void main(String[] args) {
        int tags = -1, ids = -1, intervals = -1, seed = -1, firstTag = 0, firstId = 0, firstInterval = 0;
        try {
            if (args.length < 3)
                throw new IllegalArgumentException("Must provide at least 3 arguments");

            tags = Integer.parseInt(args[0]);
            ids = Integer.parseInt(args[1]);
            intervals = Integer.parseInt(args[2]);
            if (args.length > 3) seed = Integer.parseInt(args[3]);
            if (args.length > 4) firstTag = Integer.parseInt(args[4]);
            if (args.length > 5) firstId = Integer.parseInt(args[5]);
            if (args.length > 6) firstInterval = Integer.parseInt(args[6]);
        }
        catch (RuntimeException e) {
            e.printStackTrace();
            usage();
        }

        Random random = new Random(seed);
        for (int i = 0; i < ids; i++) {
            for (int j = 0; j < tags; j++) {
                for (int k = 0; k < intervals; k++) {
                    long t1 = (long) ((k + random.nextDouble()) * 1000),
                         t2 = (long) ((k + random.nextDouble()) * 1000);
                    if (i < firstId || j < firstTag || k < firstInterval) random.nextInt(1000);
                    else System.out.println(new TagDoc(Integer.toString(i),
                                                       Math.min(t1, t2),
                                                       Math.max(t1, t2),
                                                       j,
                                                       random.nextInt(1000))
                                                    .asJson());
                }
            }
        }
    }

    static void usage() {
        System.err.println("Usage: java DataGenerator.java <tags> <ids> <intervals> [seed] [first-tag] [first-id] [first-interval]");
        System.exit(1);
    }

    static abstract class Doc {

        final String id;
        final long start;
        final long end;

        Doc(String id, long start, long end) {
            this.id = id;
            this.start = start;
            this.end = end;
        }

        abstract String docId();

        Map<String, Object> fields() {
            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put("id", id);
            fields.put("start", start);
            fields.put("end", end);
            return fields;
        }

        String asJson() {
            return "{\n" +
                   "  \"id\": \"" + docId() + "\",\n" +
                   "  \"fields\": {\n" +
                   fields().entrySet().stream()
                           .map(field -> json(field.getKey()) + ": " + json(field.getValue()))
                           .collect(Collectors.joining(",\n    ", "    ", "\n")) +
                   "  }\n" +
                   "}";
        }

        static String json(Object raw) {
            return raw instanceof String ? "\"" + raw + "\"" : raw.toString();
        }

    }

    static class TagDoc extends Doc {

        final int tag;
        final int score;

        TagDoc(String id, long start, long end, int tag, int score) {
            super(id, start, end);
            this.tag = tag;
            this.score = score;
        }

        @Override
        String docId() {
            return "id:joins:tag::" + tag + "-" + id + "-" + start + "-" + end;
        }

        @Override
        Map<String, Object> fields() {
            Map<String, Object> fields = super.fields();
            fields.put("tagid", tag);
            fields.put("score", score);
            return fields;
        }

    }

}

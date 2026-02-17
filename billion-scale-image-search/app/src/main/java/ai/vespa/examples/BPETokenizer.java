// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.yahoo.component.AbstractComponent;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;

import javax.inject.Inject;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public class BPETokenizer extends AbstractComponent {

    private final Pattern pattern = Pattern.compile("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+", Pattern.CASE_INSENSITIVE);

    private final int contextLength;
    private final Map<Bigram, Integer> bigramRanks;
    private final Map<Integer, Character> byteEncoder;
    private final Map<Character, Integer> byteDecoder;
    private final Map<String, Integer> encoder = new HashMap<>();
    private final Map<Integer, String> decoder = new HashMap<>();
    private final Map<String, List<String>> cache = new HashMap<>();


    @Inject
    public BPETokenizer(ai.vespa.examples.BpeTokenizerConfig config) {
        contextLength = config.contextlength();
        List<Bigram> bigrams = loadBigrams(config.vocabulary());
        bigramRanks = getBigramRanks(bigrams);
        buildEncoderDecoder(bigrams);

        byteEncoder = bytesToUnicode();
        byteDecoder = byteEncoder.entrySet().stream().collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
    }

    public Tensor encode(String text) {
        return encode(text, contextLength, "d0");
    }

    public Tensor encode(String text, String dim) {
        return encode(text, contextLength, dim);
    }

    public Tensor encode(String text, int contextLength, String dim) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(encoder.get("<|startoftext|>"));

        Matcher matcher = pattern.matcher(clean(text).toLowerCase());
        while (matcher.find() && tokens.size() < contextLength) {
            String encodedWord = encodeBytes(matcher.group());
            for (String token : extractTokens(encodedWord)) {
                tokens.add(encoder.get(token));
            }
        }
        tokens.add(encoder.get("<|endoftext|>"));
        if (tokens.size() < contextLength) {
            IntStream.range(tokens.size(), contextLength).forEach(i -> tokens.add(0));
        } else {
            tokens.set(contextLength - 1, encoder.get("<|endoftext|>"));
        }
        return toTensor(tokens, dim);
    }

    public String decode(Tensor tensor) {
        StringBuilder sb = new StringBuilder();
        for (Iterator<Double> iter = tensor.valueIterator(); iter.hasNext();) {
            sb.append(decoder.get(iter.next().intValue()));
        }
        return decodeBytes(sb.toString()).replace("</w>", " ");
    }

    private Tensor toTensor(List<Integer> tokens, String dimName) {
        TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).indexed(dimName, tokens.size()).build();
        IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
        IntStream.range(0, tokens.size()).forEach(i -> builder.cell(tokens.get(i), i));
        return builder.build();
    }

    private void buildEncoderDecoder(List<Bigram> bigrams) {
        List<String> vocab = bytesToUnicode().values().stream().map(Object::toString).collect(Collectors.toList());
        vocab.addAll(vocab.stream().map(v -> v + "</w>").collect(Collectors.toList()));
        vocab.addAll(bigrams.stream().map(Bigram::merge).collect(Collectors.toList()));
        vocab.add("<|startoftext|>");
        vocab.add("<|endoftext|>");
        for (int i = 0; i < vocab.size(); ++i) {
            encoder.put(vocab.get(i), i);
            decoder.put(i, vocab.get(i));
        }
    }

    private Map<Bigram, Integer> getBigramRanks(List<Bigram> bigrams) {
        Map<Bigram, Integer> bigramRanks = new HashMap<>();
        IntStream.range(0, bigrams.size()).forEach(i -> bigramRanks.put(bigrams.get(i), i));
        return bigramRanks;
    }

    private List<Bigram> loadBigrams(Path file) {
        try {
            FileInputStream fis = new FileInputStream(file.toFile());  // get from config or something
            GZIPInputStream gzis = new GZIPInputStream(fis);
            InputStreamReader isr = new InputStreamReader(gzis, StandardCharsets.UTF_8);
            try (BufferedReader reader = new BufferedReader(isr)) {
                List<Bigram> bigrams = new ArrayList<>();
                String token = reader.readLine();  // skip first line
                for (int i = 0; ((token = reader.readLine()) != null) && i < 49152-256-2; ++i) {
                    bigrams.add(Bigram.of(token.split(" ")));
                }
                return bigrams;
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Vocab file not found", e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static String clean(String text) {
        return text.replaceAll("\\s+", " ").strip();
    }

    private static Map<Integer, Character> bytesToUnicode() {
        Map<Integer, Character> map = new LinkedHashMap<>();  // insertion order matters here for token encoder/decoder
        IntStream.rangeClosed('!', '~').forEach(i -> map.put(i, (char) i));
        IntStream.rangeClosed('¡', '¬').forEach(i -> map.put(i, (char) i));
        IntStream.rangeClosed('®', 'ÿ').forEach(i -> map.put(i, (char) i));
        int n = 0;
        for (int i = 0; i < 256; ++i) {
            if ( ! map.containsKey(i)) {
                map.put(i, (char)(256 + n++));
            }
        }
        return map;
    }

    private String encodeBytes(String text) {
        StringBuilder sb = new StringBuilder();
        for (byte b : text.getBytes(StandardCharsets.UTF_8)) {
            sb.append(byteEncoder.get((int)b & 0xff));  // signed to unsigned
        }
        return sb.toString();
    }

    private String decodeBytes(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        byte[] decoded = new byte[bytes.length];
        for (int i = 0; i < bytes.length; ++i) {
            decoded[i] = byteDecoder.get((char) bytes[i]).byteValue();
        }
        return new String(decoded, StandardCharsets.UTF_8);
    }

    private List<String> extractTokens(String token) {
        return cache.computeIfAbsent(token, t -> {
            List<String> tokenList = new ArrayList<>(token.length());
            for (int i = 0; i < token.length(); ++i) {
                tokenList.add(i, Character.toString(token.charAt(i)));
            }
            tokenList.set(tokenList.size()-1, tokenList.get(tokenList.size()-1) + "</w>");
            return extractTokens(tokenList);
        });
    }

    private List<String> extractTokens(List<String> tokenList) {
        Bigram bigram = findBestBigram(tokenList);  // e.g.  ["i", "ng"]
        if (bigram == null || ! bigramRanks.containsKey(bigram))
            return tokenList;

        List<String> newTokenList = new ArrayList<>();
        for (int i = 0; i < tokenList.size(); ++i) {
            if (bigram.at(tokenList, i)) {
                newTokenList.add(bigram.merge());  // e.g. "ing"
                i += 1;
            } else {
                newTokenList.add(tokenList.get(i));
            }
        }
        return newTokenList.size() == 1 ? newTokenList : extractTokens(newTokenList);
    }

    private Bigram findBestBigram(List<String> tokenList) {
        return bigrams(tokenList).stream().min((a, b) -> a.rankCompare(b, bigramRanks)).orElse(null);
    }

    private static Set<Bigram> bigrams(List<String> list) {
        Set<Bigram> bigrams = new HashSet<>();
        for (int i = 1; i < list.size(); ++i) {
            bigrams.add(Bigram.of(list.get(i-1), list.get(i)));
        }
        return bigrams;
    }


    public static class Bigram {
        private final String first;
        private final String second;

        public Bigram(String first, String second) {
            this.first = first;
            this.second = second;
        }

        public String first() { return first; }
        public String second() { return second; }
        public String merge() { return first + second; }

        public static Bigram of(String first, String second) {
            return new Bigram(first, second);
        }

        public static Bigram of(String[] arr) {
            return new Bigram(arr[0], arr[1]);
        }

        public boolean at(List<String> list, int start) {
            if (start < 0 || start >= list.size() - 1)
                return false;
            return list.get(start).equals(first) && list.get(start + 1).equals(second);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Bigram)) return false;
            Bigram p = (Bigram) o;
            return p.first.equals(first) && p.second.equals(second);
        }

        @Override
        public int hashCode() {
            return Objects.hash(first(), second());
        }

        @Override
        public String toString() {
            return "[" + first() + ", " + second() + "]";
        }

        public int rankCompare(Bigram other, Map<Bigram, Integer> ranks) {
            return ranks.getOrDefault(this, Integer.MAX_VALUE) - ranks.getOrDefault(other, Integer.MAX_VALUE);
        }

    }

}

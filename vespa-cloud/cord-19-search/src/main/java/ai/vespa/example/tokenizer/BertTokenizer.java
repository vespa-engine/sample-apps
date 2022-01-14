// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.tokenizer;

import com.google.inject.Inject;
import com.yahoo.collections.Tuple2;
import com.yahoo.component.AbstractComponent;
import com.yahoo.language.Language;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.process.Tokenizer;
import com.yahoo.language.simple.SimpleLinguistics;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;
import java.util.logging.Logger;

/**
 * Adapted from
 * https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/tokenization/tokenizer/BertWordPieceTokenizer.java
 *
 * @author jobergum
 */
public class BertTokenizer extends AbstractComponent {

    private final Logger logger = Logger.getLogger(BertTokenizer.class.getName());
    private final NavigableMap<String, Integer> vocabulary;
    private final Tokenizer tokenizer;

    @Inject
    public BertTokenizer(BertModelConfig config) throws IOException {
        super();
        Path path = config.vocabulary();
        this.tokenizer = new SimpleLinguistics().getTokenizer();
        logger.info("Loading vocabulary from " + path.toString());
        this.vocabulary = new TreeMap<>(Collections.reverseOrder());
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(path.toFile()), StandardCharsets.UTF_8))) {
            String token;
            int i = 0;
            while ((token = reader.readLine()) != null) {
                this.vocabulary.put(token, i);
                i++;
            }
            logger.info("Loaded " + i + " tokens from vocabulary file");
        }
    }

   /**
    * Tokenizes the input string and return a list of token ids
    *
    * @param input The string input to tokenize and map to bert token ids
    * @param maxLength The max length
    * @param padToMaxLength If true padd with 0s up to the max sequence lenght
    * @return List of token_ids
    */
   public List<Integer> tokenize(String input,  int maxLength, boolean padToMaxLength) {
        List<Integer> tensor = new ArrayList<>();
        input = input.toLowerCase();
        for(Token t: this.tokenizer.tokenize(input, Language.ENGLISH, StemMode.NONE, true)) {
            String originalToken = t.getTokenString();
            String candidate = originalToken;
            int count = 0;
            while(candidate.length() > 0 && ! "##".equals(candidate)){
                Tuple2<String,Integer> entry = findLongestSubstring(candidate);
                if (entry == null)
                    break;
                if(tensor.size() < maxLength) {
                    tensor.add(entry.second);
                }
                candidate = "##" +candidate.substring(entry.first.length());
                if (count++ > originalToken.length())
                    break;
            }
        }

        if (padToMaxLength) {
            for (int i = tensor.size(); i < maxLength; i++)
                tensor.add(0);
        }
        return tensor;
    }

    protected Tuple2<String,Integer> findLongestSubstring(String candidate) {
        NavigableMap<String, Integer> tailMap = this.vocabulary.tailMap(candidate, true);
        if (tailMap.isEmpty())
            return null;
        String longestSubstring = tailMap.firstKey();
        Integer id = tailMap.firstEntry().getValue();
        int subStringLength = Math.min(candidate.length(), longestSubstring.length());
        while(!candidate.startsWith(longestSubstring)){
            subStringLength--;
            tailMap = tailMap.tailMap(candidate.substring(0, subStringLength), true);
            if (tailMap.isEmpty())
                return null;
            longestSubstring = tailMap.firstKey();
            id = tailMap.firstEntry().getValue();
        }
        return new Tuple2<>(longestSubstring,id);
    }

}

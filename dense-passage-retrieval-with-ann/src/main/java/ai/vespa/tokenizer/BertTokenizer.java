package ai.vespa.tokenizer;
import com.google.inject.Inject;
import com.yahoo.collections.Tuple2;
import com.yahoo.component.AbstractComponent;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.*;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Adopted from
 * https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nlp-parent/deeplearning4j-nlp/src/main/java/org/deeplearning4j/text/tokenization/tokenizer/BertWordPieceTokenizer.java
 */

public class BertTokenizer extends AbstractComponent {

    private final Logger logger = Logger.getLogger(BertTokenizer.class.getName());
    private int max_length;

    private NavigableMap<String, Integer> vocabulary;
    private Map<Integer,String> tokenId2Token;

    static final Pattern splitPattern = Pattern.compile("\\p{javaWhitespace}+|((?<=\\p{Punct})+|(?=\\p{Punct}+))");

    @Inject
    public BertTokenizer(BertModelConfig config) throws IOException {
        super();
        Path path = config.vocabulary();
        max_length = config.max_input();

        logger.info("Loading vocabulary from " + path.toString());
        this.vocabulary = new TreeMap<>(Collections.reverseOrder());
        this.tokenId2Token = new HashMap<>();
        try (final BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(path.toFile()), Charset.forName("UTF-8")))) {
            String token;
            int i = 0;
            while ((token = reader.readLine()) != null) {
                this.vocabulary.put(token, i);
                this.tokenId2Token.put(i,token);
                i++;
            }
            logger.info("Loaded " + i + " tokens from vocabulary file");
        }
    }

    public int getMaxLength() {
        return max_length;
    }


   public List<Integer>tokenize(String input)  {
        return tokenize(input, false);
   }

   /**
    *
    * Tokenize the input string and return a list of token ids
    * @param input The string input to tokenize and map to bert token ids
    * @param padToMax If true padd with 0s up to the max sequence lenght
    * @return List of token_ids
    */

   public List<Integer> tokenize(String input,  boolean padToMax) {
        List<Integer> tensor = new ArrayList<>();
        input = input.toLowerCase();
        String[] splitTokens = splitPattern.split(input);

        int nTokens = 0;
        for(String originalToken: splitTokens) {
            String candidate = originalToken;
            int count = 0;
            while(candidate.length() > 0 && !"##".equals(candidate)){
                Tuple2<String,Integer> e = findLongestSubstring(candidate);
                if (e == null)
                    break;
                if(nTokens < max_length) {
                    tensor.add(e.second);
                    nTokens++;
                }
                candidate = "##" +candidate.substring(e.first.length());
                if (count++ > originalToken.length())
                    break;
            }
        }

        if (padToMax) {
            for (int i = nTokens; i < max_length; i++)
                tensor.add(0);
        }

        return tensor;
    }

    /**
     *
     * @param tokenIds The tokens ids to map back to string
     * @return
     */

    public List<String> convert2Tokens(List<Integer> tokenIds) {
        List<String> tokens = new ArrayList<>();
        for(Integer token_id : tokenIds)
            tokens.add(this.tokenId2Token.get(token_id));
        return tokens;
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
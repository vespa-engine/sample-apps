package ai.vespa.example.searchsuggest;


import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.FieldValue;


import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.logging.Logger;

public class QueryDocumentProcessor extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(QueryDocumentProcessor.class.getName());

    private static final String QUERY_DOCUMENT_TYPE  = "query";
    private static final String filePath = "files/accepted_words.txt";
    private final List<String> blockWords;
    private final HashSet<String> acceptedWords;



    @Inject
    public QueryDocumentProcessor(BlocklistConfig config){
        this.blockWords = config.blocklist();
        this.acceptedWords = getAcceptedWords();
    }

    public QueryDocumentProcessor(){
        //default constructor typically used for tests
        this.blockWords = new ArrayList<>();
        this.acceptedWords = getAcceptedWords();
    }

    @Override
    public Progress process(Processing processing) {
        logger.info("In process");
        for (DocumentOperation op : processing.getDocumentOperations()){
            //if op is DocumentPut
            if (op instanceof DocumentPut){
                //gets the document
                DocumentPut put = (DocumentPut) op;
                Document document = put.getDocument();
                if (document.getDataType().isA(QUERY_DOCUMENT_TYPE)){
                    //checking if query contains anny of the blocked words
                    boolean containsBlockWords = checkForBlockWords(document);
                    boolean allWordsAccepted = checkAllWordsAccepted(document);
                    if (containsBlockWords || !allWordsAccepted){
                        logger.info("not all words are accepted");
                        processing.getDocumentOperations().clear();
                        return Progress.DONE;
                    }
                }


            }
        }
        logger.info("  found no blocking words");
        return Progress.DONE;
    }

    private Boolean checkForBlockWords(Document doc) {
        logger.info("  Searching through document");
        FieldValue inputValue = doc.getFieldValue("input");
        String query = inputValue.toString().toLowerCase();
        String[] terms = query.split("\\s+");
        for (String blockWord : blockWords){
            for (String term : terms){
                //goes through and checks if query contains any of the blockWords
                if (Objects.equals(blockWord.toLowerCase(), term)){
                    logger.info("  found blocking word");
                    return true;
                }
            }
        }
        return false;
    }

    private Boolean checkAllWordsAccepted(Document doc){
        logger.info("  Checking if all words are accepted");
        if (!acceptedWords.isEmpty()){
            FieldValue inputValue = doc.getFieldValue("input");
            String query = inputValue.toString().toLowerCase();
            String[] terms = query.split("\\s+");
            for (String term : terms) {
                if (!acceptedWords.contains(term)) {
                    return false;
                }
            }
        }
        return true;
    }

    private HashSet<String> getAcceptedWords(){
        logger.info("getting set of accepted words");
        HashSet<String> acceptedWords = new HashSet<String>();
        if (resourceExists()){
            try{
                ClassLoader cl = getClass().getClassLoader();
                InputStream is = cl.getResourceAsStream(filePath);
                InputStreamReader isr = new InputStreamReader(is, StandardCharsets.UTF_8);
                BufferedReader br = new BufferedReader(isr);
                String term;
                while ((term = br.readLine()) != null){
                    acceptedWords.add(term);
                }
            }catch (IOException e){
                e.printStackTrace();
            }
        }
        return acceptedWords;
    }

    private boolean resourceExists() {
        return getClass().getClassLoader().getResource(QueryDocumentProcessor.filePath) != null;
    }

    @Override
    public void deconstruct() {
        super.deconstruct();
    }
}

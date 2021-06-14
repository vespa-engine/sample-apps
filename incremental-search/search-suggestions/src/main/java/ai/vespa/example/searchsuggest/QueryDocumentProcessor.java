package ai.vespa.example.searchsuggest;


import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.documentapi.*;


import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class QueryDocumentProcessor extends DocumentProcessor {
    private static final Logger logger = Logger.getLogger(QueryDocumentProcessor.class.getName());

    private static final String QUERY_DOCUMENT_TYPE  = "query";
    private final List<String> blockWords;




    @Inject
    public QueryDocumentProcessor(BlocklistConfig config){
        this.blockWords = config.blocklist();
    }

    public QueryDocumentProcessor(){
        //default constructor typically used for tests
        this.blockWords = new ArrayList<>();
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
                    if (containsBlockWords){
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
        for (String blockWord : blockWords){
            //goes through and checks if query contains any of the blockWords
            if (query.contains(blockWord.toLowerCase())){
                logger.info("  found blocking word");
                return true;
            }
        }
        return false;
    }

    @Override
    public void deconstruct() {
        super.deconstruct();
        asyncSession.destroy();
    }
}

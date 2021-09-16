// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.language.simple.SimpleLinguistics;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.fail;

public class TokenizerFactory {

    static BertTokenizer getTokenizer() {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(
                new com.yahoo.config.FileReference("src/main/application/files/bert-base-uncased-vocab.txt"))
                .max_input(512);
        BertModelConfig bertModelConfig = builder.build();
        try {
            return new BertTokenizer(bertModelConfig, new SimpleLinguistics());
        } catch (IOException e) {
            fail("IO Error during bert model read");
        }
        return null;
    }
}

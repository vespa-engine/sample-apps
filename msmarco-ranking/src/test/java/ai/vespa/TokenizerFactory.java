// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa;

import com.yahoo.config.FileReference;
import com.yahoo.language.wordpiece.WordPieceConfig;
import com.yahoo.language.wordpiece.WordPieceEmbedder;

public class TokenizerFactory {

    public static WordPieceEmbedder getEmbedder() {
        WordPieceConfig.Builder b = new WordPieceConfig.Builder().model(
                new WordPieceConfig.Model.Builder()
                .language("unknown")
                .path(new FileReference("src/main/application/files/bert-base-uncased-vocab.txt")));
        return new WordPieceEmbedder(b.build());
    }
}

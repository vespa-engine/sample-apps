// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.yahoo.docproc.SimpleDocumentProcessor;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentPut;

public class DemoDocumentProcessor extends SimpleDocumentProcessor {

  @Override
  public void process(DocumentPut put) {
    // This dummy processor, in only a "Title Setter"
    super.process(put);
    Document document = put.getDocument();
    // ... Imagine something complicated happened here to compute the Title
    String title = "A simple title";
    document.setFieldValue("title", title);
  }
}

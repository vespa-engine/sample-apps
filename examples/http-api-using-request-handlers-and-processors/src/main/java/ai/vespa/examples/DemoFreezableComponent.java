// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.google.inject.Inject;
import com.yahoo.component.provider.FreezableSimpleComponent;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.documentapi.DocumentAccess;
import com.yahoo.documentapi.SyncParameters;
import com.yahoo.documentapi.SyncSession;

public class DemoFreezableComponent extends FreezableSimpleComponent {

  private final SyncSession session;

  @Inject
  public DemoFreezableComponent(DocumentAccess acc) {
    this.session = acc.createSyncSession(new SyncParameters.Builder().build());
  }

  public void syncProcess(Processing processing) {
    for (DocumentOperation docOpp: processing.getDocumentOperations()) {
      if (docOpp instanceof DocumentPut) {
        session.put((DocumentPut) docOpp);
      }
    }
  }
}

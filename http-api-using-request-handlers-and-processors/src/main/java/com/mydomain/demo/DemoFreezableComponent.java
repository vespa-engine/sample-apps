package com.mydomain.demo;

import com.yahoo.component.provider.FreezableSimpleComponent;
import com.yahoo.docproc.Processing;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.documentapi.DocumentAccess;
import com.yahoo.documentapi.SyncParameters;
import com.yahoo.documentapi.SyncSession;

public class DemoFreezableComponent extends FreezableSimpleComponent {

  private final DocumentAccess access = DocumentAccess.createDefault();
  private final SyncSession session = access.createSyncSession(new SyncParameters());

  public DemoFreezableComponent() {
  }

  public void syncProcess(Processing processing) {
    for (DocumentOperation docOpp: processing.getDocumentOperations()) {
      if (docOpp instanceof DocumentPut) {
        session.put((DocumentPut) docOpp);
      }
    }
  }
}

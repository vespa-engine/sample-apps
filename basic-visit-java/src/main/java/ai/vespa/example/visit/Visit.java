package ai.vespa.example.visit;

import com.yahoo.document.Document;
import com.yahoo.document.DocumentId;
import com.yahoo.documentapi.*;
import com.yahoo.log.LogSetup;
import com.yahoo.messagebus.Message;

import java.util.concurrent.TimeoutException;

public class Visit {

    public static void main(String[] args) throws Exception {
        LogSetup.initVespaLogging("visit-example");

        VisitorParameters params = new VisitorParameters("true");
        params.setVisitInconsistentBuckets(true);
        params.setLocalDataHandler(new DumpVisitorDataHandler() {

            @Override
            public void onDocument(Document doc, long timeStamp) {
                System.out.print(doc.toXML(""));
            }

            @Override
            public void onRemove(DocumentId id) {
                System.out.println("id=" + id);
            }

            @Override
            public void onMessage(Message m, AckToken token) { System.out.println("message=" + m);}
        });
        params.setControlHandler(new VisitorControlHandler() {

            @Override
            public void onProgress(ProgressToken token) {
                System.err.format("%.1f %% finished.\n", token.percentFinished());
                super.onProgress(token);
            }

            @Override
            public void onDone(CompletionCode code, String message) {
                System.err.println("Completed visitation, code " + code + ": " + message);
                super.onDone(code, message);
            }
        });
        params.setRoute(args.length > 0 ? args[0] : "[Storage:cluster=music;clusterconfigid=music]");
        params.setFieldSet(args.length > 1 ? args[1] : "[all]");

        DocumentAccess access = DocumentAccess.createDefault();
        VisitorSession session = access.createVisitorSession(params);
        if (!session.waitUntilDone(0)) {
            throw new TimeoutException();
        }
        session.destroy();
        access.shutdown();
        System.exit(0);
    }
}
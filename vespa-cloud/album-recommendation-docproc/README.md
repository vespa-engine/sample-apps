<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - album recommendations docproc

This sample application explores the Vespa data feeding - items:
- Document Processors: modify / enrich data in the feed pipeline
- Multiple Schemas: store different kinds of data, like different database tables
- Enrich data from multiple sources: here, look up data in one schema and add to another
- Document API: write asynchronous code to fetch data


Follow steps 1-8 in https://cloud.vespa.ai/getting-started-custom-code,
  stop after setting $ENDPOINT

Feed a lyrics document
```
   curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
     -H "Content-Type:application/json" --data-binary @src/test/resources/A-Head-Full-of-Dreams-lyrics.json \
    $ENDPOINT/document/v1/mynamespace/lyrics/docid/1
```

Get the document to validate - dump all docs in lyrics schema:
```
curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
  "$ENDPOINT/document/v1/mynamespace/lyrics/docid?wantedDocumentCount=100"
```

Feed a music document
```
curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
  -H "Content-Type:application/json" --data-binary @src/test/resources/A-Head-Full-of-Dreams.json \
  $ENDPOINT/document/v1/mynamespace/music/docid/1
```

Get the document to validate - dump all docs in music schema:
```
curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
  "$ENDPOINT/document/v1/mynamespace/music/docid?wantedDocumentCount=100"
```


Use the https://console.vespa.oath.cloud to dump logs, to inspect what happened

    Container.ai.vespa.example.album.LyricsDocumentProcessor	info	In process
    Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Added to requests pending: 1
    Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Request pending ID: 1, Progress.LATER
    Container.ai.vespa.example.album.LyricsDocumentProcessor	info	In handleResponse
    Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  requestID: 1
    Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Found lyrics for : document
    Container.ai.vespa.example.album.LyricsDocumentProcessor	info	In process
    Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Set lyrics, Progress.DONE

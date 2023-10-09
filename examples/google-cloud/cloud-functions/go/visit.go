// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// vespa visit examples
// Author: arnej, kkraune

package vespasamples

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/storage"
	"github.com/GoogleCloudPlatform/functions-framework-go/functions"
)

func init() {
	functions.HTTP("visit", visit)
	functions.HTTP("backup", backup)
}

func ReaderToJSON(reader io.Reader) string {
	bodyBytes, _ := io.ReadAll(reader)
	var prettyJSON bytes.Buffer
	parseError := json.Indent(&prettyJSON, bodyBytes, "", "    ")
	if parseError != nil { // Not JSON: Print plainly
		return string(bodyBytes)
	}
	return prettyJSON.String()
}

// Service represents a Vespa service.
type Service struct {
	BaseURL string
	Name    string
	Cert    tls.Certificate
}

type OperationResult struct {
	Success bool
	Message string // Mandatory message
	Detail  string // Optional detail message
	Payload string // Optional payload - may be present whether or not the operation was success
}

func Success(message string) OperationResult {
	return OperationResult{Success: true, Message: message}
}

func Failure(message string) OperationResult {
	return OperationResult{Success: false, Message: message}
}

func FailureWithPayload(message string, payload string) OperationResult {
	return OperationResult{Success: false, Message: message, Payload: payload}
}

var backupBucket string
var backupFolder = "backup-" + time.Now().Format("20060102150405") + "/"
var chunkIndex = 0

var d struct {
	ContentCluster string `json:"contentCluster"`
	Selection      string `json:"selection"`
	ChunkCount     int    `json:"chunkCount"`
	Endpoint       string `json:"endpoint"`
	Bucket         string `json:"bucket"`
	JsonLines      bool   `json:"jsonLines"`
}

func backup(w http.ResponseWriter, r *http.Request) {
	if err := json.NewDecoder(r.Body).Decode(&d); err != nil {
		fmt.Fprint(w, "Backup: Error decoding arguments")
		return
	}
	if d.Bucket == "" {
		fmt.Fprint(w, "Backup: bucket missing")
		return
	}
	backupBucket = d.Bucket
	fmt.Fprintf(w, "Running backup using visit, using bucket %s\n", backupBucket)
	log.Printf("Running backup using visit, using bucket %s", backupBucket)

	visit(w, r)
}

func visit(w http.ResponseWriter, r *http.Request) {
	if d.Endpoint == "" {
		// Arguments are not read yet - visit can be called from the backup function or directly
		if err := json.NewDecoder(r.Body).Decode(&d); err != nil {
			fmt.Fprint(w, "Visit: Error decoding arguments")
			return
		}
	}
	if d.Endpoint == "" {
		fmt.Fprint(w, "Visit: endpoint missing")
		return
	}
	if d.ContentCluster == "" {
		fmt.Fprint(w, "Visit: contentCluster missing")
		return
	}

	intro := fmt.Sprintf("Running visit with arguments: %s, %s, %s, %d, %s, %t\n",
		d.Endpoint,
		d.ContentCluster,
		d.Selection,
		d.ChunkCount,
		d.Bucket,
		d.JsonLines)
	fmt.Fprintf(w, intro)
	log.Printf(intro)

	var vArgs visitArgs
	vArgs.contentCluster = d.ContentCluster
	if d.Selection != "" {
		vArgs.selection = d.Selection
	}
	if d.ChunkCount != 0 {
		vArgs.chunkCount = d.ChunkCount
	}
	if d.JsonLines {
		vArgs.jsonLines = d.JsonLines
	}

	certPem := []byte(os.Getenv("SEC_CERT"))
	keyPem := []byte(os.Getenv("SEC_KEY"))
	cert, err := tls.X509KeyPair(certPem, keyPem)
	if err != nil {
		fmt.Fprintf(w, "Cert error: %v", err)
		return
	}

	var service Service
	service.BaseURL = d.Endpoint //"https://vespacloud-docsearch.vespa-team.aws-eu-west-1a.z.vespa-app.cloud"
	service.Cert = cert

	res := visitClusters(vArgs, &service)

	fmt.Fprintf(w, "Msg: %s\n", res.Message)
}

type visitArgs struct {
	contentCluster string
	fieldSet       string
	selection      string
	makeFeed       bool
	jsonLines      bool
	quietMode      bool
	chunkCount     int
}

var totalDocCount int

func visitClusters(vArgs visitArgs, service *Service) (res OperationResult) {
	clusters := []string{
		vArgs.contentCluster,
	}
	if vArgs.contentCluster == "*" {
		clusters = probeVisit(vArgs, service)
	}
	if vArgs.makeFeed {
		fmt.Printf("[")
	}
	for _, c := range clusters {
		vArgs.contentCluster = c
		res = runVisit(vArgs, service)
		if !res.Success {
			return res
		}
	}
	if vArgs.makeFeed {
		fmt.Println("{}\n]")
	}
	return res
}

func probeVisit(vArgs visitArgs, service *Service) []string {
	clusters := make([]string, 0, 3)
	vvo, _ := runOneVisit(vArgs, service, "")
	if vvo != nil {
		msg := vvo.ErrorMsg
		if strings.Contains(msg, "no content cluster '*'") {
			for idx, value := range strings.Split(msg, ",") {
				if idx > 0 {
					parts := strings.Split(value, "'")
					if len(parts) == 3 {
						clusters = append(clusters, parts[1])
					}
				}
			}
		}
	}
	return clusters
}

func runVisit(vArgs visitArgs, service *Service) (res OperationResult) {
	var totalDocuments int = 0
	var continuationToken string
	for {
		var vvo *VespaVisitOutput
		vvo, res = runOneVisit(vArgs, service, continuationToken)
		if !res.Success {
			return res
		}
		if vArgs.makeFeed {
			dumpDocuments(vvo.Documents, true, false)
		} else if vArgs.jsonLines {
			dumpDocuments(vvo.Documents, false, false)
		}
		//if !vArgs.quietMode {
		//  fmt.Fprintln(os.Stderr, "got", len(vvo.Documents), "documents")
		//}
		totalDocuments += len(vvo.Documents)
		continuationToken = vvo.Continuation
		if continuationToken == "" {
			break
		}
	}
	res.Message = fmt.Sprintf("%s [%d documents visited]", res.Message, totalDocuments)
	return
}

func runOneVisit(vArgs visitArgs, service *Service, contToken string) (*VespaVisitOutput, OperationResult) {
	urlPath := service.BaseURL + "/document/v1/?cluster=" + vArgs.contentCluster
	if vArgs.fieldSet != "" {
		urlPath = urlPath + "&fieldSet=" + vArgs.fieldSet
	}
	if vArgs.selection != "" {
		urlPath = urlPath + "&selection=" + vArgs.selection
	}
	if contToken != "" {
		urlPath = urlPath + "&continuation=" + contToken
	}
	if vArgs.chunkCount > 0 {
		urlPath = urlPath + fmt.Sprintf("&wantedDocumentCount=%d", vArgs.chunkCount)
	}
	theUrl, urlParseError := url.Parse(urlPath)
	if urlParseError != nil {
		return nil, Failure("Invalid request path: '" + urlPath + "': " + urlParseError.Error())
	}
	request := &http.Request{
		URL:    theUrl,
		Method: "GET",
	}

	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				Certificates:       []tls.Certificate{service.Cert},
				MinVersion:         tls.VersionTLS12,
				InsecureSkipVerify: true,
			},
		},
		Timeout: time.Duration(900) * time.Second,
	}

	response, err := client.Do(request)
	if err != nil {
		return nil, Failure("Request failed: " + err.Error())
	}
	defer response.Body.Close()
	vvo, err := parseVisitOutput(response.Body)
	if response.StatusCode == 200 {
		if err == nil {
			totalDocCount += vvo.DocumentCount
			if vvo.DocumentCount != len(vvo.Documents) {
				fmt.Fprintln(os.Stderr, "Inconsistent contents from:", theUrl)
				fmt.Fprintln(os.Stderr, "claimed count: ", vvo.DocumentCount)
				fmt.Fprintln(os.Stderr, "document blobs: ", len(vvo.Documents))
				// return nil, util.Failure("Inconsistent contents from document API")
			}
			return vvo, Success("visited " + vArgs.contentCluster)
		} else {
			return nil, Failure("error reading response: " + err.Error())
		}
	} else if response.StatusCode/100 == 4 {
		return vvo, FailureWithPayload("Invalid document operation: "+response.Status, ReaderToJSON(response.Body))
	} else {
		return vvo, FailureWithPayload("At "+request.URL.Host+": "+response.Status, ReaderToJSON(response.Body))
	}
}

type DocumentBlob struct {
	blob []byte
}

func (d *DocumentBlob) UnmarshalJSON(data []byte) error {
	d.blob = make([]byte, len(data))
	copy(d.blob, data)
	return nil
}

func (d *DocumentBlob) MarshalJSON() ([]byte, error) {
	return d.blob, nil
}

type VespaVisitOutput struct {
	PathId        string         `json:"pathId"`
	Documents     []DocumentBlob `json:"documents"`
	DocumentCount int            `json:"documentCount"`
	Continuation  string         `json:"continuation"`
	ErrorMsg      string         `json:"message"`
}

func parseVisitOutput(r io.Reader) (*VespaVisitOutput, error) {
	codec := json.NewDecoder(r)
	var parsedJson VespaVisitOutput
	err := codec.Decode(&parsedJson)
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not decode JSON, error:", err)
		return nil, err
	}
	return &parsedJson, nil
}

func dumpDocuments(documents []DocumentBlob, comma, pretty bool) {
	if backupBucket == "" {
		return
	}
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		log.Printf("storage.NewClient: %v", err)
		return
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(ctx, time.Second*10)
	defer cancel()

	obj := client.Bucket(backupBucket).Object(backupFolder + strconv.Itoa(chunkIndex))
	wr := obj.NewWriter(ctx)
	defer wr.Close()

	for _, value := range documents {
		_, err := wr.Write(value.blob)
		_, err2 := wr.Write([]byte("\n"))
		if err != nil || err2 != nil {
			log.Printf("NewWriter.Write: %v", err)
			return
		}
	}
	chunkIndex += 1
}

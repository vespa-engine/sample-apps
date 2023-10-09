// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// Misc examples
// Author: kkraune

package vespasamples

import (
	"cloud.google.com/go/storage"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"github.com/GoogleCloudPlatform/functions-framework-go/functions"
	"html"
	"io"
	"net/http"
	"os"
	"time"
)

func init() {
	functions.HTTP("helloVespaWorld", helloVespaWorld)
	functions.HTTP("getPage", getPageSimple)
	functions.HTTP("storePage", storePage)
}

func helloVespaWorld(w http.ResponseWriter, r *http.Request) {
	var d struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&d); err != nil {
		fmt.Fprint(w, "Hello, World!")
		return
	}
	if d.Name == "" {
		fmt.Fprint(w, "Hello, World!")
		return
	}
	fmt.Fprintf(w, "Hello, %s!", html.EscapeString(d.Name))
}

func getPageSimple(w http.ResponseWriter, r *http.Request) {
	url, _, _ := getParams(w, r)
	if url == "" {
		fmt.Fprint(w, "Input error - url missing")
		return
	}
	response, err := http.Get(url)
	if err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, "Request error: %v", err)
		return
	}
	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "Body read error: %v", err)
		return
	}
	fmt.Fprint(w, string(body))
}

func storePage(w http.ResponseWriter, r *http.Request) {
	url, bucket, object := getParams(w, r)
	if url == "" || bucket == "" || object == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, "Input error - url or bucket or object missing")
		return
	}
	bytes := getPageTLS(w, url)
	err := writeDocument(bucket, object, bytes)
	if err != nil {
		fmt.Fprintf(w, "Store error: %v", err)
		return
	}
}

func getPageTLS(w http.ResponseWriter, url string) []byte {
	// The credentials are stored as Secrets in Google Cloud, exposed as environment variables
	certPem := []byte(os.Getenv("SEC_CERT"))
	keyPem := []byte(os.Getenv("SEC_KEY"))
	cert, err := tls.X509KeyPair(certPem, keyPem)
	if err != nil {
		fmt.Fprintf(w, "Cert error: %v", err)
		return nil
	}
	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				Certificates:       []tls.Certificate{cert},
				InsecureSkipVerify: true,
			},
		},
	}

	response, err := client.Get(url)
	if err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, "Request error: %v", err)
		return nil
	}
	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "Body read error: %v", err)
		return nil
	}
	return body
}

func writeDocument(bucket, object string, doc []byte) error {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		return fmt.Errorf("storage.NewClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(ctx, time.Second*10)
	defer cancel()

	obj := client.Bucket(bucket).Object(object)
	wr := obj.NewWriter(ctx)
	defer wr.Close()

	if _, err := wr.Write(doc); err != nil {
		return fmt.Errorf("Write error: %v", err)
	}
	return nil
}

func getParams(w http.ResponseWriter, r *http.Request) (string, string, string) {
	var d struct {
		Url    string `json:"url"`
		Bucket string `json:"bucket"`
		Object string `json:"object"`
	}
	if err := json.NewDecoder(r.Body).Decode(&d); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "JSON parse error: %v", err)
		return "", "", ""
	}
	return d.Url, d.Bucket, d.Object
}

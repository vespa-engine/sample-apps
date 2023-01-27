package vespasamples

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"github.com/GoogleCloudPlatform/functions-framework-go/functions"
	"html"
	"io"
	"net/http"
	"os"
)

func init() {
	functions.HTTP("helloWorld", helloWorld)
	functions.HTTP("getPage", getPageSimple)
	functions.HTTP("getPageTLS", getPageTLS)
}

// helloHTTP is an HTTP Cloud Function with a request parameter.
func helloWorld(w http.ResponseWriter, r *http.Request) {
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
	Url := getUrl(w, r)
	if Url == "" {
		return
	}
	response, err := http.Get(Url)
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

func getPageTLS(w http.ResponseWriter, r *http.Request) {
	Url := getUrl(w, r)
	if Url == "" {
		return
	}

	// The credentials are stored as Secrets in Google Cloud, exposed as environment variables
	certPem := []byte(os.Getenv("SEC_CERT"))
	keyPem := []byte(os.Getenv("SEC_KEY"))
	cert, err := tls.X509KeyPair(certPem, keyPem)
	if err != nil {
		fmt.Fprintf(w, "Cert error: %v", err)
		return
	}
	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				Certificates:       []tls.Certificate{cert},
				InsecureSkipVerify: true,
			},
		},
	}
	response, err := client.Get(Url)
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
	//fmt.Fprint(w, "Response body: ", string(body))
	fmt.Fprint(w, string(body))
}

func getUrl(w http.ResponseWriter, r *http.Request) string {
	var d struct {
		Url string `json:"url"`
	}
	if err := json.NewDecoder(r.Body).Decode(&d); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "JSON parse error: %v", err)
		return ""
	}
	if d.Url == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, "Input error - url missing")
		return ""
	}
	return d.Url
}

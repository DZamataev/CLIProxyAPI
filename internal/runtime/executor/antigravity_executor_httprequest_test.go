package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
)

// The fork's Antigravity quota revalidation calls retrieveUserQuotaSummary through
// HttpRequest and relies on it for User-Agent, bearer token and per-auth proxy parity.
func TestAntigravityExecutorHttpRequest_UserAgentBearerProxy(t *testing.T) {
	type seen struct {
		requestURI, userAgent, authorization, contentType, junk, body string
	}
	var (
		mu  sync.Mutex
		got []seen
	)
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		got = append(got, seen{
			requestURI:    r.RequestURI,
			userAgent:     r.Header.Get("User-Agent"),
			authorization: r.Header.Get("Authorization"),
			contentType:   r.Header.Get("Content-Type"),
			junk:          r.Header.Get("X-Junk"),
			body:          string(body),
		})
		mu.Unlock()
		_, _ = io.WriteString(w, `{"buckets":[]}`)
	}))
	defer proxy.Close()

	exec := NewAntigravityExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{
		ID:         "antigravity-quota-parity",
		Provider:   "antigravity",
		ProxyURL:   proxy.URL,
		Attributes: map[string]string{"user_agent": "antigravity/9.9.9 linux/amd64"},
		Metadata: map[string]any{
			"access_token": "token-parity",
			"project_id":   "project-1",
			"expired":      time.Now().Add(time.Hour).Format(time.RFC3339),
		},
	}
	const target = "http://quota.example.invalid/v1internal:retrieveUserQuotaSummary"
	req, errRequest := http.NewRequest(http.MethodPost, target, strings.NewReader(`{"project":"project-1"}`))
	if errRequest != nil {
		t.Fatalf("NewRequest: %v", errRequest)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Junk", "must-not-leak")
	req.Header.Set("Authorization", "Bearer caller-token")

	resp, errDo := exec.HttpRequest(context.Background(), auth, req)
	if errDo != nil {
		t.Fatalf("HttpRequest: %v", errDo)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 {
		t.Fatalf("proxy saw %d requests, want 1 (auth proxy not used?)", len(got))
	}
	r := got[0]
	if r.requestURI != target {
		t.Errorf("proxied request URI = %q, want %q", r.requestURI, target)
	}
	if want := resolveUserAgent(auth); r.userAgent != want {
		t.Errorf("User-Agent = %q, want executor UA %q", r.userAgent, want)
	}
	if r.authorization != "Bearer token-parity" {
		t.Errorf("Authorization = %q, want the auth's bearer token", r.authorization)
	}
	if r.contentType != "application/json" || r.body != `{"project":"project-1"}` {
		t.Errorf("content-type %q body %q", r.contentType, r.body)
	}
	if r.junk != "" {
		t.Errorf("inbound header leaked: X-Junk = %q", r.junk)
	}
}

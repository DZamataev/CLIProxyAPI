package auth

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
)

// agQuotaTestExecutor stands in for the Antigravity executor: HttpRequest injects the
// auth's bearer token and sends the request to the test server.
type agQuotaTestExecutor struct {
	mu           sync.Mutex
	calls        int
	refreshCalls int
	authHeaders  []string
	bodies       []string
	contentTypes []string
	refreshErr   error
	duringFetch  func()
}

func (e *agQuotaTestExecutor) Identifier() string { return "antigravity" }

func (e *agQuotaTestExecutor) Execute(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not implemented"}
}

func (e *agQuotaTestExecutor) ExecuteStream(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (*cliproxyexecutor.StreamResult, error) {
	return nil, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not implemented"}
}

func (e *agQuotaTestExecutor) CountTokens(context.Context, *Auth, cliproxyexecutor.Request, cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, &Error{HTTPStatus: http.StatusNotImplemented, Message: "not implemented"}
}

func (e *agQuotaTestExecutor) Refresh(_ context.Context, auth *Auth) (*Auth, error) {
	e.mu.Lock()
	e.refreshCalls++
	refreshErr := e.refreshErr
	e.mu.Unlock()
	if refreshErr != nil {
		return nil, refreshErr
	}
	updated := auth.Clone()
	updated.Metadata["access_token"] = "tok-refreshed"
	updated.Metadata["expired"] = time.Now().Add(time.Hour).Format(time.RFC3339)
	return updated, nil
}

func (e *agQuotaTestExecutor) HttpRequest(ctx context.Context, auth *Auth, req *http.Request) (*http.Response, error) {
	token, _ := auth.Metadata["access_token"].(string)
	req.Header.Set("Authorization", "Bearer "+token)
	body, _ := io.ReadAll(req.Body)
	e.mu.Lock()
	e.calls++
	e.authHeaders = append(e.authHeaders, req.Header.Get("Authorization"))
	e.bodies = append(e.bodies, string(body))
	e.contentTypes = append(e.contentTypes, req.Header.Get("Content-Type"))
	hook := e.duringFetch
	e.mu.Unlock()
	if hook != nil {
		hook()
	}
	out, errReq := http.NewRequestWithContext(ctx, req.Method, req.URL.String(), nil)
	if errReq != nil {
		return nil, errReq
	}
	return http.DefaultClient.Do(out)
}

func (e *agQuotaTestExecutor) callCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.calls
}

const agQuotaSummaryGeminiOK = `{"groups":[{"buckets":[
	{"bucketId":"gemini-5h","remainingFraction":0.95},
	{"bucketId":"gemini-weekly","remainingFraction":0.91},
	{"bucketId":"3p-weekly","remainingFraction":0}
]}]}`

var agQuotaTestSettings = antigravityQuotaRevalidateSettings{interval: time.Minute, minCooldown: 10 * time.Minute, minLeft: 0.02}

func newAGQuotaTestManager(t *testing.T, handler http.HandlerFunc) (*Manager, *agQuotaTestExecutor) {
	t.Helper()
	previous := quotaCooldownDisabled.Load()
	quotaCooldownDisabled.Store(false)
	t.Cleanup(func() { quotaCooldownDisabled.Store(previous) })

	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	previousURL := antigravityQuotaSummaryURL
	antigravityQuotaSummaryURL = server.URL + "/v1internal:retrieveUserQuotaSummary"
	t.Cleanup(func() { antigravityQuotaSummaryURL = previousURL })

	manager := NewManager(nil, &FillFirstSelector{}, nil)
	exec := &agQuotaTestExecutor{}
	manager.RegisterExecutor(exec)
	return manager, exec
}

func agQuotaSummaryHandler(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}
}

func agQuotaCooldown(left time.Duration) *ModelState {
	now := time.Now()
	next := now.Add(left)
	return &ModelState{
		Status:         StatusError,
		StatusMessage:  "quota exhausted",
		Unavailable:    true,
		NextRetryAfter: next,
		LastError:      &Error{HTTPStatus: http.StatusTooManyRequests, Message: "quota exhausted"},
		Quota:          QuotaState{Exceeded: true, Reason: "quota", NextRecoverAt: next},
		UpdatedAt:      now,
	}
}

func agQuotaAuth(id string, states map[string]*ModelState) *Auth {
	return &Auth{
		ID:         id,
		Provider:   "antigravity",
		FileName:   id + ".json",
		Status:     StatusActive,
		Attributes: map[string]string{AttributeAuthKind: AuthKindOAuth},
		Metadata: map[string]any{
			"access_token": "tok-1",
			"expired":      time.Now().Add(time.Hour).Format(time.RFC3339),
			"project_id":   "p1",
		},
		ModelStates: states,
	}
}

func registerAGQuotaAuth(t *testing.T, manager *Manager, auth *Auth) {
	t.Helper()
	reg := registry.GetGlobalRegistry()
	models := make([]*registry.ModelInfo, 0, len(auth.ModelStates))
	for key := range auth.ModelStates {
		models = append(models, &registry.ModelInfo{ID: key})
	}
	reg.RegisterClient(auth.ID, "antigravity", models)
	t.Cleanup(func() { reg.UnregisterClient(auth.ID) })
	if _, errRegister := manager.Register(context.Background(), auth); errRegister != nil {
		t.Fatalf("register %s: %v", auth.ID, errRegister)
	}
}

func agQuotaState(t *testing.T, manager *Manager, authID, key string) *ModelState {
	t.Helper()
	auth, ok := manager.GetByID(authID)
	if !ok || auth == nil {
		t.Fatalf("auth %s missing", authID)
	}
	state := auth.ModelStates[key]
	if state == nil {
		t.Fatalf("model state %s missing", key)
	}
	return state
}

func assertAGQuotaCooled(t *testing.T, manager *Manager, authID, key string) {
	t.Helper()
	state := agQuotaState(t, manager, authID, key)
	if !state.Quota.Exceeded || !state.NextRetryAfter.After(time.Now()) {
		t.Fatalf("%s: cooldown released, want kept (quota %+v next %v)", key, state.Quota, state.NextRetryAfter)
	}
}

func assertAGQuotaReleased(t *testing.T, manager *Manager, authID, key string) {
	t.Helper()
	state := agQuotaState(t, manager, authID, key)
	if state.Quota.Exceeded || state.Unavailable || !state.NextRetryAfter.IsZero() || state.Status != StatusActive || state.LastError != nil {
		t.Fatalf("%s: cooldown kept, want released (status %q quota %+v next %v)", key, state.Status, state.Quota, state.NextRetryAfter)
	}
}

func TestAntigravityQuotaModelFamily(t *testing.T) {
	cases := map[string]string{
		"gemini-4-neon-xhigh":         "gemini",
		"gemini-3.8-flash-high":       "gemini",
		"gemini-3.1-flash-lite":       "gemini",
		"Gemini-4-Neon":               "gemini",
		"gemini-4-neon(high)":         "gemini",
		"gemini-argon-x":              "argon",
		"gemini-4-argon-pro":          "argon",
		"gemini-carbon-mini":          "carbon",
		"gemini-4-carbon":             "carbon",
		"claude-sonnet-4-6":           "3p",
		"Claude-Opus-4-6-thinking":    "3p",
		"gpt-oss-120b":                "3p",
		"antigravity/claude-sonnet-4": "3p",
	}
	for key, want := range cases {
		if got := antigravityQuotaModelFamily(key); got != want {
			t.Errorf("antigravityQuotaModelFamily(%q) = %q, want %q", key, got, want)
		}
	}
	buckets := map[string]string{"gemini-argon-weekly": "argon", "gemini-carbon-5h": "carbon", "3p-weekly": "3p", "gemini-weekly": "gemini", "other": "gemini"}
	for id, want := range buckets {
		if got := antigravityQuotaBucketFamily(id); got != want {
			t.Errorf("antigravityQuotaBucketFamily(%q) = %q, want %q", id, got, want)
		}
	}
}

func TestAntigravityQuotaFamilies(t *testing.T) {
	families, err := antigravityQuotaFamilies([]byte(`{
		"groups":[{"buckets":[
			{"bucketId":"gemini-5h","remainingFraction":0.8},
			{"bucketId":"gemini-weekly","remainingFraction":0.3},
			{"bucketId":"gemini-argon-weekly","remainingFraction":0.9,"disabled":true},
			{"bucketId":"3p-weekly"}
		]}],
		"buckets":[{"bucketId":"gemini-carbon-5h","remainingFraction":0.5}]
	}`))
	if err != nil {
		t.Fatalf("antigravityQuotaFamilies: %v", err)
	}
	want := map[string]float64{"gemini": 0.3, "3p": 0, "carbon": 0.5}
	if len(families) != len(want) {
		t.Fatalf("families = %v, want %v", families, want)
	}
	for family, left := range want {
		if got, ok := families[family]; !ok || got != left {
			t.Errorf("family %s = %v (present %v), want %v", family, got, ok, left)
		}
	}
	if _, ok := families["argon"]; ok {
		t.Errorf("argon present although its only bucket is disabled")
	}
	if _, errBad := antigravityQuotaFamilies([]byte(`{not json`)); errBad == nil {
		t.Errorf("bad JSON: want error")
	}
}

func TestReleasableModelQuotaCooldown(t *testing.T) {
	now := time.Now()
	future := now.Add(time.Hour)
	cases := []struct {
		name   string
		auth   func(*Auth)
		state  func(*ModelState)
		wantOK bool
	}{
		{name: "plain quota", wantOK: true},
		{name: "expired", state: func(s *ModelState) {
			s.NextRetryAfter = now.Add(-time.Second)
			s.Quota.NextRecoverAt = now.Add(-time.Second)
		}},
		{name: "legacy zero deadline", state: func(s *ModelState) { s.NextRetryAfter = time.Time{}; s.Quota.NextRecoverAt = time.Time{} }},
		{name: "credential_quota reason", state: func(s *ModelState) { s.Quota.Reason = "credential_quota" }},
		{name: "cloudflare reason", state: func(s *ModelState) { s.Quota.Reason = "cloudflare challenge" }},
		{name: "shadow", state: func(s *ModelState) { s.credentialQuotaShadow = agQuotaCooldown(time.Hour) }},
		{name: "propagated", state: func(s *ModelState) { s.credentialQuotaPropagated = true }},
		{name: "seeded force", state: func(s *ModelState) { s.forceCooldownSeeded = true; s.forceCooldownUntil = future }},
		{name: "force last error", state: func(s *ModelState) {
			s.LastError = &Error{Code: ErrorCodeForceCooldown, HTTPStatus: http.StatusTooManyRequests}
		}},
		{name: "quota force", state: func(s *ModelState) { s.quotaForceCooldownUntil = future }},
		{name: "non-quota layer", state: func(s *ModelState) { s.nonQuotaCooldownUntil = future }},
		{name: "non-429 last error", state: func(s *ModelState) { s.LastError = &Error{HTTPStatus: http.StatusServiceUnavailable} }},
		{name: "state disabled", state: func(s *ModelState) { s.Status = StatusDisabled }},
		{name: "auth disabled", auth: func(a *Auth) { a.Disabled = true }},
		{name: "auth status disabled", auth: func(a *Auth) { a.Status = StatusDisabled }},
		{name: "auth unauthorized", auth: func(a *Auth) {
			a.LastError = &Error{Code: "unauthorized", HTTPStatus: http.StatusUnauthorized}
			a.Unavailable = true
			a.Status = StatusError
		}},
		{name: "auth credential quota", auth: func(a *Auth) {
			a.Quota = QuotaState{Exceeded: true, Reason: "credential_quota", NextRecoverAt: future}
		}},
		{name: "auth force", auth: func(a *Auth) { a.forceCooldownSeeded = true; a.forceCooldownUntil = future }},
		{name: "auth quota force", auth: func(a *Auth) { a.quotaForceCooldownUntil = future }},
		{name: "auth credential retry", auth: func(a *Auth) { a.credentialRetryAfter = future }},
		{name: "auth non-quota layer", auth: func(a *Auth) { a.nonQuotaCooldownUntil = future }},
		{name: "expired auth layers", auth: func(a *Auth) {
			a.credentialRetryAfter = now.Add(-time.Second)
			a.nonQuotaCooldownUntil = now.Add(-time.Second)
		}, wantOK: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			state := agQuotaCooldown(time.Hour)
			auth := agQuotaAuth("releasable", map[string]*ModelState{"gemini-4-neon-xhigh": state})
			if tc.auth != nil {
				tc.auth(auth)
			}
			if tc.state != nil {
				tc.state(state)
			}
			deadline, ok := releasableModelQuotaCooldown(auth, state, now)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && !deadline.Equal(state.NextRetryAfter) {
				t.Fatalf("deadline = %v, want %v", deadline, state.NextRetryAfter)
			}
		})
	}
	t.Run("deadline is the later of retry and recover", func(t *testing.T) {
		state := agQuotaCooldown(time.Hour)
		state.Quota.NextRecoverAt = now.Add(2 * time.Hour)
		deadline, ok := releasableModelQuotaCooldown(agQuotaAuth("releasable", nil), state, now)
		if !ok || !deadline.Equal(state.Quota.NextRecoverAt) {
			t.Fatalf("deadline = %v ok %v, want %v", deadline, ok, state.Quota.NextRecoverAt)
		}
	})
}

func TestQuotaReleaseFenceFor_MinCooldownBoundary(t *testing.T) {
	manager, _ := newAGQuotaTestManager(t, agQuotaSummaryHandler(agQuotaSummaryGeminiOK))
	now := time.Now()
	atMin := agQuotaCooldown(time.Hour)
	atMin.NextRetryAfter = now.Add(10 * time.Minute)
	atMin.Quota.NextRecoverAt = atMin.NextRetryAfter
	belowMin := agQuotaCooldown(time.Hour)
	belowMin.NextRetryAfter = now.Add(10*time.Minute - time.Second)
	belowMin.Quota.NextRecoverAt = belowMin.NextRetryAfter
	registerAGQuotaAuth(t, manager, agQuotaAuth("ag-quota-boundary", map[string]*ModelState{"gemini-at-min": atMin, "gemini-below-min": belowMin}))

	fence, snapshot, ok := manager.quotaReleaseFenceFor("ag-quota-boundary", now, 10*time.Minute)
	if !ok || snapshot == nil {
		t.Fatalf("fence not taken")
	}
	if _, has := fence.models["gemini-at-min"]; !has {
		t.Errorf("cooldown with exactly min-cooldown left not fenced")
	}
	if _, has := fence.models["gemini-below-min"]; has {
		t.Errorf("cooldown below min-cooldown fenced")
	}
	if fence.projectID != "p1" {
		t.Errorf("fence project = %q, want p1", fence.projectID)
	}
}

func TestAntigravityQuotaRevalidate_ReleasesOnlyRecoveredFamily(t *testing.T) {
	manager, exec := newAGQuotaTestManager(t, agQuotaSummaryHandler(agQuotaSummaryGeminiOK))
	const id = "ag-quota-release"
	registerAGQuotaAuth(t, manager, agQuotaAuth(id, map[string]*ModelState{
		"gemini-4-neon-xhigh": agQuotaCooldown(100 * time.Hour),
		"gemini-4-neon":       agQuotaCooldown(100 * time.Hour),
		"claude-sonnet-4-6":   agQuotaCooldown(100 * time.Hour),
		"gemini-argon-x":      agQuotaCooldown(100 * time.Hour),
	}))
	reg := registry.GetGlobalRegistry()
	reg.SuspendClientModel(id, "gemini-4-neon-xhigh", "quota")
	if count := reg.GetModelCount("gemini-4-neon-xhigh"); count != 0 {
		t.Fatalf("registry model count before = %d, want 0", count)
	}
	if picked, _, errPick := manager.pickNext(context.Background(), "antigravity", "gemini-4-neon-xhigh", cliproxyexecutor.Options{}, nil); errPick == nil && picked != nil {
		t.Fatalf("cooled auth selectable before revalidation")
	}
	claudeBefore := agQuotaState(t, manager, id, "claude-sonnet-4-6").NextRetryAfter
	before, _ := manager.GetByID(id)

	if ran := manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings); !ran {
		t.Fatalf("pass did not run")
	}

	assertAGQuotaReleased(t, manager, id, "gemini-4-neon-xhigh")
	assertAGQuotaReleased(t, manager, id, "gemini-4-neon")
	assertAGQuotaCooled(t, manager, id, "claude-sonnet-4-6")
	assertAGQuotaCooled(t, manager, id, "gemini-argon-x")
	if got := agQuotaState(t, manager, id, "claude-sonnet-4-6").NextRetryAfter; !got.Equal(claudeBefore) {
		t.Errorf("kept claude deadline moved: %v → %v", claudeBefore, got)
	}
	after, _ := manager.GetByID(id)
	if after.Generation != before.Generation+1 {
		t.Errorf("generation %d → %d, want one bump", before.Generation, after.Generation)
	}
	picked, _, errPick := manager.pickNext(context.Background(), "antigravity", "gemini-4-neon-xhigh", cliproxyexecutor.Options{}, nil)
	if errPick != nil || picked == nil || picked.ID != id {
		t.Fatalf("released auth not selectable: %v", errPick)
	}
	if count := reg.GetModelCount("gemini-4-neon-xhigh"); count != 1 {
		t.Errorf("registry model count after = %d, want 1", count)
	}
	if exec.callCount() != 1 {
		t.Fatalf("summary calls = %d, want 1", exec.callCount())
	}
	if exec.bodies[0] != `{"project":"p1"}` || exec.authHeaders[0] != "Bearer tok-1" || exec.contentTypes[0] != "application/json" {
		t.Errorf("request body %q auth %q content-type %q", exec.bodies[0], exec.authHeaders[0], exec.contentTypes[0])
	}
}

func TestAntigravityQuotaRevalidate_RecomputesAuthAvailability(t *testing.T) {
	manager, _ := newAGQuotaTestManager(t, agQuotaSummaryHandler(agQuotaSummaryGeminiOK))
	const id, key = "ag-quota-aggregate", "gemini-4-neon-xhigh"
	registerAGQuotaAuth(t, manager, agQuotaAuth(id, map[string]*ModelState{key: {Status: StatusActive}}))
	retry := 100 * time.Hour
	manager.MarkResult(context.Background(), Result{
		AuthID: id, Provider: "antigravity", Model: key, Success: false,
		Error:      &Error{HTTPStatus: http.StatusTooManyRequests, Message: "Individual quota reached."},
		RetryAfter: &retry,
	})
	if before, _ := manager.GetByID(id); !before.Unavailable || !before.Quota.Exceeded {
		t.Fatalf("MarkResult did not cool the auth: unavailable %v quota %+v", before.Unavailable, before.Quota)
	}

	manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings)

	assertAGQuotaReleased(t, manager, id, key)
	after, _ := manager.GetByID(id)
	// auth.Quota stays a sticky model aggregate here, exactly as after a real success;
	// selection and the cooldown view ignore it while model states exist.
	if after.Unavailable || after.Status != StatusActive || after.LastError != nil || !after.NextRetryAfter.IsZero() {
		t.Fatalf("auth not recomputed: unavailable %v status %q last error %v next %v", after.Unavailable, after.Status, after.LastError, after.NextRetryAfter)
	}
	if views := CooldownSnapshotForAuth(after, time.Now()); len(views) != 0 {
		t.Fatalf("cooldown view after release = %+v, want none", views)
	}
}

func TestAntigravityQuotaRevalidate_MinLeftBoundary(t *testing.T) {
	for _, tc := range []struct {
		name    string
		left    string
		release bool
	}{{"equal min-left kept", "0.02", false}, {"above min-left released", "0.0201", true}} {
		t.Run(tc.name, func(t *testing.T) {
			manager, _ := newAGQuotaTestManager(t, agQuotaSummaryHandler(`{"buckets":[{"bucketId":"gemini-weekly","remainingFraction":`+tc.left+`}]}`))
			id := "ag-quota-minleft-" + tc.left
			registerAGQuotaAuth(t, manager, agQuotaAuth(id, map[string]*ModelState{"gemini-4-neon-xhigh": agQuotaCooldown(time.Hour)}))
			manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings)
			if tc.release {
				assertAGQuotaReleased(t, manager, id, "gemini-4-neon-xhigh")
			} else {
				assertAGQuotaCooled(t, manager, id, "gemini-4-neon-xhigh")
			}
		})
	}
}

func TestAntigravityQuotaRevalidate_FailuresKeepCooldowns(t *testing.T) {
	cases := []struct {
		name       string
		handler    http.HandlerFunc
		mutate     func(*Auth)
		refreshErr error
		closeSrv   bool
		wantCalls  int
	}{
		{name: "http 500", handler: func(w http.ResponseWriter, _ *http.Request) { http.Error(w, "boom", http.StatusInternalServerError) }, wantCalls: 1},
		{name: "bad json", handler: agQuotaSummaryHandler(`{"groups":`), wantCalls: 1},
		{name: "network error", handler: agQuotaSummaryHandler(agQuotaSummaryGeminiOK), closeSrv: true, wantCalls: 1},
		{name: "missing project", handler: agQuotaSummaryHandler(agQuotaSummaryGeminiOK), mutate: func(a *Auth) { delete(a.Metadata, "project_id") }},
		{name: "refresh error", handler: agQuotaSummaryHandler(agQuotaSummaryGeminiOK), mutate: func(a *Auth) {
			a.Metadata["expired"] = time.Now().Add(2 * time.Minute).Format(time.RFC3339)
			a.Metadata["refresh_token"] = "rt-1"
		}, refreshErr: errors.New("refresh failed")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var server *httptest.Server
			handler := tc.handler
			manager, exec := newAGQuotaTestManager(t, func(w http.ResponseWriter, r *http.Request) { handler(w, r) })
			exec.refreshErr = tc.refreshErr
			if tc.closeSrv {
				server = httptest.NewServer(handler)
				server.Close()
				antigravityQuotaSummaryURL = server.URL
			}
			id := "ag-quota-fail"
			auth := agQuotaAuth(id, map[string]*ModelState{"gemini-4-neon-xhigh": agQuotaCooldown(time.Hour)})
			if tc.mutate != nil {
				tc.mutate(auth)
			}
			registerAGQuotaAuth(t, manager, auth)
			before := agQuotaState(t, manager, id, "gemini-4-neon-xhigh")

			manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings)

			after := agQuotaState(t, manager, id, "gemini-4-neon-xhigh")
			if !modelQuotaFenceOf(after).equal(modelQuotaFenceOf(before)) || !after.Quota.Exceeded {
				t.Fatalf("cooldown changed on failure: before %+v after %+v", modelQuotaFenceOf(before), modelQuotaFenceOf(after))
			}
			if got := exec.callCount(); got != tc.wantCalls {
				t.Fatalf("summary calls = %d, want %d", got, tc.wantCalls)
			}
		})
	}
}

func TestAntigravityQuotaRevalidate_RefreshesExpiringTokenThroughManager(t *testing.T) {
	manager, exec := newAGQuotaTestManager(t, agQuotaSummaryHandler(agQuotaSummaryGeminiOK))
	const id = "ag-quota-refresh"
	auth := agQuotaAuth(id, map[string]*ModelState{"gemini-4-neon-xhigh": agQuotaCooldown(time.Hour)})
	auth.Metadata["expired"] = time.Now().Add(2 * time.Minute).Format(time.RFC3339)
	auth.Metadata["refresh_token"] = "rt-1"
	registerAGQuotaAuth(t, manager, auth)

	manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings)

	if exec.refreshCalls != 1 {
		t.Fatalf("refresh calls = %d, want 1", exec.refreshCalls)
	}
	committed, _ := manager.GetByID(id)
	if token, _ := committed.Metadata["access_token"].(string); token != "tok-refreshed" {
		t.Fatalf("committed token = %q, want tok-refreshed", token)
	}
	if exec.callCount() != 1 || exec.authHeaders[0] != "Bearer tok-refreshed" {
		t.Fatalf("summary request auth = %v, want the refreshed token", exec.authHeaders)
	}
	assertAGQuotaReleased(t, manager, id, "gemini-4-neon-xhigh")
}

func TestReleaseModelQuotaCooldowns_FenceAborts(t *testing.T) {
	const key = "gemini-4-neon-xhigh"
	cases := []struct {
		name  string
		hook  func(t *testing.T, manager *Manager, id string)
		check func(t *testing.T, manager *Manager, id string)
	}{
		{
			name: "second 429 extends the cooldown",
			hook: func(_ *testing.T, manager *Manager, id string) {
				retry := 200 * time.Hour
				manager.MarkResult(context.Background(), Result{
					AuthID: id, Provider: "antigravity", Model: key, Success: false,
					Error:      &Error{HTTPStatus: http.StatusTooManyRequests, Message: "quota exhausted"},
					RetryAfter: &retry,
				})
			},
			check: func(t *testing.T, manager *Manager, id string) {
				assertAGQuotaCooled(t, manager, id, key)
				if next := agQuotaState(t, manager, id, key).NextRetryAfter; next.Before(time.Now().Add(199 * time.Hour)) {
					t.Fatalf("new 429 deadline lost: %v", next)
				}
			},
		},
		{
			name: "credential replaced",
			hook: func(t *testing.T, manager *Manager, id string) {
				current, _ := manager.GetByID(id)
				current.Metadata["access_token"] = "tok-other"
				if _, errUpdate := manager.Update(context.Background(), current); errUpdate != nil {
					t.Errorf("update: %v", errUpdate)
				}
			},
			check: func(t *testing.T, manager *Manager, id string) { assertAGQuotaCooled(t, manager, id, key) },
		},
		{
			name: "project changed",
			hook: func(t *testing.T, manager *Manager, id string) {
				current, _ := manager.GetByID(id)
				current.Metadata["project_id"] = "p2"
				if _, errUpdate := manager.Update(context.Background(), current); errUpdate != nil {
					t.Errorf("update: %v", errUpdate)
				}
			},
			check: func(t *testing.T, manager *Manager, id string) {
				current, _ := manager.GetByID(id)
				if project, _ := current.Metadata["project_id"].(string); project != "p2" {
					t.Fatalf("project not updated: %q", project)
				}
				assertAGQuotaCooled(t, manager, id, key)
			},
		},
		{
			name: "auth re-registered",
			hook: func(t *testing.T, manager *Manager, id string) {
				// Same model state values, so only the registration epoch tells them apart.
				current, _ := manager.GetByID(id)
				manager.Remove(context.Background(), id)
				replacement := agQuotaAuth(id, map[string]*ModelState{key: current.ModelStates[key].Clone()})
				if _, errRegister := manager.Register(context.Background(), replacement); errRegister != nil {
					t.Errorf("re-register: %v", errRegister)
				}
			},
			check: func(t *testing.T, manager *Manager, id string) { assertAGQuotaCooled(t, manager, id, key) },
		},
		{
			name: "success clears it first",
			hook: func(_ *testing.T, manager *Manager, id string) {
				manager.MarkResult(context.Background(), Result{AuthID: id, Provider: "antigravity", Model: key, Success: true})
			},
			check: func(t *testing.T, manager *Manager, id string) { assertAGQuotaReleased(t, manager, id, key) },
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			manager, exec := newAGQuotaTestManager(t, agQuotaSummaryHandler(agQuotaSummaryGeminiOK))
			const id = "ag-quota-fence"
			registerAGQuotaAuth(t, manager, agQuotaAuth(id, map[string]*ModelState{key: agQuotaCooldown(time.Hour)}))
			exec.duringFetch = func() { tc.hook(t, manager, id) }
			fence, snapshot, ok := manager.quotaReleaseFenceFor(id, time.Now(), agQuotaTestSettings.minCooldown)
			if !ok {
				t.Fatalf("fence not taken")
			}
			if _, errFetch := manager.fetchAntigravityQuotaFamilies(context.Background(), snapshot, fence.projectID); errFetch != nil {
				t.Fatalf("fetch: %v", errFetch)
			}
			released, errRelease := manager.releaseModelQuotaCooldowns(context.Background(), id, fence, []string{key})
			if errRelease != nil || len(released) != 0 {
				t.Fatalf("released = %v err %v, want nothing", released, errRelease)
			}
			tc.check(t, manager, id)
		})
	}
}

// agQuotaMemoryStore is a CooldownStateStore whose Load returns what was last saved.
type agQuotaMemoryStore struct {
	mu      sync.Mutex
	records []CooldownStateRecord
}

func (s *agQuotaMemoryStore) Load(context.Context) ([]CooldownStateRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]CooldownStateRecord(nil), s.records...), nil
}

func (s *agQuotaMemoryStore) Save(_ context.Context, records []CooldownStateRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records = append([]CooldownStateRecord(nil), records...)
	return nil
}

func (s *agQuotaMemoryStore) models(authID string) map[string]bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]bool)
	for _, record := range s.records {
		if record.AuthID == authID && record.Model != "" {
			out[record.Model] = true
		}
	}
	return out
}

func TestReleaseModelQuotaCooldowns_CooldownStoreRoundTrip(t *testing.T) {
	manager, _ := newAGQuotaTestManager(t, agQuotaSummaryHandler(agQuotaSummaryGeminiOK))
	store := &agQuotaMemoryStore{}
	manager.SetCooldownStateStore(store)
	const id = "ag-quota-store"
	registerAGQuotaAuth(t, manager, agQuotaAuth(id, map[string]*ModelState{
		"gemini-4-neon-xhigh": agQuotaCooldown(time.Hour),
		"claude-sonnet-4-6":   agQuotaCooldown(time.Hour),
	}))
	manager.persistCooldownStates(context.Background())
	if models := store.models(id); !models["gemini-4-neon-xhigh"] || !models["claude-sonnet-4-6"] {
		t.Fatalf("seeded store models = %v", models)
	}

	manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings)

	if models := store.models(id); models["gemini-4-neon-xhigh"] || !models["claude-sonnet-4-6"] {
		t.Fatalf("store models after release = %v, want only claude-sonnet-4-6", models)
	}

	restored := NewManager(nil, &FillFirstSelector{}, nil)
	restored.RegisterExecutor(&agQuotaTestExecutor{})
	restored.SetCooldownStateStore(store)
	if _, errRegister := restored.Register(context.Background(), agQuotaAuth(id, nil)); errRegister != nil {
		t.Fatalf("register restored: %v", errRegister)
	}
	if errRestore := restored.RestoreCooldownStates(context.Background()); errRestore != nil {
		t.Fatalf("restore: %v", errRestore)
	}
	auth, _ := restored.GetByID(id)
	if state := auth.ModelStates["gemini-4-neon-xhigh"]; state != nil && state.Quota.Exceeded {
		t.Fatalf("released cooldown restored: %+v", state.Quota)
	}
	if state := auth.ModelStates["claude-sonnet-4-6"]; state == nil || !state.Quota.Exceeded {
		t.Fatalf("kept cooldown not restored")
	}
}

func TestAntigravityQuotaRevalidate_GuardSkipsConcurrentPass(t *testing.T) {
	entered := make(chan struct{})
	unblock := make(chan struct{})
	var once sync.Once
	manager, exec := newAGQuotaTestManager(t, agQuotaSummaryHandler(agQuotaSummaryGeminiOK))
	exec.duringFetch = func() {
		first := false
		once.Do(func() { first = true; close(entered) })
		if first {
			<-unblock
		}
	}
	registerAGQuotaAuth(t, manager, agQuotaAuth("ag-quota-guard", map[string]*ModelState{"gemini-4-neon-xhigh": agQuotaCooldown(time.Hour)}))

	done := make(chan bool, 1)
	go func() { done <- manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings) }()
	<-entered
	if ran := manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings); ran {
		t.Fatalf("second pass ran while the first was in flight")
	}
	close(unblock)
	if ran := <-done; !ran {
		t.Fatalf("first pass reported skipped")
	}
	if got := exec.callCount(); got != 1 {
		t.Fatalf("summary calls = %d, want 1", got)
	}
	exec.mu.Lock()
	exec.duringFetch = nil
	exec.mu.Unlock()
	if ran := manager.revalidateAntigravityQuotaCooldowns(context.Background(), agQuotaTestSettings); !ran {
		t.Fatalf("pass after the first finished was skipped")
	}
}

func TestAntigravityQuotaRevalidateSettingsFromConfig(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	cases := []struct {
		name        string
		cfg         internalconfig.AntigravityConfig
		interval    time.Duration
		minCooldown time.Duration
		minLeft     float64
		warn        bool
	}{
		{name: "defaults", minCooldown: 10 * time.Minute, minLeft: 0.02},
		{name: "off", cfg: internalconfig.AntigravityConfig{QuotaRevalidateInterval: "OFF"}, minCooldown: 10 * time.Minute, minLeft: 0.02},
		{name: "zero", cfg: internalconfig.AntigravityConfig{QuotaRevalidateInterval: "0"}, minCooldown: 10 * time.Minute, minLeft: 0.02},
		{name: "enabled", cfg: internalconfig.AntigravityConfig{QuotaRevalidateInterval: "15m", QuotaRevalidateMinCooldown: "0", QuotaRevalidateMinLeft: f(0)}, interval: 15 * time.Minute},
		{name: "garbage interval", cfg: internalconfig.AntigravityConfig{QuotaRevalidateInterval: "soon"}, minCooldown: 10 * time.Minute, minLeft: 0.02, warn: true},
		{name: "negative interval", cfg: internalconfig.AntigravityConfig{QuotaRevalidateInterval: "-5m"}, minCooldown: 10 * time.Minute, minLeft: 0.02, warn: true},
		{name: "bad min-cooldown", cfg: internalconfig.AntigravityConfig{QuotaRevalidateInterval: "15m", QuotaRevalidateMinCooldown: "-1m"}, interval: 15 * time.Minute, minCooldown: 10 * time.Minute, minLeft: 0.02, warn: true},
		{name: "min-left one", cfg: internalconfig.AntigravityConfig{QuotaRevalidateMinLeft: f(1)}, minCooldown: 10 * time.Minute, minLeft: 1},
		{name: "min-left above one", cfg: internalconfig.AntigravityConfig{QuotaRevalidateMinLeft: f(1.5)}, minCooldown: 10 * time.Minute, minLeft: 0.02, warn: true},
		{name: "min-left negative", cfg: internalconfig.AntigravityConfig{QuotaRevalidateMinLeft: f(-0.1)}, minCooldown: 10 * time.Minute, minLeft: 0.02, warn: true},
		{name: "min-left NaN", cfg: internalconfig.AntigravityConfig{QuotaRevalidateMinLeft: f(math.NaN())}, minCooldown: 10 * time.Minute, minLeft: 0.02, warn: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			settings, warning := antigravityQuotaRevalidateSettingsFromConfig(&internalconfig.Config{Antigravity: tc.cfg})
			if settings.interval != tc.interval || settings.minCooldown != tc.minCooldown || settings.minLeft != tc.minLeft {
				t.Fatalf("settings = %+v, want interval %s min-cooldown %s min-left %v", settings, tc.interval, tc.minCooldown, tc.minLeft)
			}
			if (warning != "") != tc.warn {
				t.Fatalf("warning = %q, want warning %v", warning, tc.warn)
			}
		})
	}
	if settings, warning := antigravityQuotaRevalidateSettingsFromConfig(nil); settings.interval != 0 || warning != "" {
		t.Fatalf("nil config: %+v %q", settings, warning)
	}
}

func TestParseConfigBytes_AntigravityQuotaRevalidate(t *testing.T) {
	cfg, errParse := internalconfig.ParseConfigBytes([]byte("antigravity:\n  quota-revalidate-interval: 15m\n  quota-revalidate-min-cooldown: 5m\n  quota-revalidate-min-left: 0.05\n"))
	if errParse != nil {
		t.Fatalf("parse: %v", errParse)
	}
	settings, warning := antigravityQuotaRevalidateSettingsFromConfig(cfg.CloneForRuntime())
	if warning != "" || settings.interval != 15*time.Minute || settings.minCooldown != 5*time.Minute || settings.minLeft != 0.05 {
		t.Fatalf("settings = %+v warning %q", settings, warning)
	}
}

func TestAntigravityQuotaRevalidateLoop_HotReloadAndStop(t *testing.T) {
	var hits atomic.Int64
	manager, _ := newAGQuotaTestManager(t, func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		_, _ = io.WriteString(w, `{"buckets":[{"bucketId":"gemini-weekly","remainingFraction":0}]}`)
	})
	// Quota stays exhausted, so the cooldown is kept and every due pass probes again.
	registerAGQuotaAuth(t, manager, agQuotaAuth("ag-quota-loop", map[string]*ModelState{"gemini-4-neon-xhigh": agQuotaCooldown(time.Hour)}))

	ctx, cancel := context.WithCancel(context.Background())
	stopped := make(chan struct{})
	go func() {
		manager.runAntigravityQuotaRevalidateEvery(ctx, 5*time.Millisecond)
		close(stopped)
	}()
	waitFor := func(what string, cond func() bool) {
		t.Helper()
		deadline := time.Now().Add(5 * time.Second)
		for !cond() {
			if time.Now().After(deadline) {
				t.Fatalf("timed out waiting for %s", what)
			}
			time.Sleep(5 * time.Millisecond)
		}
	}

	time.Sleep(50 * time.Millisecond)
	if got := hits.Load(); got != 0 {
		t.Fatalf("disabled by default, but %d summary calls", got)
	}
	manager.SetConfig(&internalconfig.Config{Antigravity: internalconfig.AntigravityConfig{QuotaRevalidateInterval: "1ms"}})
	waitFor("passes after enabling", func() bool { return hits.Load() >= 2 })

	manager.SetConfig(&internalconfig.Config{Antigravity: internalconfig.AntigravityConfig{QuotaRevalidateInterval: "off"}})
	time.Sleep(50 * time.Millisecond)
	settled := hits.Load()
	time.Sleep(50 * time.Millisecond)
	if got := hits.Load(); got != settled {
		t.Fatalf("passes continued after disabling: %d → %d", settled, got)
	}

	cancel()
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Fatalf("worker did not stop on context cancel")
	}
}

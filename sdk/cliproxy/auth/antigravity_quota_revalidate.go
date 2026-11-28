package auth

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/MoeForge/moego/json"
	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	log "github.com/sirupsen/logrus"
)

// Antigravity quota revalidation (fork-only).
//
// Google's 429 QUOTA_EXHAUSTED can carry a multi-day reset that later turns out
// to be wrong or out of date, and MarkResult holds such a model cooldown until it
// expires. When antigravity.quota-revalidate-interval is set, a background pass
// asks retrieveUserQuotaSummary for every Antigravity auth with a long plain
// quota cooldown and releases exactly the model states whose quota family still
// has quota left.

// antigravityQuotaSummaryURL is a variable only so tests can point it at a local server.
var antigravityQuotaSummaryURL = "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary"

const (
	antigravityQuotaRevalidateTick        = time.Minute
	antigravityQuotaRevalidateAuthTimeout = 30 * time.Second
	// Longer than the executor's 5 min request safety window, so the executor never
	// refreshes on its own (and uncommitted) while serving the summary request.
	antigravityQuotaTokenMargin        = 10 * time.Minute
	antigravityQuotaSummaryMaxBytes    = 1 << 20
	antigravityQuotaDefaultMinCooldown = 10 * time.Minute
	antigravityQuotaDefaultMinLeft     = 0.02
)

type antigravityQuotaRevalidateSettings struct {
	interval    time.Duration
	minCooldown time.Duration
	minLeft     float64
}

// antigravityQuotaRevalidateSettingsFromConfig resolves the antigravity.quota-revalidate-*
// keys. Invalid values never fail config load: a bad interval disables revalidation and
// a bad threshold falls back to its default; the returned warning describes what was ignored.
func antigravityQuotaRevalidateSettingsFromConfig(cfg *internalconfig.Config) (antigravityQuotaRevalidateSettings, string) {
	settings := antigravityQuotaRevalidateSettings{
		minCooldown: antigravityQuotaDefaultMinCooldown,
		minLeft:     antigravityQuotaDefaultMinLeft,
	}
	if cfg == nil {
		return settings, ""
	}
	var warnings []string
	raw := strings.TrimSpace(cfg.Antigravity.QuotaRevalidateInterval)
	switch strings.ToLower(raw) {
	case "", "0", "off", "false", "none", "disabled":
	default:
		interval, errParse := time.ParseDuration(raw)
		if errParse != nil || interval <= 0 {
			warnings = append(warnings, fmt.Sprintf("quota-revalidate-interval %q is invalid, revalidation disabled", raw))
		} else {
			settings.interval = interval
		}
	}
	if raw := strings.TrimSpace(cfg.Antigravity.QuotaRevalidateMinCooldown); raw != "" {
		minCooldown, errParse := time.ParseDuration(raw)
		if errParse != nil || minCooldown < 0 {
			warnings = append(warnings, fmt.Sprintf("quota-revalidate-min-cooldown %q is invalid, using %s", raw, antigravityQuotaDefaultMinCooldown))
		} else {
			settings.minCooldown = minCooldown
		}
	}
	if minLeft := cfg.Antigravity.QuotaRevalidateMinLeft; minLeft != nil {
		if math.IsNaN(*minLeft) || *minLeft < 0 || *minLeft > 1 {
			warnings = append(warnings, fmt.Sprintf("quota-revalidate-min-left %v is outside [0,1], using %v", *minLeft, antigravityQuotaDefaultMinLeft))
		} else {
			settings.minLeft = *minLeft
		}
	}
	return settings, strings.Join(warnings, "; ")
}

// antigravityQuotaModelFamily maps a ModelStates key to the quota family Google reports it under.
func antigravityQuotaModelFamily(modelKey string) string {
	key := canonicalModelLookupKey(modelKey)
	if i := strings.LastIndex(key, "/"); i >= 0 {
		key = key[i+1:]
	}
	switch {
	case strings.HasPrefix(key, "gemini-argon-"), strings.HasPrefix(key, "gemini-4-argon"):
		return "argon"
	case strings.HasPrefix(key, "gemini-carbon-"), strings.HasPrefix(key, "gemini-4-carbon"):
		return "carbon"
	case strings.HasPrefix(key, "claude"), strings.HasPrefix(key, "gpt"):
		return "3p"
	}
	return "gemini"
}

// antigravityQuotaBucketFamily maps a retrieveUserQuotaSummary bucketId to its quota family.
func antigravityQuotaBucketFamily(bucketID string) string {
	id := strings.ToLower(strings.TrimSpace(bucketID))
	switch {
	case strings.HasPrefix(id, "gemini-argon-"):
		return "argon"
	case strings.HasPrefix(id, "gemini-carbon-"):
		return "carbon"
	case strings.HasPrefix(id, "3p-"):
		return "3p"
	}
	return "gemini"
}

type antigravityQuotaBucket struct {
	BucketID          string   `json:"bucketId"`
	RemainingFraction *float64 `json:"remainingFraction"`
	Disabled          bool     `json:"disabled"`
}

type antigravityQuotaSummary struct {
	Groups []struct {
		Buckets []antigravityQuotaBucket `json:"buckets"`
	} `json:"groups"`
	Buckets []antigravityQuotaBucket `json:"buckets"`
}

// antigravityQuotaFamilies returns, per family, the minimum remaining fraction over its
// non-disabled buckets (a missing remainingFraction counts as 0).
func antigravityQuotaFamilies(body []byte) (map[string]float64, error) {
	var summary antigravityQuotaSummary
	if errUnmarshal := json.Unmarshal(body, &summary); errUnmarshal != nil {
		return nil, fmt.Errorf("%w: %w", errAntigravityQuotaDecode, errUnmarshal)
	}
	families := make(map[string]float64)
	add := func(buckets []antigravityQuotaBucket) {
		for _, bucket := range buckets {
			if bucket.Disabled {
				continue
			}
			left := 0.0
			if bucket.RemainingFraction != nil {
				left = *bucket.RemainingFraction
			}
			family := antigravityQuotaBucketFamily(bucket.BucketID)
			if current, ok := families[family]; !ok || left < current {
				families[family] = left
			}
		}
	}
	for _, group := range summary.Groups {
		add(group.Buckets)
	}
	add(summary.Buckets)
	return families, nil
}

// releasableModelQuotaCooldown reports whether state's only active block is its own plain
// "quota" cooldown, and returns that cooldown's deadline. Any other active layer on the
// model or the credential (force, quota-force, non-quota, credential retry, credential
// quota, shadow/propagation, disabled, unauthorized) keeps the state as is.
func releasableModelQuotaCooldown(auth *Auth, state *ModelState, now time.Time) (time.Time, bool) {
	if auth == nil || state == nil {
		return time.Time{}, false
	}
	if auth.Disabled || auth.Status == StatusDisabled || hasUnauthorizedAuthFailure(auth) {
		return time.Time{}, false
	}
	if auth.Quota.Exceeded && auth.Quota.Reason == "credential_quota" && auth.Quota.NextRecoverAt.After(now) {
		return time.Time{}, false
	}
	if effectiveAuthForceUntil(auth).After(now) || auth.quotaForceCooldownUntil.After(now) ||
		auth.credentialRetryAfter.After(now) || auth.nonQuotaCooldownUntil.After(now) {
		return time.Time{}, false
	}
	if state.Status == StatusDisabled || !state.Quota.Exceeded || strings.TrimSpace(state.Quota.Reason) != "quota" {
		return time.Time{}, false
	}
	if state.credentialQuotaShadow != nil || state.credentialQuotaPropagated {
		return time.Time{}, false
	}
	if effectiveModelForceUntil(state).After(now) || state.quotaForceCooldownUntil.After(now) || state.nonQuotaCooldownUntil.After(now) {
		return time.Time{}, false
	}
	if state.LastError != nil && (state.LastError.HTTPStatus != http.StatusTooManyRequests || state.LastError.Code == ErrorCodeForceCooldown) {
		return time.Time{}, false
	}
	if !isModelStateQuotaCooldown(state, now) {
		return time.Time{}, false
	}
	deadline := state.NextRetryAfter
	if state.Quota.NextRecoverAt.After(deadline) {
		deadline = state.Quota.NextRecoverAt
	}
	if !deadline.After(now) {
		return time.Time{}, false
	}
	return deadline, true
}

// modelQuotaFence identifies one model cooldown; any MarkResult on the model changes it.
type modelQuotaFence struct {
	nextRetryAfter time.Time
	nextRecoverAt  time.Time
	updatedAt      time.Time
	recordedSeq    uint64
	backoffLevel   int
}

func modelQuotaFenceOf(state *ModelState) modelQuotaFence {
	return modelQuotaFence{
		nextRetryAfter: state.NextRetryAfter,
		nextRecoverAt:  state.Quota.NextRecoverAt,
		updatedAt:      state.UpdatedAt,
		recordedSeq:    state.cooldownRecordedSeq,
		backoffLevel:   state.Quota.BackoffLevel,
	}
}

func (f modelQuotaFence) equal(other modelQuotaFence) bool {
	return f.nextRetryAfter.Equal(other.nextRetryAfter) &&
		f.nextRecoverAt.Equal(other.nextRecoverAt) &&
		f.updatedAt.Equal(other.updatedAt) &&
		f.recordedSeq == other.recordedSeq &&
		f.backoffLevel == other.backoffLevel
}

// quotaReleaseFence is the state a quota answer was fetched for. A release commits only
// for model states that still match it. auth.Generation is deliberately not part of it:
// it moves on every result for any model of the auth and would starve releases.
type quotaReleaseFence struct {
	registrationEpoch uint64
	credentialVersion uint64
	projectID         string
	models            map[string]modelQuotaFence // keyed by the exact ModelStates key
}

func antigravityQuotaProjectID(auth *Auth) string {
	if auth == nil || auth.Metadata == nil {
		return ""
	}
	projectID, _ := auth.Metadata["project_id"].(string)
	return strings.TrimSpace(projectID)
}

// quotaReleaseFenceFor snapshots every model state of authID that is releasable with at
// least minCooldown left, together with a clone of the auth taken under the same lock.
func (m *Manager) quotaReleaseFenceFor(authID string, now time.Time, minCooldown time.Duration) (quotaReleaseFence, *Auth, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	auth := m.auths[authID]
	if auth == nil {
		return quotaReleaseFence{}, nil, false
	}
	fence := quotaReleaseFence{
		registrationEpoch: auth.RegistrationEpoch,
		credentialVersion: auth.CredentialVersion,
		projectID:         antigravityQuotaProjectID(auth),
		models:            make(map[string]modelQuotaFence),
	}
	for key, state := range auth.ModelStates {
		if strings.TrimSpace(key) == "" {
			continue
		}
		deadline, ok := releasableModelQuotaCooldown(auth, state, now)
		if !ok || deadline.Sub(now) < minCooldown {
			continue
		}
		fence.models[key] = modelQuotaFenceOf(state)
	}
	if len(fence.models) == 0 {
		return quotaReleaseFence{}, nil, false
	}
	return fence, auth.Clone(), true
}

// releaseModelQuotaCooldowns clears the plain quota cooldown of exactly the listed
// ModelStates keys, for those still matching fence and still releasable, and publishes
// the result the way ResetQuota does. It returns the keys actually released; with none
// it changes nothing.
func (m *Manager) releaseModelQuotaCooldowns(ctx context.Context, authID string, fence quotaReleaseFence, keys []string) ([]string, error) {
	if m == nil || len(keys) == 0 {
		return nil, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	now := time.Now()

	releaseMutation := m.lockAuthMutation(authID)
	defer releaseMutation()
	m.mu.Lock()
	auth := m.auths[authID]
	if auth == nil || auth.RegistrationEpoch != fence.registrationEpoch ||
		auth.CredentialVersion != fence.credentialVersion || antigravityQuotaProjectID(auth) != fence.projectID {
		m.mu.Unlock()
		return nil, nil
	}

	var cooldownRecordsBefore []CooldownStateRecord
	trackCooldownState := m.cooldownStore != nil
	if trackCooldownState {
		cooldownRecordsBefore = m.cooldownStateRecordsForAuthLocked(auth, now)
	}
	released := make([]string, 0, len(keys))
	for _, key := range keys {
		want, okFence := fence.models[key]
		state := auth.ModelStates[key]
		if !okFence || state == nil || !modelQuotaFenceOf(state).equal(want) {
			continue
		}
		if _, ok := releasableModelQuotaCooldown(auth, state, now); !ok {
			continue
		}
		// Safe: the predicate proved every layer besides the plain quota cooldown is inactive.
		resetModelState(state, now)
		released = append(released, key)
	}
	if len(released) == 0 {
		m.mu.Unlock()
		return nil, nil
	}

	updateAggregatedAvailability(auth, now)
	if !auth.Disabled && auth.Status != StatusDisabled && !hasUnauthorizedAuthFailure(auth) &&
		!hasModelError(auth, now) && !effectiveAuthForceUntil(auth).After(now) {
		auth.LastError = nil
		auth.StatusMessage = ""
		auth.Status = StatusActive
	}
	auth.Generation++
	auth.UpdatedAt = now
	snapshot := auth.Clone()
	cooldownStateChanged := false
	if trackCooldownState {
		cooldownStateChanged = !cooldownStateRecordsEqual(cooldownRecordsBefore, m.cooldownStateRecordsForAuthLocked(auth, now))
	}
	errPersist := m.persistLocked(ctx, auth)
	m.mu.Unlock()
	releaseMutation()

	// Known window, same as ResetQuota: a RestoreCooldownStates (config reload with
	// save-cooldown-status on) between the unlock above and this save can bring the
	// released record back; the next revalidation pass releases it again.
	if cooldownStateChanged {
		m.persistCooldownStates(context.Background())
	}

	supportedModels, regEpoch := registry.GetGlobalRegistry().GetModelsAndEpochForClient(authID)
	projections := make([]registry.ClientModelProjection, 0, len(supportedModels))
	for _, sm := range supportedModels {
		if sm == nil || strings.TrimSpace(sm.ID) == "" {
			continue
		}
		projections = append(projections, m.clientModelProjectionForAuth(snapshot, sm.ID, now))
	}
	registry.GetGlobalRegistry().ApplyClientModelProjections(authID, regEpoch, snapshot.Generation, projections)
	if m.scheduler != nil {
		m.scheduler.upsertAuth(snapshot)
	}
	return released, errPersist
}

// fetchAntigravityQuotaFamilies asks retrieveUserQuotaSummary through the registered
// Antigravity executor, which sets its own User-Agent, bearer token and proxy.
func (m *Manager) fetchAntigravityQuotaFamilies(ctx context.Context, auth *Auth, projectID string) (map[string]float64, error) {
	body, errMarshal := json.Marshal(map[string]string{"project": projectID})
	if errMarshal != nil {
		return nil, errMarshal
	}
	if rt := m.roundTripperFor(auth); rt != nil {
		ctx = context.WithValue(ctx, roundTripperContextKey{}, rt)
		ctx = context.WithValue(ctx, "cliproxy.roundtripper", rt)
	}
	req, errReq := http.NewRequestWithContext(ctx, http.MethodPost, antigravityQuotaSummaryURL, bytes.NewReader(body))
	if errReq != nil {
		return nil, errReq
	}
	req.Header.Set("Content-Type", "application/json")
	resp, errDo := m.HttpRequest(ctx, auth, req)
	if errDo != nil {
		return nil, errDo
	}
	defer func() { _ = resp.Body.Close() }()
	data, errRead := io.ReadAll(io.LimitReader(resp.Body, antigravityQuotaSummaryMaxBytes))
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, antigravityQuotaStatusError(resp.StatusCode)
	}
	if errRead != nil {
		return nil, fmt.Errorf("read quota summary: %w", errRead)
	}
	return antigravityQuotaFamilies(data)
}

var errAntigravityQuotaDecode = errors.New("decode quota summary")

type antigravityQuotaStatusError int

func (e antigravityQuotaStatusError) Error() string {
	return fmt.Sprintf("quota summary returned HTTP %d", int(e))
}
func (e antigravityQuotaStatusError) StatusCode() int { return int(e) }

// antigravityQuotaErrorCategory describes err for a log line without its text: executor
// and token-endpoint errors carry upstream response bodies, which must never be logged.
func antigravityQuotaErrorCategory(err error) string {
	var withStatus interface{ StatusCode() int }
	if errors.As(err, &withStatus) && withStatus.StatusCode() > 0 {
		return fmt.Sprintf("HTTP %d", withStatus.StatusCode())
	}
	var netErr net.Error
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, errAntigravityQuotaDecode):
		return "decode error"
	case errors.As(err, &netErr):
		if netErr.Timeout() {
			return "timeout"
		}
		return "network error"
	}
	return "error"
}

// revalidateAntigravityAuthQuota checks one auth: token (refreshed through the manager when
// close to expiry), fence, summary fetch, decision, compare-and-release.
func (m *Manager) revalidateAntigravityAuthQuota(ctx context.Context, authID string, settings antigravityQuotaRevalidateSettings) {
	authCtx, cancel := context.WithTimeout(ctx, antigravityQuotaRevalidateAuthTimeout)
	defer cancel()

	current, ok := m.GetByID(authID)
	if !ok || current == nil {
		return
	}
	identity := formatAuthIdentity(current, "antigravity")
	if exp, okExp := current.AccessTokenExpirationTime(); !okExp || exp.Before(time.Now().Add(antigravityQuotaTokenMargin)) {
		if _, errRefresh := m.refreshAuthForRequest(authCtx, authID, ""); errRefresh != nil {
			log.Warnf("antigravity quota revalidate: %s: token refresh failed, cooldowns kept: %s", identity, antigravityQuotaErrorCategory(errRefresh))
			return
		}
	}
	// The fence and the auth used for the request come from one locked snapshot, taken
	// after our own refresh, so token, project and cooldowns belong together.
	fence, snapshot, ok := m.quotaReleaseFenceFor(authID, time.Now(), settings.minCooldown)
	if !ok {
		return
	}
	if fence.projectID == "" {
		log.Warnf("antigravity quota revalidate: %s: auth has no project_id, cooldowns kept", identity)
		return
	}
	families, errFetch := m.fetchAntigravityQuotaFamilies(authCtx, snapshot, fence.projectID)
	if errFetch != nil {
		log.Warnf("antigravity quota revalidate: %s: quota summary failed, cooldowns kept: %s", identity, antigravityQuotaErrorCategory(errFetch))
		return
	}

	keys := make([]string, 0, len(fence.models))
	for key := range fence.models {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	toRelease := make([]string, 0, len(keys))
	for _, key := range keys {
		family := antigravityQuotaModelFamily(key)
		left, present := families[family]
		if present && left > settings.minLeft {
			toRelease = append(toRelease, key)
			continue
		}
		cooldownLeft := time.Duration(0)
		if want := fence.models[key]; want.nextRetryAfter.After(want.nextRecoverAt) {
			cooldownLeft = time.Until(want.nextRetryAfter)
		} else {
			cooldownLeft = time.Until(want.nextRecoverAt)
		}
		if present {
			log.Debugf("antigravity quota revalidate: %s: kept %s (%s %.0f%% left, cooldown %s)", identity, key, family, left*100, cooldownLeft.Round(time.Second))
		} else {
			log.Debugf("antigravity quota revalidate: %s: kept %s (family %s not in summary, cooldown %s)", identity, key, family, cooldownLeft.Round(time.Second))
		}
	}
	if len(toRelease) == 0 {
		return
	}

	released, errRelease := m.releaseModelQuotaCooldowns(context.WithoutCancel(ctx), authID, fence, toRelease)
	if len(released) < len(toRelease) {
		log.Debugf("antigravity quota revalidate: %s: %d of %d cooldowns changed during the check, kept", identity, len(toRelease)-len(released), len(toRelease))
	}
	if len(released) > 0 {
		seen := make(map[string]struct{})
		remaining := make([]string, 0, 2)
		for _, key := range released {
			family := antigravityQuotaModelFamily(key)
			if _, dup := seen[family]; dup {
				continue
			}
			seen[family] = struct{}{}
			remaining = append(remaining, fmt.Sprintf("%s %.0f%%", family, families[family]*100))
		}
		log.Infof("antigravity quota revalidate: released %s models=[%s] remaining=%s", identity, strings.Join(released, " "), strings.Join(remaining, ", "))
	}
	if errRelease != nil {
		log.Warnf("antigravity quota revalidate: %s: persisting released cooldowns failed: %s", identity, antigravityQuotaErrorCategory(errRelease))
	}
}

// antigravityQuotaCandidates lists Antigravity auths with at least one releasable quota
// cooldown that has minCooldown or more left.
func (m *Manager) antigravityQuotaCandidates(now time.Time, minCooldown time.Duration) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	ids := make([]string, 0)
	for id, auth := range m.auths {
		if auth == nil || !strings.EqualFold(strings.TrimSpace(auth.Provider), "antigravity") {
			continue
		}
		for _, state := range auth.ModelStates {
			if deadline, ok := releasableModelQuotaCooldown(auth, state, now); ok && deadline.Sub(now) >= minCooldown {
				ids = append(ids, id)
				break
			}
		}
	}
	sort.Strings(ids)
	return ids
}

// revalidateAntigravityQuotaCooldowns runs one pass over all candidates, sequentially. It
// returns false without doing anything when another pass of this Manager is running.
func (m *Manager) revalidateAntigravityQuotaCooldowns(ctx context.Context, settings antigravityQuotaRevalidateSettings) bool {
	if !m.antigravityQuotaRevalidateMu.TryLock() {
		return false
	}
	defer m.antigravityQuotaRevalidateMu.Unlock()
	for _, authID := range m.antigravityQuotaCandidates(time.Now(), settings.minCooldown) {
		if ctx.Err() != nil {
			break
		}
		m.revalidateAntigravityAuthQuota(ctx, authID, settings)
	}
	return true
}

// runAntigravityQuotaRevalidate is the worker started with the auto-refresh loop; it stops
// with the same context. Config is re-read every tick, so enabling, disabling or changing
// the interval takes effect within one tick.
func (m *Manager) runAntigravityQuotaRevalidate(ctx context.Context) {
	m.runAntigravityQuotaRevalidateEvery(ctx, antigravityQuotaRevalidateTick)
}

func (m *Manager) runAntigravityQuotaRevalidateEvery(ctx context.Context, tick time.Duration) {
	if m == nil || ctx == nil {
		return
	}
	ticker := time.NewTicker(tick)
	defer ticker.Stop()
	var lastRun time.Time
	lastWarning := ""
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		if m.HomeEnabled() {
			continue
		}
		settings, warning := antigravityQuotaRevalidateSettingsFromConfig(m.runtimeConfigSnapshot())
		if warning != lastWarning {
			if warning != "" {
				log.Warnf("antigravity quota revalidate: %s", warning)
			}
			lastWarning = warning
		}
		if settings.interval <= 0 {
			lastRun = time.Time{}
			continue
		}
		now := time.Now()
		if !lastRun.IsZero() && now.Sub(lastRun) < settings.interval {
			continue
		}
		if m.revalidateAntigravityQuotaCooldowns(ctx, settings) {
			lastRun = now
		}
	}
}

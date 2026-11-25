// Package gemini provides request translation functionality for Gemini to Antigravity API compatibility.
// It handles parsing and transforming Gemini API requests into Antigravity API format,
// extracting model information, system instructions, message contents, and tool declarations.
// The package performs JSON data transformation to ensure compatibility
// between Gemini API format and Antigravity API's expected format.
package gemini

import (
	"bytes"
	"context"
	"fmt"
	"strings"

	"github.com/MoeForge/moego/ojson"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v8/internal/translator/common"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/util"
)

type convertAntigravityGeminiParams struct {
	SawResponse     bool
	SawFinishReason bool
	ModelVersion    string
	ResponseID      string
	UsageMetadata   []byte
}

// observe records the metadata required to synthesize a faithful terminal chunk
// when upstream never emits finishReason.
func (p *convertAntigravityGeminiParams) observe(rawJSON []byte) {
	if !p.SawResponse {
		p.SawResponse = hasAntigravityResponsePayload(rawJSON)
	}
	if !p.SawFinishReason {
		for _, path := range []string{"response.candidates", "candidates"} {
			for _, candidate := range ojson.GetBytesNoCopy(rawJSON, path).Array() {
				if candidate.Get("finishReason").String() != "" {
					p.SawFinishReason = true
					break
				}
			}
			if p.SawFinishReason {
				break
			}
		}
	}
	if p.ModelVersion == "" {
		p.ModelVersion = firstStringValue(rawJSON, "response.modelVersion", "modelVersion")
	}
	if p.ResponseID == "" {
		p.ResponseID = firstStringValue(rawJSON, "response.responseId", "responseId")
	}
	// Keep the latest usage snapshot. FilterSSEUsageMetadata renames non-terminal
	// usage to cpaUsageMetadata, so both spellings must be tracked.
	for _, path := range []string{"response.usageMetadata", "response.cpaUsageMetadata", "usageMetadata", "cpaUsageMetadata"} {
		if usage := ojson.GetBytesNoCopy(rawJSON, path); usage.Exists() {
			p.UsageMetadata = append(p.UsageMetadata[:0], usage.Raw...)
			break
		}
	}
}

// syntheticTerminalChunk mirrors the terminal chunk shape observed from upstream:
// a model-role candidate carrying an empty text part, finishReason, and the last
// known usage metadata. Fields are set in upstream key order
// (candidates, usageMetadata, modelVersion, responseId).
func (p *convertAntigravityGeminiParams) syntheticTerminalChunk() []byte {
	chunkDoc := ojson.NewDoc([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}]}`))
	if len(p.UsageMetadata) > 0 {
		_ = chunkDoc.SetRawBytes("usageMetadata", p.UsageMetadata)
	}
	if p.ModelVersion != "" {
		_ = chunkDoc.Set("modelVersion", p.ModelVersion)
	}
	if p.ResponseID != "" {
		_ = chunkDoc.Set("responseId", p.ResponseID)
	}
	return chunkDoc.Bytes()
}

// hasAntigravityResponsePayload reports whether a chunk carries actual generated
// content or token accounting. Presence alone is not enough: an envelope such as
// `{}`, `{"response":{}}` or `{"response":{"candidates":[]}}` must not count as a
// started stream, otherwise a keepalive or malformed event would be enough to
// finalize a stream that produced nothing.
func hasAntigravityResponsePayload(rawJSON []byte) bool {
	for _, path := range []string{"response.candidates", "candidates"} {
		if candidates := ojson.GetBytesNoCopy(rawJSON, path); candidates.IsArray() && len(candidates.Array()) > 0 {
			return true
		}
	}
	// FilterSSEUsageMetadata renames non-terminal usage to cpaUsageMetadata before
	// the translator sees it, so both spellings must be accepted.
	for _, path := range []string{
		"response.usageMetadata", "response.cpaUsageMetadata",
		"usageMetadata", "cpaUsageMetadata",
	} {
		if usage := ojson.GetBytesNoCopy(rawJSON, path); usage.IsObject() && len(usage.Map()) > 0 {
			return true
		}
	}
	return false
}

func firstStringValue(rawJSON []byte, paths ...string) string {
	for _, path := range paths {
		if value := ojson.GetBytesNoCopy(rawJSON, path).String(); value != "" {
			return strings.Clone(value)
		}
	}
	return ""
}

// ConvertAntigravityResponseToGemini parses and transforms a Antigravity API request into Gemini API format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the Gemini API.
// The function performs the following transformations:
// 1. Extracts the response data from the request
// 2. Handles alternative response formats
// 3. Processes array responses by extracting individual response objects
// 4. Synthesizes finishReason: "STOP" on [DONE] if omitted by upstream
//
// Parameters:
//   - ctx: The context for the request, used for cancellation and timeout handling
//   - modelName: The name of the model to use for the request (unused in current implementation)
//   - rawJSON: The raw JSON request data from the Antigravity API
//   - param: A pointer to a parameter object for the conversion
//
// Returns:
//   - [][]byte: The transformed response data in Gemini API format.
func ConvertAntigravityResponseToGemini(ctx context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) [][]byte {
	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
	}

	var state *convertAntigravityGeminiParams
	if param != nil {
		if *param == nil {
			*param = &convertAntigravityGeminiParams{}
		}
		if s, ok := (*param).(*convertAntigravityGeminiParams); ok {
			state = s
		}
	}

	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		// Never finalize a stream that produced no response at all: an empty
		// upstream body must stay detectable as a failure rather than be reported
		// as a successful empty completion.
		if state == nil || !state.SawResponse || state.SawFinishReason {
			return [][]byte{}
		}
		state.SawFinishReason = true
		finalChunk := state.syntheticTerminalChunk()
		if alt, ok := ctx.Value("alt").(string); ok && alt != "" {
			finalChunk, _ = ojson.SetRawBytes([]byte("[]"), "-1", finalChunk)
		}
		return [][]byte{finalChunk}
	}

	if state != nil {
		state.observe(rawJSON)
	}

	if alt, ok := ctx.Value("alt").(string); ok {
		var chunk []byte
		if alt == "" {
			responseResult := util.GetGJSONBytesNoCopy(rawJSON, "response")
			if responseResult.Exists() {
				if responseResult.Index > 0 && responseResult.Index+len(responseResult.Raw) <= len(rawJSON) {
					chunk = rawJSON[responseResult.Index : responseResult.Index+len(responseResult.Raw)]
				} else {
					chunk = []byte(responseResult.Raw)
				}
				// Non-terminal usage arrives as cpaUsageMetadata (FilterSSEUsageMetadata);
				// a real usageMetadata is the terminal count and must not be dropped with the parts.
				hasUsage := responseResult.Get("usageMetadata").Exists()
				chunk = restoreUsageMetadata(chunk)
				chunk = restoreGeminiFunctionNames(chunk, originalRequestRawJSON)
				var onlyEmptyThoughts bool
				chunk, onlyEmptyThoughts = stripEmptyThoughtParts(chunk)
				if onlyEmptyThoughts && !hasUsage {
					return [][]byte{}
				}
			}
		} else {
			chunkDoc := ojson.NewDoc([]byte("[]"))
			responseResult := util.ParseGJSONBytesNoCopy(rawJSON)
			if responseResult.IsArray() {
				keptItems := 0
				responseResultItems := responseResult.Array()
				for i := 0; i < len(responseResultItems); i++ {
					responseResultItem := responseResultItems[i]
					if response := responseResultItem.Get("response"); response.Exists() {
						hasUsage := response.Get("usageMetadata").Exists()
						item, onlyEmptyThoughts := stripEmptyThoughtParts([]byte(response.Raw))
						if !onlyEmptyThoughts || hasUsage {
							_ = chunkDoc.SetRawBytes("-1", item)
							keptItems++
						}
					}
				}
				// Nothing survived filtering, so deliver no chunk at all instead of
				// an empty array that clients read as an empty response.
				if keptItems == 0 {
					return [][]byte{}
				}
			}
			chunk = chunkDoc.Bytes()
		}
		return [][]byte{chunk}
	}
	return [][]byte{}
}

// ConvertAntigravityResponseToGeminiNonStream converts a non-streaming Antigravity request to a non-streaming Gemini response.
// This function processes the complete Antigravity request and transforms it into a single Gemini-compatible
// JSON response. It extracts the response data from the request and returns it in the expected format.
//
// Parameters:
//   - ctx: The context for the request, used for cancellation and timeout handling
//   - modelName: The name of the model being used for the response (unused in current implementation)
//   - rawJSON: The raw JSON request data from the Antigravity API
//   - param: A pointer to a parameter object for the conversion (unused in current implementation)
//
// Returns:
//   - []byte: A Gemini-compatible JSON response containing the response data.
var cpaUsageMetadataMarker = []byte(`"cpaUsageMetadata"`)

func ConvertAntigravityResponseToGeminiNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) []byte {
	var chunk []byte
	responseResult := util.GetGJSONBytesNoCopy(rawJSON, "response")
	if responseResult.Exists() {
		if responseResult.Index > 0 && responseResult.Index+len(responseResult.Raw) <= len(rawJSON) {
			chunk = rawJSON[responseResult.Index : responseResult.Index+len(responseResult.Raw)]
		} else {
			chunk = []byte(responseResult.Raw)
		}
		chunk = restoreUsageMetadata(chunk)
		chunk = restoreGeminiFunctionNames(chunk, originalRequestRawJSON)
	} else {
		chunk = restoreGeminiFunctionNames(rawJSON, originalRequestRawJSON)
	}
	chunk, _ = stripEmptyThoughtParts(chunk)
	candidates := util.GetGJSONBytesNoCopy(chunk, "candidates")
	if candidates.IsArray() {
		// Default every candidate, not just the first one, so a multi-candidate
		// response cannot leave some of them unterminated.
		var chunkDoc *ojson.Doc
		idx := 0
		candidates.ForEach(func(_, candidate ojson.Result) bool {
			if candidate.Get("finishReason").String() == "" {
				if chunkDoc == nil {
					chunkDoc = ojson.NewDoc(chunk)
				}
				_ = chunkDoc.Set(fmt.Sprintf("candidates.%d.finishReason", idx), "STOP")
			}
			idx++
			return true
		})
		if chunkDoc != nil {
			chunk = chunkDoc.Bytes()
		}
	}
	return chunk
}

var thoughtFieldMarker = []byte(`"thought"`)

// stripEmptyThoughtParts drops parts that carry nothing but `"thought":true` and an
// empty text. Antigravity flash-lite models emit such parts ahead of every answer even
// when thoughts are not requested; Gemini API clients such as the Antigravity CLI then
// read the response as empty (its search_web summary comes back blank). Parts with a
// thoughtSignature or any other field are kept.
//
// The second result reports that the chunk consisted solely of such parts and carries
// nothing else worth delivering (no finishReason, no grounding metadata), so a streaming
// caller can skip it entirely.
func stripEmptyThoughtParts(chunk []byte) ([]byte, bool) {
	if !bytes.Contains(chunk, thoughtFieldMarker) {
		return chunk, false
	}
	candidates := ojson.GetBytesNoCopy(chunk, "candidates")
	if !candidates.IsArray() {
		return chunk, false
	}
	var chunkDoc *ojson.Doc
	removedAny := false
	nothingLeft := true
	for candidateIndex, candidate := range candidates.Array() {
		parts := candidate.Get("content.parts")
		kept := make([]string, 0, len(parts.Array()))
		removed := false
		for _, part := range parts.Array() {
			if isEmptyThoughtPart(part) {
				removed = true
				continue
			}
			kept = append(kept, part.Raw)
		}
		if removed {
			removedAny = true
			if chunkDoc == nil {
				chunkDoc = ojson.NewDoc(chunk)
			}
			_ = chunkDoc.SetRaw(fmt.Sprintf("candidates.%d.content.parts", candidateIndex), "["+strings.Join(kept, ",")+"]")
		}
		if len(kept) > 0 || candidate.Get("finishReason").String() != "" || candidate.Get("groundingMetadata").Exists() {
			nothingLeft = false
		}
	}
	if chunkDoc == nil {
		return chunk, false
	}
	return chunkDoc.Bytes(), removedAny && nothingLeft
}

// isEmptyThoughtPart matches only `thought` as a JSON true and `text` absent or an empty
// JSON string: ojson.Result.Bool/String would also accept "true", 1 and null.
func isEmptyThoughtPart(part ojson.Result) bool {
	if !part.IsObject() || part.Get("thought").Type != ojson.True {
		return false
	}
	for key, value := range part.Map() {
		switch key {
		case "thought":
		case "text":
			if value.Type != ojson.String || value.Str != "" {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func restoreGeminiFunctionNames(chunk, originalRequestRawJSON []byte) []byte {
	nameMap := util.DisambiguatedToolNameMap(originalRequestRawJSON)
	if len(nameMap) == 0 {
		return chunk
	}
	var chunkDoc *ojson.Doc
	candidates := ojson.GetBytesNoCopy(chunk, "candidates")
	for candidateIndex, candidate := range candidates.Array() {
		for partIndex, part := range candidate.Get("content.parts").Array() {
			for _, field := range []string{"functionCall", "functionResponse", "function_call", "function_response"} {
				nameResult := part.Get(field + ".name")
				name := nameResult.String()
				if name == "" {
					continue
				}
				restoredName := util.RestoreSanitizedToolName(nameMap, name)
				if nameResult.Type == ojson.String && restoredName == name {
					continue
				}
				if chunkDoc == nil {
					chunkDoc = ojson.NewDoc(chunk)
				}
				path := fmt.Sprintf("candidates.%d.content.parts.%d.%s.name", candidateIndex, partIndex, field)
				_ = chunkDoc.Set(path, restoredName)
			}
		}
	}
	if chunkDoc != nil {
		return chunkDoc.Bytes()
	}
	return chunk
}

func GeminiTokenCount(ctx context.Context, count int64) []byte {
	return translatorcommon.GeminiTokenCountJSON(count)
}

// restoreUsageMetadata renames cpaUsageMetadata back to usageMetadata.
// The executor renames usageMetadata to cpaUsageMetadata in non-terminal chunks
// to preserve usage data while hiding it from clients that don't expect it.
// When returning standard Gemini API format, we must restore the original name.
func restoreUsageMetadata(chunk []byte) []byte {
	if !bytes.Contains(chunk, cpaUsageMetadataMarker) {
		return chunk
	}
	if cpaUsage := util.GetGJSONBytesNoCopy(chunk, "cpaUsageMetadata"); cpaUsage.Exists() {
		if cpaUsage.Index > 0 && cpaUsage.Index <= len(chunk) {
			searchStart := cpaUsage.Index - 36
			if searchStart < 0 {
				searchStart = 0
			}
			if rel := bytes.LastIndex(chunk[searchStart:cpaUsage.Index], cpaUsageMetadataMarker); rel >= 0 {
				keyPos := searchStart + rel + 1
				out := make([]byte, 0, len(chunk)-3)
				out = append(out, chunk[:keyPos]...)
				out = append(out, "usageMetadata"...)
				out = append(out, chunk[keyPos+len("cpaUsageMetadata"):]...)
				return out
			}
		}
		chunkDoc := ojson.NewDoc(chunk)
		_ = chunkDoc.SetRaw("usageMetadata", cpaUsage.Raw)
		_ = chunkDoc.Delete("cpaUsageMetadata")
		chunk = chunkDoc.Bytes()
	}
	return chunk
}

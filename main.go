package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/charmbracelet/huh"
)

// --- Config ---

type Config struct {
	BaseURL   string `json:"base_url"`
	APIKey    string `json:"api_key"`
	LastModel string `json:"last_model"`
}

func getConfigPath() string {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, "claude-launcher", "config.json")
}

// --- Anthropic → OpenAI Conversion ---

// convertAnthropicToOpenAI converts an Anthropic MessagesRequest to an OpenAI ChatCompletionRequest
func convertAnthropicToOpenAI(req MessagesRequest) ChatCompletionRequest {
	var messages []OAMessage

	// 1. System prompt
	if req.System != nil {
		sysStr := extractSystemText(req.System)
		if sysStr != "" {
			messages = append(messages, OAMessage{Role: "system", Content: sysStr})
		}
	}

	// 2. Messages
	for _, msg := range req.Messages {
		messages = append(messages, convertMessageParam(msg)...)
	}

	// 3. Tools
	var tools []OAToolDef
	for _, t := range req.Tools {
		if strings.HasPrefix(t.Type, "web_search") {
			continue // skip built-in web search tools
		}
		tools = append(tools, OAToolDef{
			Type: "function",
			Function: OAToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}

	oaReq := ChatCompletionRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   req.Stream,
		Tools:    tools,
	}

	if req.MaxTokens > 0 {
		oaReq.MaxTokens = &req.MaxTokens
	}
	if req.Temperature != nil {
		oaReq.Temperature = req.Temperature
	}
	if req.TopP != nil {
		oaReq.TopP = req.TopP
	}
	if len(req.StopSequences) > 0 {
		oaReq.Stop = req.StopSequences
	}
	if req.Stream {
		oaReq.StreamOptions = &StreamOptions{IncludeUsage: true}
	}

	return oaReq
}

// extractSystemText extracts text from system prompt (string or content block array)
func extractSystemText(system any) string {
	switch sys := system.(type) {
	case string:
		return sys
	case []any:
		var sb strings.Builder
		for _, block := range sys {
			if blockMap, ok := block.(map[string]any); ok {
				if blockMap["type"] == "text" {
					if text, ok := blockMap["text"].(string); ok {
						sb.WriteString(text)
					}
				}
			}
		}
		return sb.String()
	}
	return ""
}

// convertMessageParam converts an Anthropic MessageParam to OpenAI message(s)
func convertMessageParam(msg MessageParam) []OAMessage {
	var messages []OAMessage
	role := msg.Role

	var textParts strings.Builder
	var toolCalls []OAToolCall
	var reasoning string

	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			if block.Text != nil {
				textParts.WriteString(*block.Text)
			}

		case "thinking":
			if block.Thinking != nil {
				reasoning = *block.Thinking
			}

		case "tool_use":
			inputStr := "{}"
			if len(block.Input) > 0 {
				inputStr = string(block.Input)
			}
			toolCalls = append(toolCalls, OAToolCall{
				ID:    block.ID,
				Type:  "function",
				Index: len(toolCalls),
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{
					Name:      block.Name,
					Arguments: inputStr,
				},
			})

		case "tool_result":
			resultContent := extractToolResultContent(block.Content)
			messages = append(messages, OAMessage{
				Role:       "tool",
				Content:    resultContent,
				ToolCallID: block.ToolUseID,
			})
		}
	}

	// Build the main message if there's text, tool calls, or reasoning
	if textParts.Len() > 0 || len(toolCalls) > 0 || reasoning != "" {
		m := OAMessage{
			Role:      role,
			Content:   textParts.String(),
			ToolCalls: toolCalls,
			Reasoning: reasoning,
		}
		if role == "user" && len(messages) > 0 {
			// For user messages with tool_results, tool messages must come first
			// so they immediately follow the assistant's tool_calls in OpenAI format.
			messages = append(messages, m)
		} else {
			// For assistant messages, prepend (text + tool_calls come before any tool results)
			messages = append([]OAMessage{m}, messages...)
		}
	}

	return messages
}

// extractToolResultContent extracts text from tool_result content
func extractToolResultContent(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var sb strings.Builder
		for _, item := range c {
			if m, ok := item.(map[string]any); ok {
				if m["type"] == "text" {
					if text, ok := m["text"].(string); ok {
						sb.WriteString(text)
					}
				}
			}
		}
		return sb.String()
	}
	return ""
}

// --- OpenAI → Anthropic Conversion (Non-streaming) ---

func convertOpenAIToAnthropic(comp ChatCompletion, model string) MessagesResponse {
	var content []ContentBlock

	if len(comp.Choices) > 0 {
		choice := comp.Choices[0]
		msg := choice.Message

		// Reasoning/thinking (support both "reasoning" and "reasoning_content" fields)
		reasoning := msg.Reasoning
		if reasoning == "" {
			reasoning = msg.ReasoningContent
		}
		if reasoning != "" {
			content = append(content, ContentBlock{
				Type:     "thinking",
				Thinking: ptr(reasoning),
			})
		}

		// Text content
		if textStr, ok := msg.Content.(string); ok && textStr != "" {
			content = append(content, ContentBlock{
				Type: "text",
				Text: ptr(textStr),
			})
		}

		// Tool calls
		for _, tc := range msg.ToolCalls {
			content = append(content, ContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			})
		}
	}

	// Ensure at least one content block
	if len(content) == 0 {
		content = append(content, ContentBlock{
			Type: "text",
			Text: ptr(""),
		})
	}

	stopReason := "end_turn"
	if len(comp.Choices) > 0 && comp.Choices[0].FinishReason != nil {
		stopReason = mapOAStopReason(*comp.Choices[0].FinishReason)
	}

	return MessagesResponse{
		ID:      GenerateMessageID(),
		Type:    "message",
		Role:    "assistant",
		Model:   model,
		Content: content,
		StopReason: stopReason,
		Usage: Usage{
			InputTokens:  comp.Usage.PromptTokens,
			OutputTokens: comp.Usage.CompletionTokens,
		},
	}
}

func mapOAStopReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}

// --- OpenAI → Anthropic Conversion (Streaming) ---

type streamConverter struct {
	id              string
	model           string
	firstWrite      bool
	contentIndex    int
	thinkingStarted bool
	thinkingDone    bool
	textStarted     bool
	toolCallsSent   map[string]bool
	// Accumulate partial tool call data
	toolCallAccum map[int]*OAToolCall
}

func newStreamConverter(model string) *streamConverter {
	return &streamConverter{
		id:            GenerateMessageID(),
		model:         model,
		firstWrite:    true,
		toolCallsSent: make(map[string]bool),
		toolCallAccum: make(map[int]*OAToolCall),
	}
}

type streamEvent struct {
	Event string
	Data  any
}

func (c *streamConverter) processChunk(chunk ChatCompletionChunk) []streamEvent {
	var events []streamEvent

	if c.firstWrite {
		c.firstWrite = false
		events = append(events, streamEvent{
			Event: "message_start",
			Data: MessageStartEvent{
				Type: "message_start",
				Message: MessagesResponse{
					ID:      c.id,
					Type:    "message",
					Role:    "assistant",
					Model:   c.model,
					Content: []ContentBlock{},
					Usage:   Usage{InputTokens: 0, OutputTokens: 0},
				},
			},
		})
	}

	if len(chunk.Choices) == 0 {
		// Usage-only chunk at the end
		if chunk.Usage != nil {
			events = append(events, c.finishEvents("end_turn", chunk.Usage)...)
		}
		return events
	}

	delta := chunk.Choices[0].Delta

	// Handle reasoning/thinking (support both "reasoning" and "reasoning_content" fields)
	reasoningText := delta.Reasoning
	if reasoningText == "" {
		reasoningText = delta.ReasoningContent
	}
	if reasoningText != "" {
		if !c.thinkingStarted {
			c.thinkingStarted = true
			events = append(events, streamEvent{
				Event: "content_block_start",
				Data: ContentBlockStartEvent{
					Type:  "content_block_start",
					Index: c.contentIndex,
					ContentBlock: ContentBlock{
						Type:     "thinking",
						Thinking: ptr(""),
					},
				},
			})
		}
		events = append(events, streamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type:     "thinking_delta",
					Thinking: reasoningText,
				},
			},
		})
	}

	// Handle text content
	if contentStr, ok := delta.Content.(string); ok && contentStr != "" {
		// Close thinking block if transitioning to text
		if c.thinkingStarted && !c.thinkingDone {
			c.thinkingDone = true
			events = append(events, streamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
		}

		if !c.textStarted {
			c.textStarted = true
			events = append(events, streamEvent{
				Event: "content_block_start",
				Data: ContentBlockStartEvent{
					Type:  "content_block_start",
					Index: c.contentIndex,
					ContentBlock: ContentBlock{
						Type: "text",
						Text: ptr(""),
					},
				},
			})
		}
		events = append(events, streamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type: "text_delta",
					Text: contentStr,
				},
			},
		})
	}

	// Handle tool calls (accumulated incrementally)
	for _, tc := range delta.ToolCalls {
		idx := tc.Index
		if _, exists := c.toolCallAccum[idx]; !exists {
			c.toolCallAccum[idx] = &OAToolCall{
				ID:   tc.ID,
				Type: tc.Type,
			}
			c.toolCallAccum[idx].Function.Name = tc.Function.Name
		}
		accum := c.toolCallAccum[idx]
		if tc.ID != "" {
			accum.ID = tc.ID
		}
		if tc.Function.Name != "" {
			accum.Function.Name = tc.Function.Name
		}
		accum.Function.Arguments += tc.Function.Arguments
	}

	// Handle finish
	if chunk.Choices[0].FinishReason != nil {
		reason := *chunk.Choices[0].FinishReason

		// Emit accumulated tool calls
		if reason == "tool_calls" || len(c.toolCallAccum) > 0 {
			events = append(events, c.emitToolCalls()...)
		}

		var usage *OAUsage
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		events = append(events, c.finishEvents(reason, usage)...)
	}

	return events
}

func (c *streamConverter) emitToolCalls() []streamEvent {
	var events []streamEvent

	// Close thinking if still open
	if c.thinkingStarted && !c.thinkingDone {
		c.thinkingDone = true
		events = append(events, streamEvent{
			Event: "content_block_stop",
			Data: ContentBlockStopEvent{
				Type:  "content_block_stop",
				Index: c.contentIndex,
			},
		})
		c.contentIndex++
	}

	// Close text if still open
	if c.textStarted {
		events = append(events, streamEvent{
			Event: "content_block_stop",
			Data: ContentBlockStopEvent{
				Type:  "content_block_stop",
				Index: c.contentIndex,
			},
		})
		c.contentIndex++
		c.textStarted = false
	}

	// Emit each accumulated tool call
	for i := 0; i < len(c.toolCallAccum); i++ {
		tc := c.toolCallAccum[i]
		if tc == nil || c.toolCallsSent[tc.ID] {
			continue
		}

		argsJSON := tc.Function.Arguments
		if argsJSON == "" {
			argsJSON = "{}"
		}

		events = append(events, streamEvent{
			Event: "content_block_start",
			Data: ContentBlockStartEvent{
				Type:  "content_block_start",
				Index: c.contentIndex,
				ContentBlock: ContentBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: json.RawMessage("{}"),
				},
			},
		})

		events = append(events, streamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type:        "input_json_delta",
					PartialJSON: argsJSON,
				},
			},
		})

		events = append(events, streamEvent{
			Event: "content_block_stop",
			Data: ContentBlockStopEvent{
				Type:  "content_block_stop",
				Index: c.contentIndex,
			},
		})

		c.toolCallsSent[tc.ID] = true
		c.contentIndex++
	}

	return events
}

func (c *streamConverter) finishEvents(reason string, usage *OAUsage) []streamEvent {
	var events []streamEvent

	// Close any open blocks
	if c.textStarted {
		events = append(events, streamEvent{
			Event: "content_block_stop",
			Data: ContentBlockStopEvent{
				Type:  "content_block_stop",
				Index: c.contentIndex,
			},
		})
		c.textStarted = false
	} else if c.thinkingStarted && !c.thinkingDone {
		c.thinkingDone = true
		events = append(events, streamEvent{
			Event: "content_block_stop",
			Data: ContentBlockStopEvent{
				Type:  "content_block_stop",
				Index: c.contentIndex,
			},
		})
	}

	stopReason := mapOAStopReason(reason)

	var inputTokens, outputTokens int
	if usage != nil {
		inputTokens = usage.PromptTokens
		outputTokens = usage.CompletionTokens
	}

	events = append(events, streamEvent{
		Event: "message_delta",
		Data: MessageDeltaEvent{
			Type: "message_delta",
			Delta: MessageDelta{
				StopReason: stopReason,
			},
			Usage: DeltaUsage{
				InputTokens:  inputTokens,
				OutputTokens: outputTokens,
			},
		},
	})

	events = append(events, streamEvent{
		Event: "message_stop",
		Data: MessageStopEvent{
			Type: "message_stop",
		},
	})

	return events
}

// --- HTTP Handler ---

func handleAnthropicToOpenAI(w http.ResponseWriter, r *http.Request, cfg *Config) {
	var antReq MessagesRequest
	if err := json.NewDecoder(r.Body).Decode(&antReq); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(NewError(http.StatusBadRequest, "JSON Decode Error: "+err.Error()))
		return
	}

	oaReq := convertAnthropicToOpenAI(antReq)
	payload, _ := json.Marshal(oaReq)
	targetURL := strings.TrimSuffix(cfg.BaseURL, "/") + "/chat/completions"

	proxyReq, _ := http.NewRequest("POST", targetURL, bytes.NewBuffer(payload))
	proxyReq.Header.Set("Content-Type", "application/json")
	proxyReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(NewError(http.StatusInternalServerError, err.Error()))
		return
	}
	defer resp.Body.Close()

	// If upstream returned an error, read the body and convert to Anthropic error format
	if resp.StatusCode >= 400 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		errMsg := string(bodyBytes)
		// Try to extract error message from OpenAI-style error response
		var oaErr OAErrorResponse
		if json.Unmarshal(bodyBytes, &oaErr) == nil && oaErr.Error.Message != "" {
			errMsg = oaErr.Error.Message
		}
		log.Printf("[proxy] upstream error %d: %s", resp.StatusCode, errMsg)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		json.NewEncoder(w).Encode(NewError(resp.StatusCode, errMsg))
		return
	}

	if antReq.Stream {
		handleStream(w, resp, antReq.Model)
	} else {
		handleNormal(w, resp, antReq.Model)
	}
}

func handleStream(w http.ResponseWriter, resp *http.Response, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	conv := newStreamConverter(model)
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			// EOF or error - emit finish if we haven't yet
			break
		}
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		events := conv.processChunk(chunk)
		for _, ev := range events {
			evData, _ := json.Marshal(ev.Data)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Event, evData)
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}

	// If stream ended without a finish_reason, send closing events
	if conv.firstWrite {
		// Never got any data - send minimal response
		events := conv.processChunk(ChatCompletionChunk{})
		for _, ev := range events {
			evData, _ := json.Marshal(ev.Data)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Event, evData)
		}
	}
	// Ensure message_stop is always sent
	if !conv.firstWrite && len(conv.toolCallsSent) == 0 && conv.textStarted {
		// Stream ended abruptly, emit finish
		events := conv.finishEvents("stop", nil)
		for _, ev := range events {
			evData, _ := json.Marshal(ev.Data)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Event, evData)
		}
	}

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func handleNormal(w http.ResponseWriter, resp *http.Response, model string) {
	var comp ChatCompletion
	if err := json.NewDecoder(resp.Body).Decode(&comp); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(NewError(http.StatusInternalServerError, "Failed to decode upstream response"))
		return
	}

	antResp := convertOpenAIToAnthropic(comp, model)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(antResp)
}

// --- Proxy & Launch ---

func startProxy(cfg *Config) string {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatal(err)
	}
	addr := fmt.Sprintf("http://127.0.0.1:%d", ln.Addr().(*net.TCPAddr).Port)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		handleAnthropicToOpenAI(w, r, cfg)
	})
	go http.Serve(ln, mux)
	return addr
}

// findClaudePath searches for the claude binary in PATH,
// then falls back to ~/.claude/local/claude
func findClaudePath() (string, error) {
	if p, err := exec.LookPath("claude"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	name := "claude"
	if runtime.GOOS == "windows" {
		name = "claude.exe"
	}
	fallback := filepath.Join(home, ".claude", "local", name)
	if _, err := os.Stat(fallback); err != nil {
		return "", fmt.Errorf("claude is not installed, install from https://code.claude.com/docs/en/quickstart")
	}
	return fallback, nil
}

func launchClaude(cfg *Config, model, proxyAddr string, extraArgs []string) {
	claudePath, err := findClaudePath()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	env := append(os.Environ(),
		"ANTHROPIC_BASE_URL="+proxyAddr,
		"ANTHROPIC_API_KEY=sk-ant-proxy-fixed-123",
		"CLAUDE_CODE_ATTRIBUTION_HEADER=0",
		"ANTHROPIC_DEFAULT_OPUS_MODEL="+model,
		"ANTHROPIC_DEFAULT_SONNET_MODEL="+model,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL="+model,
		"CLAUDE_CODE_SUBAGENT_MODEL="+model,
	)

	args := []string{"--model", model}
	args = append(args, extraArgs...)

	cmd := exec.Command(claudePath, args...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.Run()
}

// --- Main ---

func main() {
	// Collect args to pass through to claude (--resume, -c, etc.)
	var claudeArgs []string
	for i := 1; i < len(os.Args); i++ {
		arg := os.Args[i]
		switch {
		case arg == "--resume" || arg == "-c" || arg == "--continue":
			claudeArgs = append(claudeArgs, arg)
		case strings.HasPrefix(arg, "--resume="):
			claudeArgs = append(claudeArgs, arg)
		default:
			// Skip unknown flags
		}
	}

	config, _ := loadConfig()
	if config == nil || config.BaseURL == "" {
		config = &Config{}
		huh.NewForm(huh.NewGroup(
			huh.NewInput().Title("OpenAI Base URL").Value(&config.BaseURL),
			huh.NewInput().Title("API Key").Value(&config.APIKey),
		)).Run()
		saveConfig(config)
	}

	models, _ := fetchModels(config)
	var selectedModel string
	huh.NewSelect[string]().Title("Select Model").Options(huh.NewOptions(models...)...).Value(&selectedModel).Run()

	proxyAddr := startProxy(config)
	fmt.Printf("Proxy running at %s\n", proxyAddr)
	launchClaude(config, selectedModel, proxyAddr, claudeArgs)
}

func loadConfig() (*Config, error) {
	data, err := os.ReadFile(getConfigPath())
	if err != nil {
		return nil, err
	}
	var cfg Config
	err = json.Unmarshal(data, &cfg)
	return &cfg, err
}

func saveConfig(cfg *Config) {
	p := getConfigPath()
	_ = os.MkdirAll(filepath.Dir(p), 0755)
	data, _ := json.MarshalIndent(cfg, "", "  ")
	_ = os.WriteFile(p, data, 0644)
}

func fetchModels(cfg *Config) ([]string, error) {
	url := strings.TrimSuffix(cfg.BaseURL, "/") + "/models"
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return []string{"QAQ", "或许你的api有问题。。。？"}, nil
	}
	defer resp.Body.Close()
	var mResp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&mResp)
	var res []string
	for _, m := range mResp.Data {
		res = append(res, m.ID)
	}
	return res, nil
}

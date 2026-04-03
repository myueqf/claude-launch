package main

import "encoding/json"

// OA-prefixed types to avoid collision with Anthropic types

type OAError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   any     `json:"param"`
	Code    *string `json:"code"`
}

type OAErrorResponse struct {
	Error OAError `json:"error"`
}

type OAMessage struct {
	Role             string       `json:"role"`
	Content          any          `json:"content"`
	Reasoning        string       `json:"reasoning,omitempty"`
	ReasoningContent string       `json:"reasoning_content,omitempty"`
	ToolCalls        []OAToolCall `json:"tool_calls,omitempty"`
	Name             string       `json:"name,omitempty"`
	ToolCallID       string       `json:"tool_call_id,omitempty"`
}

type OAToolCall struct {
	ID       string `json:"id"`
	Index    int    `json:"index"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type OAUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ResponseFormat struct {
	Type       string      `json:"type"`
	JsonSchema *JsonSchema `json:"json_schema,omitempty"`
}

type JsonSchema struct {
	Schema json.RawMessage `json:"schema"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type OAToolDef struct {
	Type     string         `json:"type"`
	Function OAToolFunction `json:"function"`
}

type OAToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ChatCompletionRequest struct {
	Model         string          `json:"model"`
	Messages      []OAMessage     `json:"messages"`
	Stream        bool            `json:"stream"`
	StreamOptions *StreamOptions  `json:"stream_options,omitempty"`
	MaxTokens     *int            `json:"max_tokens,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	Stop          any             `json:"stop,omitempty"`
	Tools         []OAToolDef     `json:"tools,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

type OAChoice struct {
	Index        int      `json:"index"`
	Message      OAMessage `json:"message"`
	FinishReason *string  `json:"finish_reason"`
}

type OAChunkChoice struct {
	Index        int      `json:"index"`
	Delta        OAMessage `json:"delta"`
	FinishReason *string  `json:"finish_reason"`
}

type ChatCompletion struct {
	ID                string     `json:"id"`
	Object            string     `json:"object"`
	Created           int64      `json:"created"`
	Model             string     `json:"model"`
	SystemFingerprint string     `json:"system_fingerprint"`
	Choices           []OAChoice `json:"choices"`
	Usage             OAUsage    `json:"usage,omitempty"`
}

type ChatCompletionChunk struct {
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           int64           `json:"created"`
	Model             string          `json:"model"`
	SystemFingerprint string          `json:"system_fingerprint"`
	Choices           []OAChunkChoice `json:"choices"`
	Usage             *OAUsage        `json:"usage,omitempty"`
}

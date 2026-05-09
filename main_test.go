package main

import (
	"encoding/json"
	"testing"
)

func TestConvertAnthropicThinkingToReasoningContent(t *testing.T) {
	thinking := "previous reasoning"
	text := "answer"

	req := MessagesRequest{
		Model: "test-model",
		Messages: []MessageParam{
			{
				Role: "assistant",
				Content: []ContentBlock{
					{Type: "thinking", Thinking: &thinking},
					{Type: "text", Text: &text},
				},
			},
		},
	}

	got := convertAnthropicToOpenAI(req)
	if len(got.Messages) != 1 {
		t.Fatalf("got %d messages, want 1", len(got.Messages))
	}

	msg := got.Messages[0]
	if msg.ReasoningContent != thinking {
		t.Fatalf("ReasoningContent = %q, want %q", msg.ReasoningContent, thinking)
	}
	if msg.Reasoning != "" {
		t.Fatalf("Reasoning = %q, want empty", msg.Reasoning)
	}

	payload, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(payload) {
		t.Fatalf("invalid JSON: %s", payload)
	}
	if !containsJSONField(payload, "reasoning_content") {
		t.Fatalf("payload missing reasoning_content: %s", payload)
	}
	if containsJSONField(payload, "reasoning") {
		t.Fatalf("payload unexpectedly contains reasoning: %s", payload)
	}
}

func containsJSONField(payload []byte, field string) bool {
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		return false
	}
	messages, ok := decoded["messages"].([]any)
	if !ok || len(messages) == 0 {
		return false
	}
	msg, ok := messages[0].(map[string]any)
	if !ok {
		return false
	}
	_, ok = msg[field]
	return ok
}

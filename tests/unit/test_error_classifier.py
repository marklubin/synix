"""Unit tests — error classifier and dead letter queue."""

from __future__ import annotations

from synix.build.error_classifier import (
    DeadLetterQueue,
    ErrorVerdict,
    LLMErrorClassifier,
)


class TestLLMErrorClassifier:
    def setup_method(self):
        self.classifier = LLMErrorClassifier()

    def test_content_filter_is_dlq(self):
        exc = RuntimeError(
            "LLM API error processing episode ep-123: Error code: 400 - "
            "{'error': {'message': 'The request was rejected because it was "
            "considered high risk', 'type': 'content_filter'}}"
        )
        assert self.classifier.classify(exc, "ep-123") == ErrorVerdict.DLQ

    def test_content_moderation_is_dlq(self):
        exc = RuntimeError("content moderation flagged this request")
        assert self.classifier.classify(exc, "ep-456") == ErrorVerdict.DLQ

    def test_safety_system_is_dlq(self):
        exc = RuntimeError("safety system rejected the prompt")
        assert self.classifier.classify(exc, "ep-789") == ErrorVerdict.DLQ

    def test_input_too_large_is_dlq(self):
        exc = RuntimeError("context length exceeded: 200000 tokens > 128000 max")
        assert self.classifier.classify(exc, "ep-big") == ErrorVerdict.DLQ

    def test_too_many_tokens_is_dlq(self):
        exc = RuntimeError("too many tokens in input")
        assert self.classifier.classify(exc, "ep-big2") == ErrorVerdict.DLQ

    def test_auth_error_is_fatal(self):
        exc = RuntimeError(
            "LLM API error: Error code: 401 - {'error': {'message': "
            "'The API Key appears to be invalid', 'type': 'invalid_authentication_error'}}"
        )
        assert self.classifier.classify(exc, "ep-auth") == ErrorVerdict.FATAL

    def test_invalid_api_key_is_fatal(self):
        exc = RuntimeError("invalid API key provided")
        assert self.classifier.classify(exc, "ep-key") == ErrorVerdict.FATAL

    def test_unknown_error_is_fatal(self):
        exc = RuntimeError("some unexpected server error")
        assert self.classifier.classify(exc, "ep-unknown") == ErrorVerdict.FATAL

    def test_connection_error_post_retry_is_fatal(self):
        exc = RuntimeError("Failed to process ep-conn after 2 attempts: connection refused")
        assert self.classifier.classify(exc, "ep-conn") == ErrorVerdict.FATAL

    def test_chained_cause_checked(self):
        """Exception.__cause__ is also checked for classification patterns."""
        cause = RuntimeError("content_filter triggered")
        outer = RuntimeError("LLM API error")
        outer.__cause__ = cause
        assert self.classifier.classify(outer, "ep-chain") == ErrorVerdict.DLQ

    def test_auth_takes_precedence_over_content_filter(self):
        """If both patterns match, auth (fatal) wins because it's checked first."""
        exc = RuntimeError("unauthorized content_filter request")
        assert self.classifier.classify(exc, "ep-both") == ErrorVerdict.FATAL


class TestDeadLetterQueue:
    def test_empty_dlq(self):
        dlq = DeadLetterQueue()
        assert len(dlq) == 0
        assert dlq.summary() == "No errors"

    def test_add_entry(self):
        dlq = DeadLetterQueue()
        exc = RuntimeError("content filter hit")
        dlq.add("ep-123", exc, layer_name="episodes")
        assert len(dlq) == 1
        assert dlq.entries[0].artifact_desc == "ep-123"
        assert dlq.entries[0].error_type == "RuntimeError"
        assert dlq.entries[0].layer_name == "episodes"

    def test_summary_groups_by_type(self):
        dlq = DeadLetterQueue()
        dlq.add("ep-1", RuntimeError("filter"), layer_name="episodes")
        dlq.add("ep-2", RuntimeError("filter"), layer_name="episodes")
        dlq.add("ep-3", ValueError("too big"), layer_name="episodes")
        summary = dlq.summary()
        assert "3 artifacts skipped" in summary
        assert "2 RuntimeError" in summary
        assert "1 ValueError" in summary

    def test_slogger_receives_dlq_events(self):
        """When a SynixLogger is attached, DLQ.add() writes to the structured log."""

        class FakeLogger:
            def __init__(self):
                self.calls = []

            def artifact_dlq(self, layer_name, artifact_desc, error_type, error_message):
                self.calls.append((layer_name, artifact_desc, error_type, error_message))

        fake = FakeLogger()
        dlq = DeadLetterQueue()
        dlq.slogger = fake
        dlq.add("ep-42", RuntimeError("content_filter hit"), layer_name="episodes")

        assert len(fake.calls) == 1
        assert fake.calls[0] == ("episodes", "ep-42", "RuntimeError", "content_filter hit")

"""Tests for the sanitized, opt-out telemetry layer (issue y6to.2).

Acceptance criteria under test:
  1. Runs emit sanitized telemetry events (Segment-style Track shape).
  2. Opt-out via env var (``SWARM_TELEMETRY=off``) and the off/errors/all gate.
  3. No PII in payloads — anonymous random id, sanitized commands, scrubbed
     properties, PII-free context.
Plus: emission is best-effort (a broken sink or property never raises).
"""

from __future__ import annotations

import json

from swarm.telemetry import (
    FileSink,
    NullSink,
    Telemetry,
    TelemetryEvent,
    TelemetryLevel,
    TelemetrySink,
    anonymous_id,
    build_context,
    get_telemetry,
    level_from_env,
    reset_telemetry,
    sanitize_command,
    scrub_properties,
    scrub_value,
    should_track,
    track,
)


class RecordingSink(TelemetrySink):
    def __init__(self):
        self.events = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


# Levels / gate


class TestLevels:
    def test_parses_known_levels(self):
        assert level_from_env("off") == TelemetryLevel.OFF
        assert level_from_env("errors") == TelemetryLevel.ERRORS
        assert level_from_env("all") == TelemetryLevel.ALL

    def test_default_is_all_for_unset_or_unknown(self):
        assert level_from_env(None) == TelemetryLevel.ALL
        assert level_from_env("") == TelemetryLevel.ALL
        assert level_from_env("garbage") == TelemetryLevel.ALL

    def test_case_and_whitespace_insensitive(self):
        assert level_from_env("  OFF  ") == TelemetryLevel.OFF

    def test_should_track_matrix(self):
        assert not should_track(TelemetryLevel.OFF, True)
        assert not should_track(TelemetryLevel.OFF, False)
        assert should_track(TelemetryLevel.ERRORS, True)
        assert not should_track(TelemetryLevel.ERRORS, False)
        assert should_track(TelemetryLevel.ALL, True)
        assert should_track(TelemetryLevel.ALL, False)


# Command sanitization


class TestSanitizeCommand:
    def test_program_reduced_to_basename(self):
        out = sanitize_command(["/usr/local/bin/python", "list"], "list")
        assert out == "python list"

    def test_paths_redacted_flags_kept(self):
        out = sanitize_command(
            ["swarm", "run", "scenarios/baseline.yaml", "--seed", "42"], "run"
        )
        assert out == "swarm run VALUE --seed 42"

    def test_flag_equals_value_redacted(self):
        out = sanitize_command(["swarm", "run", "--level=secret"], "run")
        assert out == "swarm run --level=VALUE"

    def test_booleans_and_semver_kept(self):
        out = sanitize_command(
            ["swarm", "run", "--verbose", "true", "--min-version", "1.2.3"], "run"
        )
        assert out == "swarm run --verbose true --min-version 1.2.3"

    def test_nested_subcommand_tokens_kept(self):
        out = sanitize_command(
            ["swarm", "sandbox", "exec", "--", "secretcmd"], "sandbox exec"
        )
        assert out.startswith("swarm sandbox exec")
        assert "secretcmd" not in out

    def test_empty_argv(self):
        assert sanitize_command([], "") == ""


# Property scrubbing


class TestScrub:
    def test_numbers_and_bools_pass_through(self):
        assert scrub_value(42) == 42
        assert scrub_value(3.14) == 3.14
        assert scrub_value(True) is True
        assert scrub_value(None) is None

    def test_path_like_strings_redacted(self):
        assert scrub_value("/home/me/data.yaml") == "VALUE"
        assert scrub_value("~/secret") == "VALUE"
        assert scrub_value("./rel/path") == "VALUE"
        assert scrub_value("C:\\Users\\me") == "VALUE"

    def test_identifier_tokens_kept(self):
        assert scrub_value("baseline") == "baseline"
        assert scrub_value("1.2.3") == "1.2.3"

    def test_presanitized_command_string_survives(self):
        # A multi-token string already run through sanitize_command must not be
        # nuked by the defensive scrub (regression: the space heuristic ate it).
        assert scrub_value("swarm run --seed 42 --export-json VALUE") == (
            "swarm run --seed 42 --export-json VALUE"
        )

    def test_oversized_blob_redacted(self):
        assert scrub_value("x" * 300) == "VALUE"

    def test_sensitive_keys_redacted_regardless_of_value(self):
        out = scrub_properties(
            {
                "scenario": "baseline",
                "seed": 7,
                "api_key": "xyz",
                "username": "alice",
                "homedir": "anything",
                "email": "a@b.com",
            }
        )
        assert out["scenario"] == "baseline"
        assert out["seed"] == 7
        assert out["api_key"] == "REDACTED"
        assert out["username"] == "REDACTED"
        assert out["homedir"] == "REDACTED"
        assert out["email"] == "REDACTED"

    def test_nested_structures_scrubbed(self):
        out = scrub_value({"a": [1, "/p/q", {"token": "t"}], "b": "ok"})
        assert out["a"][0] == 1
        assert out["a"][1] == "VALUE"
        assert out["a"][2]["token"] == "REDACTED"
        assert out["b"] == "ok"


# Context (no PII)


class TestContext:
    def test_has_expected_keys(self):
        ctx = build_context()
        assert ctx["app"]["name"] == "swarm"
        for key in ("os", "device", "runtime", "app", "locale"):
            assert key in ctx

    def test_contains_no_identifying_fields(self):
        blob = json.dumps(build_context()).lower()
        for forbidden in ("hostname", "username", "/users/", "/home/"):
            assert forbidden not in blob


# Anonymous id


class TestAnonymousId:
    def test_stable_and_uuid_shaped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SWARM_TELEMETRY_ID_PATH", str(tmp_path / ".id"))
        first = anonymous_id()
        second = anonymous_id()
        assert first == second
        assert len(first) == 36 and first.count("-") == 4

    def test_persisted_to_disk(self, tmp_path, monkeypatch):
        idfile = tmp_path / ".id"
        monkeypatch.setenv("SWARM_TELEMETRY_ID_PATH", str(idfile))
        val = anonymous_id()
        assert idfile.read_text().strip() == val

    def test_not_derived_from_machine_identity(self, tmp_path, monkeypatch):
        # Two independent caches yield different ids → it's random, not a
        # deterministic hash of MAC/hostname/username.
        monkeypatch.setenv("SWARM_TELEMETRY_ID_PATH", str(tmp_path / "a.id"))
        a = anonymous_id()
        monkeypatch.setenv("SWARM_TELEMETRY_ID_PATH", str(tmp_path / "b.id"))
        b = anonymous_id()
        assert a != b


# Event shape


class TestEvent:
    def test_track_dict_uses_segment_keys(self):
        ev = TelemetryEvent(
            event="Test", anonymous_id="anon", properties={"k": 1}
        )
        d = ev.to_dict()
        assert set(d) == {
            "anonymousId",
            "event",
            "properties",
            "context",
            "timestamp",
            "messageId",
        }
        assert d["anonymousId"] == "anon"


# Telemetry facade


class TestTelemetry:
    def test_track_emits_when_enabled(self):
        sink = RecordingSink()
        tel = Telemetry(level=TelemetryLevel.ALL, sink=sink, anon_id="anon")
        assert tel.track("Run Completed", {"epochs": 10}) is True
        assert len(sink.events) == 1
        ev = sink.events[0]
        assert ev.event == "Run Completed"
        assert ev.anonymous_id == "anon"
        assert ev.properties == {"epochs": 10}
        assert ev.message_id and ev.timestamp

    def test_off_suppresses_all(self):
        sink = RecordingSink()
        tel = Telemetry(level=TelemetryLevel.OFF, sink=sink)
        assert tel.track("X", {}) is False
        assert tel.track("Y", {}, is_error=True) is False
        assert sink.events == []
        assert tel.enabled is False

    def test_errors_level_only_emits_errors(self):
        sink = RecordingSink()
        tel = Telemetry(level=TelemetryLevel.ERRORS, sink=sink, anon_id="a")
        assert tel.track("info", {}) is False
        assert tel.track("boom", {}, is_error=True) is True
        assert [e.event for e in sink.events] == ["boom"]

    def test_properties_are_scrubbed_on_emit(self):
        sink = RecordingSink()
        tel = Telemetry(level=TelemetryLevel.ALL, sink=sink, anon_id="a")
        tel.track("E", {"path": "/home/me/x", "n": 3})
        assert sink.events[0].properties == {"path": "REDACTED", "n": 3}

    def test_off_without_sink_uses_null_sink(self):
        tel = Telemetry(level=TelemetryLevel.OFF)
        assert isinstance(tel.sink, NullSink)

    def test_track_is_best_effort_on_sink_failure(self):
        class ExplodingSink(TelemetrySink):
            def emit(self, event):
                raise RuntimeError("boom")

        tel = Telemetry(level=TelemetryLevel.ALL, sink=ExplodingSink(), anon_id="a")
        # Must not propagate; returns False instead.
        assert tel.track("E", {"k": 1}) is False


# File sink


class TestFileSink:
    def test_appends_jsonl(self, tmp_path):
        path = tmp_path / "nested" / "events.jsonl"
        sink = FileSink(path)
        sink.emit(TelemetryEvent(event="A", anonymous_id="x"))
        sink.emit(TelemetryEvent(event="B", anonymous_id="x"))
        lines = path.read_text().splitlines()
        assert [json.loads(line)["event"] for line in lines] == ["A", "B"]

    def test_emit_never_raises_on_bad_path(self, tmp_path):
        # A directory where a file is expected → OSError, swallowed.
        bad = tmp_path / "dir"
        bad.mkdir()
        sink = FileSink(bad)
        sink.emit(TelemetryEvent(event="A", anonymous_id="x"))  # no raise


# No-PII end-to-end


class TestNoPIIEndToEnd:
    def test_emitted_event_contains_no_paths_or_identity(self):
        sink = RecordingSink()
        tel = Telemetry(level=TelemetryLevel.ALL, sink=sink, anon_id="anon-uuid")
        tel.track(
            "Command Invoked",
            {
                "subcommand": "run",
                "command": sanitize_command(
                    ["/usr/bin/python", "run", "/home/me/secret.yaml"], "run"
                ),
            },
        )
        blob = json.dumps(sink.events[0].to_dict()).lower()
        for forbidden in ("/home/me", "secret.yaml", "/usr/bin", "username"):
            assert forbidden not in blob


# Singleton


class TestSingleton:
    def test_get_telemetry_is_stable(self):
        reset_telemetry(None)
        a = get_telemetry()
        b = get_telemetry()
        assert a is b
        reset_telemetry(None)

    def test_module_track_uses_singleton(self):
        sink = RecordingSink()
        reset_telemetry(Telemetry(level=TelemetryLevel.ALL, sink=sink, anon_id="a"))
        assert track("E", {"k": 1}) is True
        assert sink.events[0].event == "E"
        reset_telemetry(None)

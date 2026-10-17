"""Geolocation failures, retries and bounded browser callbacks without real GPS."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest
from streamlit.testing.v1 import AppTest
import streamlit_js_eval

from frontend.geolocation import GEOLOCATION_JS


APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


@pytest.mark.parametrize("code, message", [
    (1, "Доступ к геолокации запрещён"),
    (2, "Браузер не смог определить координаты"),
    (3, "за 10 секунд"),
    ("deadline", "За 15 секунд"),
    ("insecure", "HTTPS"),
])
def test_failed_geolocation_finishes_and_allows_retry(monkeypatch, code, message):
    calls = []
    result = {"error": {"code": code}}

    def fake_js(*, js_expressions, key):
        calls.append(key)
        return result

    monkeypatch.setattr(streamlit_js_eval, "streamlit_js_eval", fake_js)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    previous = (app.session_state.lat, app.session_state.lon)
    app.button(key="btn_geo").click().run()
    assert not app.exception
    assert message in app.session_state.geo_note
    assert app.session_state.geo_click == 0
    assert (app.session_state.lat, app.session_state.lon) == previous
    assert not app.button(key="btn_geo").disabled

    # Errors and successful requests must both get fresh component keys.
    result = {"coords": {"latitude": 51.128345, "longitude": 71.430567, "accuracy": 25}}
    app.button(key="btn_geo").click().run()
    assert not app.exception
    assert (app.session_state.lat, app.session_state.lon) == (51.128, 71.431)
    app.button(key="btn_geo").click().run()
    assert not app.exception
    assert calls == ["geo_request_1", "geo_request_2", "geo_request_3"]


def test_pending_geolocation_does_not_start_duplicate_requests(monkeypatch):
    monkeypatch.setattr(streamlit_js_eval, "streamlit_js_eval", lambda **kwargs: None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button(key="btn_geo").click().run()
    assert not app.exception
    assert app.button(key="btn_geo").disabled
    assert app.session_state.geo_click == 1
    assert any("до 15 секунд" in caption.value for caption in app.caption)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed for the JS callback test")
def test_browser_geolocation_resolves_on_all_paths():
    # A fake clock verifies both timeout paths without waiting or requesting GPS.
    harness = r"""
    const vm = require('node:vm');
    const assert = require('node:assert/strict');
    const source = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
    async function check(mode) {
        let success, failure, options;
        const timers = new Map();
        let serial = 0;
        const context = {
            window: {isSecureContext: mode !== 'insecure'},
            navigator: {},
            setTimeout: (fn, ms) => { timers.set(++serial, {fn, ms}); return serial; },
            clearTimeout: id => timers.delete(id),
        };
        if (mode !== 'unsupported') context.navigator.geolocation = {
            getCurrentPosition: (ok, err, opts) => {
                success = ok; failure = err; options = opts;
                if (mode === 'throw') throw {name: 'SecurityError'};
            },
        };
        const pending = vm.runInNewContext(source, context);
        if (mode === 'success') {
            assert.equal(options.timeout, 10000);
            assert.equal(options.enableHighAccuracy, false);
            assert.equal(options.maximumAge, 60000);
            success({coords: {latitude: 51.128345, longitude: 71.430567, accuracy: 25}});
        } else if (['1', '2', '3'].includes(mode)) {
            failure({code: Number(mode)});
        } else if (mode === 'deadline') {
            const timer = [...timers.values()][0];
            assert.equal(timer.ms, 15000);
            timer.fn();
            success({coords: {latitude: 1, longitude: 2}}); // Ignore a late response.
        }
        const value = JSON.parse(JSON.stringify(await pending));
        assert.equal(timers.size, 0);
        if (mode === 'success') {
            assert.deepEqual(value, {coords: {latitude: 51.128, longitude: 71.431, accuracy: 25}});
        } else {
            assert.equal(String(value.error.code), mode === 'throw' ? '1' : mode);
        }
    }
    (async () => {
        for (const mode of ['success', '1', '2', '3', 'deadline', 'insecure', 'unsupported', 'throw']) {
            await check(mode);
        }
    })().catch(error => { console.error(error); process.exitCode = 1; });
    """
    result = subprocess.run(
        [shutil.which("node"), "-e", harness],
        input=json.dumps(GEOLOCATION_JS), text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr

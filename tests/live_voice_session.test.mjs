import assert from "node:assert/strict";
import test from "node:test";
import { webcrypto } from "node:crypto";

// Exercise real component handlers with fake devices/provider, without recording a person.
test("session handles reruns, tool cancellation, mute, barge-in and late grants safely", async () => {
    const elements = new Map();
    function element() { return {
        hidden: false, textContent: "", children: [], scrollHeight: 100, dataset: {},
        getBoundingClientRect: () => ({ height: 200 }),
        setAttribute() {}, append(...nodes) { this.children.push(...nodes); },
        get firstElementChild() { return { remove: () => this.children.shift() }; },
    }; }
    globalThis.document = {
        body: { scrollHeight: 300 }, getElementById: id => {
            if (!elements.has(id)) elements.set(id, element());
            return elements.get(id);
        }, createElement: element, createTextNode: text => ({ text }),
    };
    const messages = [], listeners = {};
    globalThis.parent = { postMessage: message => {
        messages.push(message);
        if (message.type === "streamlit:setFrameHeight") document.body.scrollHeight = message.height;
    } };
    globalThis.window = { isSecureContext: true, addEventListener: (name, fn) => { listeners[name] = fn; } };
    globalThis.ResizeObserver = class { observe() {} };
    Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
    let tracks = [];
    Object.defineProperty(globalThis, "navigator", { configurable: true, value: { mediaDevices: {
        getUserMedia: async () => {
            const track = { enabled: true, stopped: false, stop() { this.stopped = true; } };
            tracks.push(track);
            return { getTracks: () => [track], getAudioTracks: () => [track] };
        },
    } } });
    const contexts = [], played = [], processors = [], sockets = [];
    globalThis.AudioContext = class {
        constructor() { contexts.push(this); this.currentTime = 1; this.destination = {}; this.audioWorklet = { addModule: async () => {} }; }
        async resume() {} async close() { this.closed = true; }
        createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
        createGain() { return { gain: {}, connect() {}, disconnect() {} }; }
        createBuffer(_, length, rate) { return { duration: length / rate, getChannelData: () => new Float32Array(length) }; }
        createBufferSource() { const node = { connect() {}, start() {}, stop() { this.stopped = true; } }; played.push(node); return node; }
    };
    globalThis.AudioWorkletNode = class {
        constructor() { this.port = {}; processors.push(this); }
        connect() { return { connect() {} }; } disconnect() {}
    };
    globalThis.WebSocket = class {
        static OPEN = 1;
        constructor(url) { this.url = url; this.readyState = 1; this.bufferedAmount = 0; this.sent = []; sockets.push(this); }
        send(text) { this.sent.push(JSON.parse(text)); }
        close() { this.closed = true; }
    };
    await import("../frontend/live_voice_component/app.mjs");
    const render = args => listeners.message({ source: parent, data: { type: "streamlit:render", args } });
    const latestEvent = () => messages.filter(message => message.type === "streamlit:setComponentValue").at(-1).value;
    const tick = async () => { await new Promise(resolve => setImmediate(resolve)); };
    const deliver = async message => { sockets.at(-1).onmessage({ data: JSON.stringify(message) }); await tick(); };
    render({ configured: true });
    await elements.get("start").onclick();
    const connectEvent = latestEvent();
    assert.equal(connectEvent.type, "connect");
    const grant = { session: connectEvent.session, token: "ephemeral", setup: { model: "models/test" }, seconds: 600 };
    render({ configured: true, ack: connectEvent.id, grant });
    render({ configured: true, ack: connectEvent.id, grant });
    await tick();
    assert.equal(sockets.length, 1); // A rerun while loading the worklet must not duplicate a connection.
    sockets[0].onopen();
    assert.deepEqual(sockets[0].sent[0], { setup: grant.setup });
    await deliver({ setupComplete: {} });
    assert.equal(elements.get("mute").hidden, false);
    processors[0].port.onmessage({ data: new ArrayBuffer(4) });
    assert.ok(sockets[0].sent.at(-1).realtimeInput.audio);
    elements.get("mute").onclick();
    assert.equal(tracks[0].enabled, false);
    assert.ok(sockets[0].sent.at(-1).realtimeInput.audioStreamEnd);
    const count = sockets[0].sent.length;
    processors[0].port.onmessage({ data: new ArrayBuffer(4) });
    assert.equal(sockets[0].sent.length, count);
    elements.get("mute").onclick();
    await deliver({ toolCall: { functionCalls: [{ id: "cancelled", name: "search_places", args: { query: "cafe" } }] } });
    const toolEvent = latestEvent();
    assert.equal(toolEvent.type, "tool");
    await deliver({ toolCallCancellation: { ids: ["cancelled"] } });
    render({ ack: toolEvent.id, session: connectEvent.session, tool_results: [{ id: "cancelled", name: "search_places", response: {} }] });
    assert.equal(sockets[0].sent.filter(message => message.toolResponse).length, 0);
    await deliver({ toolCall: { functionCalls: [{ id: "valid", name: "search_places", args: { query: "park" } }] } });
    const nextTool = latestEvent();
    const result = { id: "valid", name: "search_places", response: { places: [] } };
    render({ ack: nextTool.id, session: connectEvent.session, tool_results: [result] });
    assert.deepEqual(sockets[0].sent.at(-1).toolResponse.functionResponses, [result]);
    const used = latestEvent();
    assert.equal(used.type, "used");
    render({ ack: used.id, session: connectEvent.session, tool_results: [result] });
    assert.equal(sockets[0].sent.filter(message => message.toolResponse).length, 1);
    await deliver({ serverContent: { modelTurn: { parts: [{ inlineData: { data: btoa("\0\0"), mimeType: "audio/pcm;rate=24000" } }] } } });
    assert.equal(played.length, 1);
    await deliver({ serverContent: { interrupted: true } });
    assert.equal(played[0].stopped, true);
    elements.get("stop").onclick();
    assert.ok(tracks.every(track => track.stopped));
    assert.ok(contexts.every(context => context.closed));
    assert.ok(sockets[0].closed);
    const stopped = latestEvent();
    render({ configured: true, ack: stopped.id });
    await elements.get("start").onclick();
    const secondConnect = latestEvent();
    elements.get("stop").onclick();
    render({ configured: true, ack: secondConnect.id, grant: { ...grant, session: secondConnect.session } });
    await tick();
    assert.equal(sockets.length, 1); // Stop during authentication invalidates the arriving grant.
    assert.ok(tracks.every(track => track.stopped));

    render({ configured: true, ack: latestEvent().id });
    await elements.get("start").onclick();
    const thirdConnect = latestEvent();
    render({ configured: true, ack: thirdConnect.id, grant: { ...grant, session: thirdConnect.session } });
    await tick();
    sockets[1].onopen();
    await deliver({ setupComplete: {} });
    await deliver({ serverContent: { inputTranscription: { text: "Where can I " } } });
    await deliver({ serverContent: { inputTranscription: { text: "get coffee?" } } });
    await deliver({ toolCall: { functionCalls: [{ id: "chat-call", name: "search_places", args: { query: "coffee" } }] } });
    const speech = latestEvent();
    assert.equal(speech.type, "speech");
    assert.equal(speech.question, "Where can I get coffee?");
    render({ ack: speech.id, session: thirdConnect.session });
    const chatTool = latestEvent();
    assert.equal(chatTool.type, "tool");
    assert.equal(chatTool.turn_id, speech.turn_id);
    render({ ack: chatTool.id, session: thirdConnect.session, tool_results: [
        { id: "chat-call", name: "search_places", response: { places: [] } },
    ] });
    const chatUsed = latestEvent();
    assert.equal(chatUsed.type, "used");
    render({ ack: chatUsed.id, session: thirdConnect.session });
    await deliver({ serverContent: { turnComplete: true } });
    assert.equal(latestEvent().id, chatUsed.id); // The tool boundary must not close the chat turn.
    await deliver({ serverContent: { outputTranscription: { text: "Try the nearest cafe." } } });
    await deliver({ serverContent: { turnComplete: true } });
    const finalSpeech = latestEvent();
    assert.equal(finalSpeech.type, "speech");
    assert.equal(finalSpeech.turn_id, speech.turn_id);
    assert.equal(finalSpeech.answer, "Try the nearest cafe.");
    assert.equal(finalSpeech.final, true);
    render({ ack: finalSpeech.id, session: thirdConnect.session });
    elements.get("stop").onclick();
    assert.equal(new Set(messages.filter(message => message.type === "streamlit:setFrameHeight").map(message => message.height)).size, 1);
});

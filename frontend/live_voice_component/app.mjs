import { AudioQueue, encodePCM } from "./audio.mjs";

const $ = id => document.getElementById(id);
let args = {}, socket, stream, inputContext, outputContext, processor, source, silent, audio;
let requestId = null, connecting = false, opening = false, ready = false, muted = false, timer, deadline;
let sequence = 0, inflight = null, outbox = [], turn = null, turnNumber = 0, speechTimer;
const pending = new Map(), cancelled = new Set(), delivered = new Set();
const bridge = (type, value = {}) => parent.postMessage({ isStreamlitMessage: true, type, ...value }, "*");
// Measure content, not the iframe viewport: repeated chat updates must not grow the composer.
const resize = () => bridge("streamlit:setFrameHeight", { height: Math.ceil($("panel").getBoundingClientRect().height) + 4 });

function flush() {
    if (inflight || !outbox.length) return;
    inflight = outbox.shift();
    bridge("streamlit:setComponentValue", { value: inflight, dataType: "json" });
}
function emit(type, payload = {}) {
    if (type === "speech") {
        // Replace queued partial text; never pile up reruns behind a slow search.
        outbox = outbox.filter(event => event.type !== "speech" || event.turn_id !== payload.turn_id);
    }
    outbox.push({ id: `${requestId || "idle"}:${++sequence}`, session: requestId, type, ...payload });
    flush();
}
function status(text, phase = "idle") {
    $("status").textContent = text;
    $("panel").dataset.phase = phase;
}
function error(text) { $("error").textContent = text; $("error").hidden = !text; resize(); }
function send(value) { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(value)); }
function controls(active) {
    $("start").hidden = active;
    $("stop").hidden = !active;
    $("mute").hidden = !ready;
}
function currentTurn() {
    if (!turn) turn = { id: String(++turnNumber), question: "", answer: "", awaitingTool: false, hasOutput: false };
    return turn;
}
function publishSpeech(final = false, interrupted = false) {
    clearTimeout(speechTimer); speechTimer = null;
    if (!turn || (!turn.question && !turn.answer)) return;
    emit("speech", { turn_id: turn.id, question: turn.question, answer: turn.answer, final, interrupted });
}
function scheduleSpeech() {
    if (!speechTimer) speechTimer = setTimeout(() => publishSpeech(), 600);
}
function finishTurn(interrupted = false) {
    publishSpeech(true, interrupted);
    turn = null;
}
function stop(note = "Разговор завершён", report = true) {
    const oldId = requestId;
    if (report) finishTurn();
    else { clearTimeout(speechTimer); turn = null; }
    connecting = false; opening = false; ready = false; requestId = null;
    clearTimeout(timer); clearTimeout(deadline);
    if (socket) { socket.onclose = null; socket.onerror = null; socket.close(); socket = null; }
    stream?.getTracks().forEach(track => track.stop()); stream = null;
    processor?.disconnect(); source?.disconnect(); silent?.disconnect();
    processor = null; source = null; silent = null;
    audio?.clear(); audio = null;
    if (inputContext) { void inputContext.close(); inputContext = null; }
    if (outputContext) { void outputContext.close(); outputContext = null; }
    pending.clear(); cancelled.clear(); delivered.clear();
    muted = false;
    $("mute").textContent = "Выключить микрофон";
    $("mute").setAttribute("aria-pressed", "false");
    controls(false); status(note);
    if (report && oldId) emit("stopped", { closed_session: oldId });
    resize();
}
function fail(message) { stop("Не удалось начать разговор"); error(message); }

async function start() {
    if (connecting || ready) return;
    if (!args.configured) { error("Добавьте GEMINI_API_KEY в .env на сервере и обновите страницу."); return; }
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        error("Микрофон доступен по HTTPS или на localhost. Откройте приложение в отдельной вкладке."); return;
    }
    error("");
    requestId = crypto.randomUUID();
    const id = requestId;
    connecting = true; controls(true); status("Разрешите доступ к микрофону…", "connecting");
    timer = setTimeout(() => fail("Не удалось подключиться за 45 секунд. Проверьте разрешение микрофона и сеть."), 45000);
    try {
        // Resume output during the click gesture, before waiting for network/authentication.
        outputContext = new AudioContext();
        await outputContext.resume();
        inputContext = new AudioContext();
        await inputContext.resume();
        const media = await navigator.mediaDevices.getUserMedia({ audio: {
            channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
        }, video: false });
        if (id !== requestId) { media.getTracks().forEach(track => track.stop()); return; }
        stream = media;
        stream.getAudioTracks()[0].onended = () => stop("Доступ к микрофону завершён");
        status("Подключаю голосового помощника…", "connecting");
        emit("connect");
    } catch (exc) {
        if (id !== requestId) return;
        fail(exc.name === "NotAllowedError" ? "Разрешите доступ к микрофону в настройках браузера и повторите." :
            "Не удалось открыть микрофон. Проверьте устройство и повторите.");
    }
}

async function openSocket(grant, id) {
    if (id !== requestId || socket || opening) return;
    opening = true;
    if (grant.error) { fail(grant.error); return; }
    try {
        audio = new AudioQueue(outputContext);
        await inputContext.audioWorklet.addModule(new URL("./pcm-worklet.mjs", import.meta.url));
        if (id !== requestId) return;
        processor = new AudioWorkletNode(inputContext, "microphone-pcm");
        source = inputContext.createMediaStreamSource(stream);
        silent = inputContext.createGain(); silent.gain.value = 0;
        processor.connect(silent).connect(inputContext.destination);
        source.connect(processor);
        const url = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained";
        socket = new WebSocket(`${url}?access_token=${encodeURIComponent(grant.token)}`);
        socket.onopen = () => send({ setup: grant.setup });
        // Promise chaining preserves provider event order even for Blob messages.
        let chain = Promise.resolve();
        socket.onmessage = event => { chain = chain.then(async () => {
            if (id !== requestId) return;
            const message = JSON.parse(typeof event.data === "string" ? event.data : await event.data.text());
            if (id === requestId) handleMessage(message, grant.seconds);
        }).catch(() => { if (id === requestId) fail("Ошибка голосового потока. Завершите разговор и подключитесь снова."); }); };
        socket.onerror = () => { if (id === requestId) fail("Не удалось открыть Gemini Live. Проверьте сеть, доступность модели и квоту API."); };
        socket.onclose = event => {
            if (id !== requestId) return;
            const message = event.code === 1000 ? "Разговор завершён" :
                "Соединение завершено. Проверьте квоту и доступность GEMINI_LIVE_MODEL; можно подключиться снова.";
            stop(message);
        };
        processor.port.onmessage = event => {
            if (ready && !muted) {
                if (socket.bufferedAmount > 256000) { fail("Соединение слишком медленное для голоса. Попробуйте ещё раз."); return; }
                send({ realtimeInput: { audio: { data: encodePCM(event.data), mimeType: "audio/pcm;rate=16000" } } });
            }
        };
    } catch { if (id === requestId) fail("Не удалось запустить звук. Попробуйте актуальный Chrome или Safari."); }
}

function handleMessage(message, seconds) {
    if (message.error) { fail("Gemini отклонил голосовую сессию. Проверьте модель, доступ и лимит API."); return; }
    if (message.setupComplete) {
        clearTimeout(timer); connecting = false; ready = true;
        deadline = setTimeout(() => stop("Лимит разговора — 10 минут. Подключитесь снова."), seconds * 1000);
        controls(true); status("Слушаю — можно говорить и перебивать", "listening"); resize();
    }
    const content = message.serverContent;
    if (content) {
        if (content.interrupted) {
            audio.clear(); finishTurn(true);
            status(muted ? "Микрофон выключен" : "Слушаю…", muted ? "muted" : "listening");
        }
        if (content.inputTranscription?.text) {
            currentTurn().question += content.inputTranscription.text;
            scheduleSpeech();
        }
        if (content.outputTranscription?.text) {
            const active = currentTurn();
            active.answer += content.outputTranscription.text;
            active.awaitingTool = false; active.hasOutput = true;
            scheduleSpeech();
        }
        for (const part of content.modelTurn?.parts || []) {
            if (part.inlineData?.data) {
                const mime = part.inlineData.mimeType || "audio/pcm;rate=24000";
                if (!mime.startsWith("audio/pcm")) continue;
                const rate = Number(mime.match(/rate=(\d+)/)?.[1] || 24000);
                currentTurn().awaitingTool = false; currentTurn().hasOutput = true;
                audio.enqueue(part.inlineData.data, rate);
                status("Отвечаю — можно перебить", "speaking");
            }
        }
        // Gemini emits a turnComplete at the tool boundary, before the spoken answer.
        if (content.turnComplete && !message.toolCall && !turn?.awaitingTool) {
            finishTurn();
            status(muted ? "Микрофон выключен" : "Слушаю…", muted ? "muted" : "listening");
        }
    }
    if (message.toolCall) {
        const calls = message.toolCall.functionCalls || [];
        const active = currentTurn();
        active.awaitingTool = true;
        for (const call of calls) pending.set(call.id, call.name);
        publishSpeech();
        status("Ищу места и строю маршрут…", "searching");
        emit("tool", { calls, turn_id: active.id, question: active.question.slice(0, 1000) });
    }
    for (const id of message.toolCallCancellation?.ids || []) { cancelled.add(id); pending.delete(id); }
    if (message.goAway) stop("Сессия Gemini заканчивается. Начните новый разговор.");
}

function toolResults(results) {
    const responses = [];
    for (const result of results || []) {
        if (!pending.has(result.id) || cancelled.has(result.id) || delivered.has(result.id)) continue;
        delivered.add(result.id); pending.delete(result.id);
        responses.push(result);
    }
    if (ready && responses.length) {
        send({ toolResponse: { functionResponses: responses } });
        emit("used", { call_ids: responses.map(item => item.id) });
        status("Готовлю ответ…", "searching");
    }
}

$("start").onclick = start;
$("stop").onclick = () => stop();
$("mute").onclick = () => {
    muted = !muted;
    stream?.getAudioTracks().forEach(track => { track.enabled = !muted; });
    if (muted) send({ realtimeInput: { audioStreamEnd: true } });
    $("mute").textContent = muted ? "Включить микрофон" : "Выключить микрофон";
    $("mute").setAttribute("aria-pressed", String(muted));
    status(muted ? "Микрофон выключен" : "Слушаю…", muted ? "muted" : "listening");
};
window.addEventListener("message", event => {
    if (event.source !== parent || event.data.type !== "streamlit:render") return;
    args = event.data.args || {};
    // One acknowledged event at a time prevents Streamlit reruns from dropping tool calls.
    if (inflight && args.ack === inflight.id) { inflight = null; flush(); }
    if (args.grant?.session === requestId && connecting) void openSocket(args.grant, requestId);
    if (args.session === requestId) toolResults(args.tool_results);
    resize();
});
window.addEventListener("pagehide", () => stop("", false));
new ResizeObserver(resize).observe($("panel"));
bridge("streamlit:componentReady", { apiVersion: 1 });
resize();

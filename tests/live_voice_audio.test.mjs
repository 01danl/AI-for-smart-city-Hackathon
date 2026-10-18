import assert from "node:assert/strict";
import test from "node:test";
import { AudioQueue, PCMResampler, encodePCM } from "../frontend/live_voice_component/audio.mjs";

for (const rate of [16000, 44100, 48000]) {
    test(`continuous ${rate} Hz microphone converts to 16 kHz little-endian PCM`, () => {
        const converter = new PCMResampler(rate, 16000, 160);
        const chunks = [];
        for (let i = 0; i < rate; i += 128) {
            chunks.push(...converter.push(new Float32Array(Math.min(128, rate - i)).fill(.5)));
        }
        const count = chunks.reduce((sum, chunk) => sum + chunk.byteLength / 2, 0);
        assert.ok(Math.abs(count - 16000) <= 160);
        assert.equal(new DataView(chunks[0]).getInt16(0, true), 16384);
        assert.equal(atob(encodePCM(chunks[0])).length, 320);
    });
}
test("audio clips safely and interruption stops every queued source", () => {
    const converter = new PCMResampler(16000, 16000, 2);
    const [chunk] = converter.push(Float32Array.from([-2, 2, 0]));
    assert.equal(new DataView(chunk).getInt16(0, true), -32768);
    assert.equal(new DataView(chunk).getInt16(2, true), 32767);
    const sources = [];
    const context = {
        currentTime: 10, destination: {},
        createBuffer: (_, length, rate) => ({ duration: length / rate, getChannelData: () => new Float32Array(length) }),
        createBufferSource: () => {
            const source = { connect() {}, start(time) { this.startTime = time; }, stop() { this.stopped = true; } };
            sources.push(source); return source;
        },
    };
    const queue = new AudioQueue(context);
    queue.enqueue(encodePCM(chunk)); queue.enqueue(encodePCM(chunk));
    assert.ok(sources[1].startTime > sources[0].startTime);
    queue.clear();
    assert.ok(sources.every(source => source.stopped));
    assert.equal(queue.sources.size, 0);
    queue.enqueue(encodePCM(chunk));
    assert.equal(sources[2].startTime, 10.02);
});

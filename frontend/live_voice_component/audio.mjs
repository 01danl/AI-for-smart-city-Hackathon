/** Streaming conversion keeps fractional resampling position across worklet blocks. */
export class PCMResampler {
    constructor(inputRate, outputRate = 16000, chunkSize = 2048) {
        this.ratio = inputRate / outputRate;
        this.chunkSize = chunkSize;
        this.pending = new Float32Array(0);
        this.position = 0;
        this.output = [];
    }
    push(samples) {
        const joined = new Float32Array(this.pending.length + samples.length);
        joined.set(this.pending);
        joined.set(samples, this.pending.length);
        const chunks = [];
        while (this.position + 1 < joined.length) {
            const index = Math.floor(this.position);
            const fraction = this.position - index;
            const value = joined[index] * (1 - fraction) + joined[index + 1] * fraction;
            this.output.push(Math.round(Math.max(-1, Math.min(1, value)) * (value < 0 ? 32768 : 32767)));
            this.position += this.ratio;
            if (this.output.length === this.chunkSize) {
                const bytes = new ArrayBuffer(this.chunkSize * 2);
                const view = new DataView(bytes);
                this.output.forEach((value, i) => view.setInt16(i * 2, value, true));
                chunks.push(bytes);
                this.output = [];
            }
        }
        const consumed = Math.min(Math.floor(this.position), joined.length);
        this.pending = joined.slice(consumed);
        this.position -= consumed;
        return chunks;
    }
}

export function encodePCM(buffer) {
    const bytes = new Uint8Array(buffer);
    let text = "";
    for (const byte of bytes) text += String.fromCharCode(byte);
    return btoa(text);
}

export class AudioQueue {
    constructor(context) {
        this.context = context;
        this.sources = new Set();
        this.nextTime = 0;
    }
    enqueue(base64, rate = 24000) {
        const raw = atob(base64);
        const bytes = Uint8Array.from(raw, char => char.charCodeAt(0));
        const view = new DataView(bytes.buffer);
        const buffer = this.context.createBuffer(1, Math.floor(bytes.length / 2), rate);
        const channel = buffer.getChannelData(0);
        for (let i = 0; i < channel.length; i++) channel[i] = view.getInt16(i * 2, true) / 32768;
        const source = this.context.createBufferSource();
        source.buffer = buffer;
        source.connect(this.context.destination);
        this.sources.add(source);
        source.onended = () => this.sources.delete(source);
        const start = Math.max(this.context.currentTime + .02, this.nextTime);
        source.start(start);
        this.nextTime = start + buffer.duration;
    }
    clear() {
        for (const source of this.sources) { try { source.stop(); } catch { /* Already ended. */ } }
        this.sources.clear();
        this.nextTime = 0;
    }
}

import { PCMResampler } from "./audio.mjs";

class MicrophonePCM extends AudioWorkletProcessor {
    constructor() {
        super();
        this.resampler = new PCMResampler(sampleRate);
    }
    process(inputs) {
        const channels = inputs[0];
        if (channels?.length) {
            const mono = new Float32Array(channels[0].length);
            for (const channel of channels) {
                for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / channels.length;
            }
            for (const chunk of this.resampler.push(mono)) this.port.postMessage(chunk, [chunk]);
        }
        return true;
    }
}
registerProcessor("microphone-pcm", MicrophonePCM);

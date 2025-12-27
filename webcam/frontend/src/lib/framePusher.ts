import { get } from "svelte/store";
import { lcmLiveStatus, LCMLiveStatus } from "$lib/lcmLive";
import { onFrameChangeStore } from "$lib/mediaStream";

export class FramePusher {
    private running = false;
    private fps = 20;
    private loopHandle: number | null = null;
    private sendFunc: ((data: Blob) => void) | null = null;

    constructor(sendFunc: (data: Blob) => void) {
        this.sendFunc = sendFunc;
    }

    setFPS(fps: number) {
        this.fps = fps;
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.loop();
    }

    stop() {
        this.running = false;
        if (this.loopHandle) {
            cancelAnimationFrame(this.loopHandle);
            clearTimeout(this.loopHandle);
        }
    }

    private loop = () => {
        if (!this.running) return;

        const status = get(lcmLiveStatus);
        
        const canSend = status === LCMLiveStatus.CONNECTED || status === LCMLiveStatus.SEND_FRAME;

        if (canSend) {
            const frameBlob = get(onFrameChangeStore)?.blob;
            if (frameBlob && frameBlob.size > 0 && this.sendFunc) {
                this.sendFunc(frameBlob);
            }
        }

        const delay = 1000 / this.fps;
        
        this.loopHandle = setTimeout(() => {
            requestAnimationFrame(this.loop);
        }, delay) as unknown as number;
    };
}
import { get, writable } from 'svelte/store';

export enum LCMLiveStatus {
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
    PAUSED = "paused",
    WAIT = "wait",
    SEND_FRAME = "send_frame",
    TIMEOUT = "timeout",
}

const initStatus: LCMLiveStatus = LCMLiveStatus.DISCONNECTED;

export const lcmLiveStatus = writable<LCMLiveStatus>(initStatus);
export const streamId = writable<string | null>(null);

export const latestFrameUrl = writable<string | null>(null);

let websocket: WebSocket | null = null;
let userId: string | null = null;

export const lcmLiveActions = {
    async start() {
        return new Promise((resolve, reject) => {

            try {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                    websocket.send(JSON.stringify({ status: "resume", timestamp: Date.now() }));
                    streamId.set(userId);
                    resolve({ status: "connected"});
                } else {
                    websocket = null;
                }

                userId = crypto.randomUUID();
                const websocketURL = `${window.location.protocol === "https:" ? "wss" : "ws"
                    }://${window.location.host}/api/ws/${userId}`;

                websocket = new WebSocket(websocketURL);
                
                websocket.onopen = () => {
                    console.log("Connected to websocket");
                };

                websocket.onclose = () => {
                    lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                    console.log("Disconnected from websocket");
                };

                websocket.onerror = (err) => {
                    console.error(err);
                };

                websocket.onmessage = (event) => {
                    if (event.data instanceof Blob) {
                        const oldUrl = get(latestFrameUrl);
                        if (oldUrl) {
                            URL.revokeObjectURL(oldUrl);
                        }

                        const blob = new Blob([event.data], { type: "image/jpeg" });
                        
                        const newUrl = URL.createObjectURL(event.data);
                        latestFrameUrl.set(newUrl);
                        return;
                    }

                    try {
                        const data = JSON.parse(event.data);
                        switch (data.status) {
                            case "connected":
                                lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                                streamId.set(userId);
                                resolve({ status: "connected", userId });
                                break;
                            case "send_frame":
                                // 后端说可以发了，更新状态即可，FramePusher 会监听这个状态
                                if (get(lcmLiveStatus) !== LCMLiveStatus.PAUSED) {
                                    lcmLiveStatus.set(LCMLiveStatus.SEND_FRAME);
                                }
                                break;
                            case "wait":
                                if (get(lcmLiveStatus) !== LCMLiveStatus.PAUSED) {
                                    lcmLiveStatus.set(LCMLiveStatus.WAIT);
                                }
                                break;
                            case "timeout":
                                console.log("timeout");
                                lcmLiveStatus.set(LCMLiveStatus.TIMEOUT);
                                streamId.set(null);
                                reject(new Error("timeout"));
                                break;
                            case "error":
                                console.log(data.message);
                                lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                                streamId.set(null);
                                reject(new Error(data.message));
                                break;
                        }
                    } catch (err) {
                        console.error("Unknown message type:", err);
                    }
                };

            } catch (err) {
                console.error(err);
                lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                streamId.set(null);
                reject(err);
            }
        });
    },

    send(data: Blob | { [key: string]: any }) {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (data instanceof Blob) {
                websocket.send(data);
            } else {
                websocket.send(JSON.stringify(data));
            }
        } else {
            console.log("WebSocket not connected");
        }
    },

    async stop() {
        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
        if (websocket) {
            websocket.close();
        }
        websocket = null;
        streamId.set(null);
        
        const oldUrl = get(latestFrameUrl);
        if (oldUrl) {
            URL.revokeObjectURL(oldUrl);
            latestFrameUrl.set(null);
        }
    },

    async pause() {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ status: "pause", timestamp: Date.now() }));
        }
        lcmLiveStatus.set(LCMLiveStatus.PAUSED);
        streamId.set(null);
    },
};
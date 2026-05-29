type ProfilePrimitive = string | number | boolean | null;
type ProfileValue = ProfilePrimitive | ProfilePrimitive[] | Record<string, unknown>;
type ProfileData = Record<string, ProfileValue | undefined>;

interface ProfileEvent {
    t: number;
    frame: number | null;
    name: string;
    data?: ProfileData;
}

interface ProfileFrame {
    t: number;
    frame: number;
    data: ProfileData;
}

interface ProfileExport {
    metadata: {
        href?: string;
        userAgent?: string;
        startedAt: string;
        exportedAt: string;
        timeOrigin: number;
        frames: number;
        events: number;
    };
    frames: ProfileFrame[];
    events: ProfileEvent[];
}

interface ProfilerWindowApi {
    readonly enabled: boolean;
    enable: () => void;
    disable: () => void;
    clear: () => void;
    export: () => ProfileExport;
    download: () => void;
    event: (name: string, data?: ProfileData) => void;
    summary: () => ProfileData;
}

interface ProfilableDevice {
    on: (name: string, callback: (event?: { timestamp?: number }) => void) => void;
}

declare global {
    interface Window {
        __splatProfiler?: ProfilerWindowApi;
    }
}

const absNow = () => {
    if (typeof performance === 'undefined') {
        return Date.now();
    }
    return performance.timeOrigin + performance.now();
};

const localNow = () => {
    if (typeof performance === 'undefined') {
        return Date.now();
    }
    return performance.now();
};

const readEnabledFlag = () => {
    if (typeof window === 'undefined') {
        return false;
    }

    const params = new URLSearchParams(window.location.search);
    const queryValue = params.get('profile');
    if (queryValue !== null) {
        return queryValue !== '0' && queryValue !== 'false' && queryValue !== 'off';
    }

    try {
        return window.localStorage.getItem('supersplat.profile') === '1';
    } catch {
        return false;
    }
};

class SplatProfiler {
    enabled = readEnabledFlag();
    frames: ProfileFrame[] = [];
    events: ProfileEvent[] = [];
    currentFrame: ProfileFrame | null = null;
    startedAt = new Date().toISOString();
    maxFrames = 36000;
    maxEvents = 120000;
    private shaderCompileStarts: number[] = [];
    private shaderLinkStarts: number[] = [];
    private attachedDevices = new WeakSet<object>();

    now() {
        return localNow();
    }

    absNow() {
        return absNow();
    }

    enable() {
        this.enabled = true;
        this.event('profiler.enabled');
    }

    disable() {
        this.event('profiler.disabled');
        this.enabled = false;
    }

    clear() {
        this.frames = [];
        this.events = [];
        this.currentFrame = null;
        this.shaderCompileStarts = [];
        this.shaderLinkStarts = [];
        this.startedAt = new Date().toISOString();
    }

    startFrame(frame: number, data: ProfileData = {}) {
        if (!this.enabled) {
            return;
        }

        const current = {
            t: localNow(),
            frame,
            data: { ...data }
        };

        this.currentFrame = current;
        this.frames.push(current);
        if (this.frames.length > this.maxFrames) {
            this.frames.shift();
        }
    }

    setFrameValues(data: ProfileData) {
        if (!this.enabled || !this.currentFrame) {
            return;
        }

        Object.assign(this.currentFrame.data, data);
    }

    addFrameValue(name: string, value: number) {
        if (!this.enabled || !this.currentFrame || !Number.isFinite(value)) {
            return;
        }

        const current = this.currentFrame.data[name];
        this.currentFrame.data[name] = (typeof current === 'number' ? current : 0) + value;
    }

    maxFrameValue(name: string, value: number) {
        if (!this.enabled || !this.currentFrame || !Number.isFinite(value)) {
            return;
        }

        const current = this.currentFrame.data[name];
        this.currentFrame.data[name] = typeof current === 'number' ? Math.max(current, value) : value;
    }

    event(name: string, data?: ProfileData) {
        if (!this.enabled) {
            return;
        }

        this.events.push({
            t: localNow(),
            frame: this.currentFrame?.frame ?? null,
            name,
            data
        });

        if (this.events.length > this.maxEvents) {
            this.events.shift();
        }
    }

    duration(name: string, start: number, data: ProfileData = {}) {
        const ms = localNow() - start;
        this.event(name, { ...data, ms });
        return ms;
    }

    attachGraphicsDevice(device: ProfilableDevice) {
        if (!device || this.attachedDevices.has(device)) {
            return;
        }

        this.attachedDevices.add(device);

        device.on('shader:compile:start', (event) => {
            if (!this.enabled) {
                return;
            }
            this.shaderCompileStarts.push(event?.timestamp ?? localNow());
        });

        device.on('shader:compile:end', (event) => {
            if (!this.enabled) {
                return;
            }
            const start = this.shaderCompileStarts.pop();
            const end = event?.timestamp ?? localNow();
            const ms = start === undefined ? 0 : end - start;
            this.addFrameValue('shaderCompileMs', ms);
            this.event('shader.compile', { ms });
        });

        device.on('shader:link:start', (event) => {
            if (!this.enabled) {
                return;
            }
            this.shaderLinkStarts.push(event?.timestamp ?? localNow());
        });

        device.on('shader:link:end', (event) => {
            if (!this.enabled) {
                return;
            }
            const start = this.shaderLinkStarts.pop();
            const end = event?.timestamp ?? localNow();
            const ms = start === undefined ? 0 : end - start;
            this.addFrameValue('shaderLinkMs', ms);
            this.event('shader.link', { ms });
        });
    }

    export(): ProfileExport {
        return {
            metadata: {
                href: typeof window !== 'undefined' ? window.location.href : undefined,
                userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
                startedAt: this.startedAt,
                exportedAt: new Date().toISOString(),
                timeOrigin: typeof performance !== 'undefined' ? performance.timeOrigin : 0,
                frames: this.frames.length,
                events: this.events.length
            },
            frames: this.frames.slice(),
            events: this.events.slice()
        };
    }

    download() {
        if (typeof document === 'undefined') {
            return;
        }

        const blob = new Blob([JSON.stringify(this.export(), null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `supersplat-profile-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    summary(): ProfileData {
        const last = this.frames[this.frames.length - 1]?.data ?? {};
        return {
            enabled: this.enabled,
            frames: this.frames.length,
            events: this.events.length,
            last
        };
    }
}

const profiler = new SplatProfiler();

if (typeof window !== 'undefined') {
    window.__splatProfiler = {
        get enabled() {
            return profiler.enabled;
        },
        enable: () => profiler.enable(),
        disable: () => profiler.disable(),
        clear: () => profiler.clear(),
        export: () => profiler.export(),
        download: () => profiler.download(),
        event: (name, data) => profiler.event(name, data),
        summary: () => profiler.summary()
    };
}

export { profiler };
export type { ProfileData };

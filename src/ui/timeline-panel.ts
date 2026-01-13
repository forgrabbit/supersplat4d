import { Button, Container, NumericInput, SelectInput } from '@playcanvas/pcui';

import { Events } from '../events';
import { localize } from './localization';
import { Tooltips } from './tooltips';

class Ticks extends Container {
    constructor(events: Events, tooltips: Tooltips, args = {}) {
        args = {
            ...args,
            id: 'ticks'
        };

        super(args);

        const workArea = new Container({
            id: 'ticks-area'
        });

        this.append(workArea);

        let addKey: (value: number) => void;
        let removeKey: (index: number) => void;
        let frameFromOffset: (offset: number) => number;
        let moveCursor: (frame: number) => void;

        // rebuild the timeline
        const rebuild = () => {
            console.log('🔄 Timeline rebuild() called');
            console.log('  Has timeline.frames?', events.functions.has('timeline.frames'));
            console.log('  Has timeline.frame?', events.functions.has('timeline.frame'));
            
            // clear existing labels
            workArea.dom.innerHTML = '';

            // Check if timeline functions are registered before invoking
            const numFrames = events.invoke('timeline.frames') ?? 180;
            const currentFrame = events.invoke('timeline.frame') ?? 0;

            const padding = 20;
            const width = this.dom.getBoundingClientRect().width - padding * 2;
            const labelStep = Math.max(1, Math.floor(numFrames / Math.max(1, Math.floor(width / 50))));
            const numLabels = Math.max(1, Math.ceil(numFrames / labelStep));

            const offsetFromFrame = (frame: number) => {
                return padding + Math.floor(frame / (numFrames - 1) * width);
            };

            frameFromOffset = (offset: number) => {
                return Math.max(0, Math.min(numFrames - 1, Math.floor((offset - padding) / width * (numFrames - 1))));
            };

            // timeline labels

            for (let i = 0; i < numLabels; i++) {
                const thisFrame = Math.floor(i * labelStep);
                const label = document.createElement('div');
                label.classList.add('time-label');
                label.style.left = `${offsetFromFrame(thisFrame)}px`;
                label.textContent = thisFrame.toString();
                workArea.dom.appendChild(label);
            }

            // keys

            const keys: HTMLElement[] = [];
            const createKey = (value: number) => {
                const label = document.createElement('div');
                label.classList.add('time-label', 'key');
                label.style.left = `${offsetFromFrame(value)}px`;
                let dragging = false;
                let toFrame = -1;

                label.addEventListener('pointerdown', (event) => {
                    if (!dragging && event.isPrimary) {
                        dragging = true;
                        label.classList.add('dragging');
                        label.setPointerCapture(event.pointerId);
                        event.stopPropagation();
                    }
                });

                label.addEventListener('pointermove', (event: PointerEvent) => {
                    if (dragging) {
                        toFrame = frameFromOffset(parseInt(label.style.left, 10) + event.offsetX);
                        label.style.left = `${offsetFromFrame(toFrame)}px`;
                    }
                });

                label.addEventListener('pointerup', (event: PointerEvent) => {
                    if (dragging && event.isPrimary) {
                        const fromIndex = keys.indexOf(label);
                        const fromFrame = events.invoke('timeline.keys')[fromIndex];
                        if (fromFrame !== toFrame) {
                            events.fire('timeline.move', fromFrame, toFrame);
                            events.fire('timeline.frame', events.invoke('timeline.frame'));
                        }

                        label.releasePointerCapture(event.pointerId);
                        label.classList.remove('dragging');
                        dragging = false;
                    }
                });

                workArea.dom.appendChild(label);
                keys.push(label);
            };

            const timelineKeys = events.invoke('timeline.keys') as number[];
            if (timelineKeys && Array.isArray(timelineKeys)) {
                timelineKeys.forEach(createKey);
            }

            addKey = (value: number) => {
                createKey(value);
            };

            removeKey = (index: number) => {
                workArea.dom.removeChild(keys[index]);
                keys.splice(index, 1);
            };

            // cursor

            const cursor = document.createElement('div');
            cursor.classList.add('time-label', 'cursor');
            cursor.style.left = `${offsetFromFrame(currentFrame)}px`;
            cursor.textContent = currentFrame.toString();
            workArea.dom.appendChild(cursor);

            moveCursor = (frame: number) => {
                cursor.style.left = `${offsetFromFrame(frame)}px`;
                cursor.textContent = frame.toString();
            };
        };

        // handle scrubbing

        let scrubbing = false;

        workArea.dom.addEventListener('pointerdown', (event: PointerEvent) => {
            if (!scrubbing && event.isPrimary) {
                scrubbing = true;
                workArea.dom.setPointerCapture(event.pointerId);
                events.fire('timeline.setFrame', frameFromOffset(event.offsetX));
            }
        });

        workArea.dom.addEventListener('pointermove', (event: PointerEvent) => {
            if (scrubbing) {
                events.fire('timeline.setFrame', frameFromOffset(event.offsetX));
            }
        });

        workArea.dom.addEventListener('pointerup', (event: PointerEvent) => {
            if (scrubbing && event.isPrimary) {
                workArea.dom.releasePointerCapture(event.pointerId);
                scrubbing = false;
            }
        });

        // rebuild the timeline on dom resize
        // Delay initial observation to ensure timeline events are registered
        const resizeObserver = new ResizeObserver(() => {
            // Only rebuild if timeline functions are available
            if (events.functions.has('timeline.frames')) {
                rebuild();
            }
        });
        // Use requestAnimationFrame to delay observation until after timeline registration
        console.log('⏳ Scheduling timeline rebuild via requestAnimationFrame...');
        requestAnimationFrame(() => {
            console.log('🎬 requestAnimationFrame callback executing');
            resizeObserver.observe(workArea.dom);
            // Initial rebuild after timeline is registered
            if (events.functions.has('timeline.frames')) {
                console.log('✅ timeline.frames exists, calling rebuild()');
                rebuild();
            } else {
                console.warn('⚠️ timeline.frames NOT registered yet, skipping rebuild()');
            }
        });

        // rebuild when timeline frames change
        events.on('timeline.frames', () => {
            rebuild();
        });

        events.on('timeline.frame', (frame: number) => {
            moveCursor(frame);
        });

        events.on('timeline.keyAdded', (value: number) => {
            addKey(value);
        });

        events.on('timeline.keyRemoved', (index: number) => {
            removeKey(index);
        });
    }
}

class TimelinePanel extends Container {
    constructor(events: Events, tooltips: Tooltips, args: { isMobile?: boolean } = {}) {
        const isMobile = args.isMobile || false;
        const containerArgs = {
            ...args,
            id: 'timeline-panel'
        };
        // Remove isMobile from containerArgs
        delete (containerArgs as any).isMobile;

        super(containerArgs);
        
        // Add mobile class for CSS targeting
        if (isMobile) {
            this.dom.classList.add('mobile-timeline');
        }

        // play controls
        const play = new Button({
            class: 'button',
            text: '\uE131'
        });

        // Desktop-only buttons
        let prev: Button | null = null;
        let next: Button | null = null;
        let addKey: Button | null = null;
        let removeKey: Button | null = null;

        if (!isMobile) {
            prev = new Button({
                class: 'button',
                text: '\uE162'
            });

            next = new Button({
                class: 'button',
                text: '\uE164'
            });

            addKey = new Button({
                class: 'button',
                text: '\uE120'
            });

            removeKey = new Button({
                class: 'button',
                text: '\uE121',
                enabled: false
            });
        }

        const buttonControls = new Container({
            id: 'button-controls'
        });
        
        // On mobile, only show play button; on desktop, show all buttons
        if (isMobile) {
            buttonControls.append(play);
        } else {
            buttonControls.append(prev!);
            buttonControls.append(play);
            buttonControls.append(next!);
            buttonControls.append(addKey!);
            buttonControls.append(removeKey!);
        }

        // settings

        const speed = new SelectInput({
            id: 'speed',
            defaultValue: 30,
            options: [
                { v: 1, t: '1 fps' },
                { v: 6, t: '6 fps' },
                { v: 12, t: '12 fps' },
                { v: 24, t: '24 fps' },
                { v: 30, t: '30 fps' },
                { v: 60, t: '60 fps' }
            ]
        });

        speed.on('change', (value: string) => {
            events.fire('timeline.setFrameRate', parseInt(value, 10));
        });

        events.on('timeline.frameRate', (frameRate: number) => {
            speed.value = frameRate.toString();
        });

        // Desktop-only settings
        let frames: NumericInput | null = null;
        let smoothness: NumericInput | null = null;

        if (!isMobile) {
            frames = new NumericInput({
                id: 'totalFrames',
                value: 180,
                min: 1,
                max: 10000,
                precision: 0
            });

            frames.on('change', (value: number) => {
                events.fire('timeline.setFrames', value);
            });

            events.on('timeline.frames', (framesIn: number) => {
                frames!.value = framesIn;
            });

            smoothness = new NumericInput({
                id: 'smoothness',
                min: 0,
                max: 1,
                step: 0.05,
                value: 1
            });

            smoothness.on('change', (value: number) => {
                events.fire('timeline.setSmoothness', value);
            });

            events.on('timeline.smoothness', (smoothnessIn: number) => {
                smoothness!.value = smoothnessIn;
            });
        }

        const settingsControls = new Container({
            id: 'settings-controls'
        });
        
        // On mobile, only show frame rate (speed); on desktop, show all settings
        settingsControls.append(speed);
        if (!isMobile) {
            settingsControls.append(frames!);
            settingsControls.append(smoothness!);
        }

        // append control groups

        const controlsWrap = new Container({
            id: 'controls-wrap'
        });

        // Layout: play button in center, settings on right (same for mobile and desktop)
        const spacerL = new Container({
            class: 'spacer'
        });
        const spacerR = new Container({
            class: 'spacer'
        });
        spacerR.append(settingsControls);
        
        controlsWrap.append(spacerL);
        controlsWrap.append(buttonControls);
        controlsWrap.append(spacerR);

        const ticks = new Ticks(events, tooltips);

        this.append(controlsWrap);
        this.append(ticks);

        // ui handlers

        const skip = (dir: 'forward' | 'back') => {
            const orderedKeys = (events.invoke('timeline.keys') as number[]).map((frame, index) => {
                return { frame, index };
            }).sort((a, b) => a.frame - b.frame);

            if (orderedKeys.length > 0) {
                const frame = events.invoke('timeline.frame');
                const nextKey = orderedKeys.findIndex(k => (dir === 'back' ? k.frame >= frame : k.frame > frame));
                const l = orderedKeys.length;

                if (nextKey === -1) {
                    events.fire('timeline.setFrame', orderedKeys[dir === 'back' ? l - 1 : 0].frame);
                } else {
                    events.fire('timeline.setFrame', orderedKeys[dir === 'back' ? (nextKey + l - 1) % l : nextKey].frame);
                }
            } else {
                // if there are no keys, just to start of timeline or end
                if (dir === 'back') {
                    events.fire('timeline.setFrame', 0);
                } else {
                    const maxFrames = events.invoke('timeline.frames');
                    if (maxFrames !== undefined) {
                        events.fire('timeline.setFrame', maxFrames - 1);
                    }
                }
            }
        };

        // Only attach event handlers for buttons that exist (desktop only)
        if (!isMobile && prev && next && addKey && removeKey) {
            prev.on('click', () => {
                skip('back');
            });

            next.on('click', () => {
                skip('forward');
            });

            addKey.on('click', () => {
                events.fire('timeline.add', events.invoke('timeline.frame'));
            });

            removeKey.on('click', () => {
                const index = events.invoke('timeline.keys').indexOf(events.invoke('timeline.frame'));
                if (index !== -1) {
                    events.fire('timeline.remove', index);
                    events.fire('timeline.frame', events.invoke('timeline.frame'));
                }
            });

            const canDelete = (frame: number) => events.invoke('timeline.keys').includes(frame);

            events.on('timeline.frame', (frame: number) => {
                removeKey.enabled = canDelete(frame);
            });

            events.on('timeline.keyRemoved', (index: number) => {
                removeKey.enabled = canDelete(events.invoke('timeline.frame'));
            });

            events.on('timeline.keyAdded', (frame: number) => {
                removeKey.enabled = canDelete(frame);
            });
        }

        play.on('click', () => {
            if (events.invoke('timeline.playing')) {
                events.fire('timeline.setPlaying', false);
                play.text = '\uE131';
            } else {
                events.fire('timeline.setPlaying', true);
                play.text = '\uE135';
            }
        });

        // cancel animation playback if user interacts with camera
        events.on('camera.controller', (type: string) => {
            if (events.invoke('timeline.playing')) {
                // stop
            }
        });

        // tooltips
        tooltips.register(play, localize('tooltip.timeline.play'), 'top');
        tooltips.register(speed, localize('tooltip.timeline.frame-rate'), 'top');
        
        if (!isMobile && prev && next && addKey && removeKey && frames && smoothness) {
            tooltips.register(prev, localize('tooltip.timeline.prev-key'), 'top');
            tooltips.register(next, localize('tooltip.timeline.next-key'), 'top');
            tooltips.register(addKey, localize('tooltip.timeline.add-key'), 'top');
            tooltips.register(removeKey, localize('tooltip.timeline.remove-key'), 'top');
            tooltips.register(frames, localize('tooltip.timeline.total-frames'), 'top');
            tooltips.register(smoothness, localize('tooltip.timeline.smoothness'), 'top');
        }
    }
}

export { TimelinePanel };

/**
 * Dialog for inputting dynamic gaussian parameters
 * when loading a PLY file that doesn't have cfg_args in the header
 */

import { Button, Container, Label, NumericInput } from '@playcanvas/pcui';

import { Events } from '../events';
import type { DynamicPlyParams } from '../loaders/dynamic-ply';

class DynamicParamsDialog extends Container {
    show: (filename: string) => Promise<DynamicPlyParams | null>;
    hide: () => void;
    destroy: () => void;

    constructor(events: Events, args = {}) {
        args = {
            ...args,
            id: 'dynamic-params-dialog',
            class: 'settings-dialog',
            hidden: true,
            tabIndex: -1
        };

        super(args);

        const dialog = new Container({
            id: 'dialog'
        });

        // header
        const headerText = new Label({ id: 'text', text: 'DYNAMIC GAUSSIAN PARAMETERS' });
        const header = new Container({ id: 'header' });
        header.append(headerText);

        // filename display
        const filenameLabel = new Label({ class: 'label', text: 'File:' });
        const filenameValue = new Label({ class: 'value', text: '' });
        const filenameRow = new Container({ class: 'row' });
        filenameRow.append(filenameLabel);
        filenameRow.append(filenameValue);

        // info text
        const infoText = new Label({
            class: 'label',
            text: 'This PLY file contains dynamic gaussian data but no timing parameters. Please enter the parameters:'
        });
        infoText.style.width = '100%';
        infoText.style.whiteSpace = 'normal';
        const infoRow = new Container({ class: 'row' });
        infoRow.append(infoText);

        // start time
        const startLabel = new Label({ class: 'label', text: 'Start Time (s):' });
        const startInput = new NumericInput({
            class: 'slider',
            value: 0,
            min: 0,
            max: 1000,
            precision: 2,
            step: 0.1
        });
        const startRow = new Container({ class: 'row' });
        startRow.append(startLabel);
        startRow.append(startInput);

        // duration
        const durationLabel = new Label({ class: 'label', text: 'Duration (s):' });
        const durationInput = new NumericInput({
            class: 'slider',
            value: 2.0,
            min: 0.1,
            max: 1000,
            precision: 2,
            step: 0.1
        });
        const durationRow = new Container({ class: 'row' });
        durationRow.append(durationLabel);
        durationRow.append(durationInput);

        // fps
        const fpsLabel = new Label({ class: 'label', text: 'FPS:' });
        const fpsInput = new NumericInput({
            class: 'slider',
            value: 30,
            min: 1,
            max: 120,
            precision: 0,
            step: 1
        });
        const fpsRow = new Container({ class: 'row' });
        fpsRow.append(fpsLabel);
        fpsRow.append(fpsInput);

        // content
        const content = new Container({ id: 'content' });
        content.append(filenameRow);
        content.append(infoRow);
        content.append(startRow);
        content.append(durationRow);
        content.append(fpsRow);

        // buttons
        const okButton = new Button({
            class: 'button',
            text: 'Load'
        });

        const cancelButton = new Button({
            class: 'button',
            text: 'Cancel'
        });

        const buttons = new Container({ id: 'footer' });
        buttons.append(cancelButton);
        buttons.append(okButton);

        dialog.append(header);
        dialog.append(content);
        dialog.append(buttons);

        this.append(dialog);

        // Promise resolver
        let resolvePromise: (value: DynamicPlyParams | null) => void;

        // Event handlers
        okButton.on('click', () => {
            const params: DynamicPlyParams = {
                start: startInput.value,
                duration: durationInput.value,
                fps: fpsInput.value,
                sh_degree: 0
            };
            this.hidden = true;
            resolvePromise(params);
        });

        cancelButton.on('click', () => {
            this.hidden = true;
            resolvePromise(null);
        });

        // Click outside to cancel
        this.dom.addEventListener('click', (e: MouseEvent) => {
            if (e.target === this.dom) {
                this.hidden = true;
                resolvePromise(null);
            }
        });

        // Escape key to cancel
        this.dom.addEventListener('keydown', (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                this.hidden = true;
                resolvePromise(null);
            } else if (e.key === 'Enter') {
                const params: DynamicPlyParams = {
                    start: startInput.value,
                    duration: durationInput.value,
                    fps: fpsInput.value,
                    sh_degree: 0
                };
                this.hidden = true;
                resolvePromise(params);
            }
        });

        this.show = (filename: string) => {
            return new Promise<DynamicPlyParams | null>((resolve) => {
                resolvePromise = resolve;
                filenameValue.text = filename;
                this.hidden = false;
                this.dom.focus();
            });
        };

        this.hide = () => {
            this.hidden = true;
        };

        this.destroy = () => {
            this.dom.remove();
        };
    }
}

export { DynamicParamsDialog };

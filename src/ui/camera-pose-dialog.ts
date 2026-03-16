import { BooleanInput, Button, Container, Label, TextAreaInput } from '@playcanvas/pcui';

import type { Events } from '../events';

class CameraPoseDialog extends Container {
    show: (initialJson: string, splatName: string) => Promise<{ json: string, sibrExact: boolean } | null>;
    hide: () => void;
    destroy: () => void;

    constructor(events: Events, args = {}) {
        args = {
            ...args,
            id: 'camera-pose-dialog',
            class: 'settings-dialog',
            hidden: true,
            tabIndex: -1
        };

        super(args);

        const dialog = new Container({
            id: 'dialog'
        });

        const headerText = new Label({ id: 'text', text: 'DEFAULT CAMERA JSON' });
        const header = new Container({ id: 'header' });
        header.append(headerText);

        const splatLabel = new Label({ class: 'label', text: 'Splat:' });
        const splatNameLabel = new Label({ class: 'value', text: '' });
        const splatRow = new Container({ class: 'row' });
        splatRow.append(splatLabel);
        splatRow.append(splatNameLabel);

        const infoText = new Label({
            class: 'label',
            text: 'Edit the camera JSON used as the default view for this splat.'
        });
        infoText.style.width = '100%';
        infoText.style.whiteSpace = 'normal';
        const infoRow = new Container({ class: 'row' });
        infoRow.append(infoText);

        const textArea = new TextAreaInput({
            class: 'textarea',
            value: ''
        });
        textArea.style.width = '100%';
        textArea.style.height = '220px';
        const textRow = new Container({ class: 'row' });
        textRow.append(textArea);

        const sibrLabel = new Label({
            class: 'label',
            text: 'Enable SIBR exact mode (experimental, uses training camera as-is)'
        });
        sibrLabel.style.width = '100%';
        sibrLabel.style.whiteSpace = 'normal';
        const sibrToggle = new BooleanInput({
            class: 'boolean',
            value: true
        });
        const sibrRow = new Container({ class: 'row' });
        sibrRow.append(sibrLabel);
        sibrRow.append(sibrToggle);

        const content = new Container({ id: 'content' });
        content.append(splatRow);
        content.append(infoRow);
        content.append(textRow);
        content.append(sibrRow);

        const okButton = new Button({
            class: 'button',
            text: 'Apply'
        });

        const cancelButton = new Button({
            class: 'button',
            text: 'Cancel'
        });

        const resetSibrButton = new Button({
            class: 'button',
            text: 'Reset to training view'
        });

        const buttons = new Container({ id: 'footer' });
        buttons.append(cancelButton);
        buttons.append(resetSibrButton);
        buttons.append(okButton);

        dialog.append(header);
        dialog.append(content);
        dialog.append(buttons);

        this.append(dialog);

        let resolvePromise: (value: { json: string, sibrExact: boolean } | null) => void;

        okButton.on('click', () => {
            this.hidden = true;
            resolvePromise({
                json: textArea.value as string,
                sibrExact: !!sibrToggle.value
            });
        });

        cancelButton.on('click', () => {
            this.hidden = true;
            resolvePromise(null);
        });

        resetSibrButton.on('click', () => {
            events.fire('camera.resetSibrView');
        });

        this.dom.addEventListener('click', (e: MouseEvent) => {
            if (e.target === this.dom) {
                this.hidden = true;
                resolvePromise(null);
            }
        });

        this.dom.addEventListener('keydown', (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                this.hidden = true;
                resolvePromise(null);
            } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.hidden = true;
                resolvePromise({
                    json: textArea.value as string,
                    sibrExact: !!sibrToggle.value
                });
            }
        });

        this.show = (initialJson: string, splatName: string) => {
            return new Promise<{ json: string, sibrExact: boolean } | null>((resolve) => {
                resolvePromise = resolve;
                splatNameLabel.text = splatName;
                textArea.value = initialJson;
                sibrToggle.value = true;
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

export { CameraPoseDialog };


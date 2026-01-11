/**
 * Dialog for exporting dynamic gaussian splatting data
 * Supports PLY and SOG4D formats with time range selection
 */

import { Button, Container, Label, NumericInput, SelectInput } from '@playcanvas/pcui';

import { Events } from '../events';

interface DynamicExportOptions {
    start: number;
    duration: number;
    format: 'ply' | 'sog4d';
    filename: string;
}

interface DynamicExportParams {
    originalStart: number;
    originalDuration: number;
    fps: number;
    filename: string;
}

class DynamicExportDialog extends Container {
    show: (params: DynamicExportParams) => Promise<DynamicExportOptions | null>;
    hide: () => void;
    destroy: () => void;

    constructor(events: Events, args = {}) {
        args = {
            ...args,
            id: 'dynamic-export-dialog',
            class: 'settings-dialog',
            hidden: true,
            tabIndex: -1
        };

        super(args);

        const dialog = new Container({
            id: 'dialog'
        });

        // header
        const headerText = new Label({ id: 'text', text: 'EXPORT DYNAMIC GAUSSIAN' });
        const header = new Container({ id: 'header' });
        header.append(headerText);

        // filename
        const filenameLabel = new Label({ class: 'label', text: 'Filename:' });
        const filenameValue = new Label({ class: 'value', text: '' });
        const filenameRow = new Container({ class: 'row' });
        filenameRow.append(filenameLabel);
        filenameRow.append(filenameValue);

        // format selection
        const formatLabel = new Label({ class: 'label', text: 'Format:' });
        const formatSelect = new SelectInput({
            class: 'select',
            defaultValue: 'ply',
            options: [
                { v: 'ply', t: 'PLY (.ply) - Uncompressed' },
                { v: 'sog4d', t: 'SOG4D (.sog4d) - Compressed' }
            ]
        });
        const formatRow = new Container({ class: 'row' });
        formatRow.append(formatLabel);
        formatRow.append(formatSelect);

        // SOG4D warning
        const warningText = new Label({
            class: 'label',
            text: '⚠️ SOG4D compression may take several minutes for large files'
        });
        warningText.style.color = '#ffaa00';
        warningText.style.width = '100%';
        const warningRow = new Container({ class: 'row' });
        warningRow.append(warningText);
        warningRow.hidden = true;

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

        // info about time range
        const infoText = new Label({
            class: 'label',
            text: 'Only splats visible within this time range will be exported.'
        });
        infoText.style.width = '100%';
        infoText.style.fontStyle = 'italic';
        infoText.style.opacity = '0.7';
        const infoRow = new Container({ class: 'row' });
        infoRow.append(infoText);

        // content
        const content = new Container({ id: 'content' });
        content.append(filenameRow);
        content.append(formatRow);
        content.append(warningRow);
        content.append(startRow);
        content.append(durationRow);
        content.append(infoRow);

        // buttons
        const exportButton = new Button({
            class: 'button',
            text: 'Export'
        });

        const cancelButton = new Button({
            class: 'button',
            text: 'Cancel'
        });

        const buttons = new Container({ id: 'footer' });
        buttons.append(cancelButton);
        buttons.append(exportButton);

        dialog.append(header);
        dialog.append(content);
        dialog.append(buttons);

        this.append(dialog);

        // State
        let resolvePromise: (value: DynamicExportOptions | null) => void;
        let currentParams: DynamicExportParams;

        // Show/hide warning based on format
        formatSelect.on('change', (value: string) => {
            warningRow.hidden = value !== 'sog4d';
        });

        // Validate and update time range
        const validateTimeRange = () => {
            const start = startInput.value;
            const duration = durationInput.value;
            const maxEnd = currentParams.originalStart + currentParams.originalDuration;
            
            // Clamp start
            if (start < currentParams.originalStart) {
                startInput.value = currentParams.originalStart;
            }
            if (start > maxEnd - 0.1) {
                startInput.value = maxEnd - 0.1;
            }
            
            // Update duration max based on current start
            const actualStart = startInput.value;
            const maxDuration = maxEnd - actualStart;
            durationInput.max = maxDuration;
            
            // Clamp duration
            if (duration > maxDuration) {
                durationInput.value = maxDuration;
            }
            if (duration < 0.1) {
                durationInput.value = 0.1;
            }
        };

        startInput.on('change', validateTimeRange);
        durationInput.on('change', validateTimeRange);

        // Event handlers
        exportButton.on('click', () => {
            const options: DynamicExportOptions = {
                start: startInput.value,
                duration: durationInput.value,
                format: formatSelect.value as 'ply' | 'sog4d',
                filename: currentParams.filename
            };
            this.hidden = true;
            resolvePromise(options);
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
                const options: DynamicExportOptions = {
                    start: startInput.value,
                    duration: durationInput.value,
                    format: formatSelect.value as 'ply' | 'sog4d',
                    filename: currentParams.filename
                };
                this.hidden = true;
                resolvePromise(options);
            }
        });

        this.show = (params: DynamicExportParams) => {
            return new Promise<DynamicExportOptions | null>((resolve) => {
                resolvePromise = resolve;
                currentParams = params;
                
                // Set defaults from original params
                filenameValue.text = params.filename;
                startInput.value = params.originalStart;
                startInput.min = params.originalStart;
                startInput.max = params.originalStart + params.originalDuration - 0.1;
                durationInput.value = params.originalDuration;
                durationInput.min = 0.1;
                durationInput.max = params.originalDuration;
                formatSelect.value = 'ply';
                warningRow.hidden = true;
                
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

export { DynamicExportDialog };
export type { DynamicExportOptions, DynamicExportParams };

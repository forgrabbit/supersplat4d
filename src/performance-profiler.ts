/**
 * Performance profiler for measuring frame rendering performance
 * Measures: position calculation, sorting, GPU rendering, and other operations
 */

export interface PerformanceMetrics {
    frame: number;
    timestamp: number;
    positionCalculation: number;  // Time to calculate dynamic gaussian positions (ms)
    sorting: number;                // Time for sorting to complete (ms)
    gpuRender: number;              // GPU rendering time (ms, measured with gl.finish())
    preRender: number;              // Pre-render operations time (ms)
    postRender: number;              // Post-render operations time (ms)
    totalFrame: number;              // Total frame time (ms)
}

class PerformanceProfiler {
    private enabled: boolean = false;
    private metrics: PerformanceMetrics[] = [];
    private currentFrame: PerformanceMetrics | null = null;
    private frameCounter: number = 0;
    private maxFrames: number = 1000; // Maximum frames to collect
    
    // Timing markers
    private positionCalcStart: number = 0;
    private positionCalcEnd: number = 0;
    private sortStart: number = 0;
    private sortEnd: number = 0;
    private preRenderStart: number = 0;
    private gpuRenderStart: number = 0;
    private gpuRenderEnd: number = 0;
    private postRenderEnd: number = 0;
    private frameStart: number = 0;
    
    // State-based saving mechanism
    private pendingAsyncOps: number = 0;  // Count of pending async operations
    private frameReadyToSave: boolean = false;  // Whether rendering is complete
    private frameStartTime: number = 0;  // For timeout detection

    enable() {
        this.enabled = true;
        this.metrics = [];
        this.frameCounter = 0;
        this.currentFrame = null;
        this.pendingAsyncOps = 0;
        this.frameReadyToSave = false;
        console.log('📊 Performance profiler enabled');
    }

    disable() {
        this.enabled = false;
        
        // Save the last frame if it exists
        if (this.currentFrame) {
            if (this.metrics.length < this.maxFrames) {
                this.metrics.push({ ...this.currentFrame });
            }
            this.currentFrame = null;
        }
        
        console.log('📊 Performance profiler disabled');
    }

    isEnabled(): boolean {
        return this.enabled;
    }

    startFrame() {
        if (!this.enabled) return;
        
        // Check if previous frame is stuck (timeout after 5 seconds)
        if (this.currentFrame && !this.frameReadyToSave) {
            const elapsed = performance.now() - this.frameStartTime;
            if (elapsed > 5000) {
                console.warn(`⚠️ Frame ${this.currentFrame.frame} timeout (${elapsed.toFixed(0)}ms), forcing save. pendingAsyncOps=${this.pendingAsyncOps}`);
                this.forceSaveCurrentFrame();
            }
        }
        
        // If previous frame is ready but not saved yet, try to save it
        if (this.currentFrame && this.frameReadyToSave && this.pendingAsyncOps === 0) {
            this.saveCurrentFrame();
        }
        
        // Only create new frame if current frame is saved or doesn't exist
        if (!this.currentFrame) {
            this.frameStart = performance.now();
            this.frameStartTime = this.frameStart;
            this.currentFrame = {
                frame: this.frameCounter++,
                timestamp: this.frameStart,
                positionCalculation: 0,
                sorting: 0,
                gpuRender: 0,
                preRender: 0,
                postRender: 0,
                totalFrame: 0
            };
            this.frameReadyToSave = false;
            this.pendingAsyncOps = 0;
        }
    }

    startPositionCalculation() {
        if (!this.enabled) return;
        this.positionCalcStart = performance.now();
    }

    endPositionCalculation() {
        if (!this.enabled || !this.currentFrame) return;
        this.positionCalcEnd = performance.now();
        this.currentFrame.positionCalculation = this.positionCalcEnd - this.positionCalcStart;
    }

    startSorting() {
        if (!this.enabled) return;
        this.sortStart = performance.now();
        this.pendingAsyncOps++;  // Increment async operation counter
    }

    endSorting() {
        if (!this.enabled || !this.currentFrame) return;
        
        // Check if startSorting was called
        if (this.sortStart === 0) {
            console.warn(`⚠️ endSorting() called but startSorting() was not called`);
            return;
        }
        
        this.sortEnd = performance.now();
        const sortingTime = this.sortEnd - this.sortStart;
        this.currentFrame.sorting = sortingTime;
        
        // Decrement async operation counter
        this.pendingAsyncOps--;
        
        // Reset sortStart
        this.sortStart = 0;
        
        // Try to save frame if all conditions are met
        this.tryToSaveFrame();
    }

    startPreRender() {
        if (!this.enabled) return;
        this.preRenderStart = performance.now();
    }

    startGpuRender() {
        if (!this.enabled) return;
        this.gpuRenderStart = performance.now();
    }

    endGpuRender() {
        if (!this.enabled || !this.currentFrame) return;
        this.gpuRenderEnd = performance.now();
        this.currentFrame.gpuRender = this.gpuRenderEnd - this.gpuRenderStart;
    }

    endPostRender() {
        if (!this.enabled || !this.currentFrame) return;
        this.postRenderEnd = performance.now();
        
        if (this.preRenderStart > 0) {
            this.currentFrame.preRender = this.gpuRenderStart - this.preRenderStart;
        }
        this.currentFrame.postRender = this.postRenderEnd - this.gpuRenderEnd;
        this.currentFrame.totalFrame = this.postRenderEnd - this.frameStart;
        
        // Mark frame as ready to save (rendering complete)
        this.frameReadyToSave = true;
        
        // Try to save frame if all async operations are complete
        this.tryToSaveFrame();
    }
    
    private tryToSaveFrame() {
        if (!this.enabled || !this.currentFrame) return;
        
        // Check if all conditions are met for saving
        const canSave = this.frameReadyToSave && this.pendingAsyncOps === 0;
        
        if (canSave) {
            this.saveCurrentFrame();
        }
    }
    
    private saveCurrentFrame() {
        if (!this.currentFrame) return;
        
        if (this.metrics.length < this.maxFrames) {
            this.metrics.push({ ...this.currentFrame });
        }
        
        // Clear current frame so a new one can be created
        this.currentFrame = null;
        this.frameReadyToSave = false;
    }
    
    private forceSaveCurrentFrame() {
        if (!this.currentFrame) return;
        
        console.warn(`⚠️ Force saving frame ${this.currentFrame.frame} due to timeout`);
        
        if (this.metrics.length < this.maxFrames) {
            this.metrics.push({ ...this.currentFrame });
        }
        
        // Clear current frame
        this.currentFrame = null;
        this.frameReadyToSave = false;
        this.pendingAsyncOps = 0;  // Reset counter
    }

    getMetrics(): PerformanceMetrics[] {
        return [...this.metrics];
    }

    getLatestMetrics(): PerformanceMetrics | null {
        return this.metrics.length > 0 ? this.metrics[this.metrics.length - 1] : null;
    }

    printSummary() {
        if (this.metrics.length === 0) {
            console.log('📊 No performance data collected');
            return;
        }

        const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
        const min = (arr: number[]) => Math.min(...arr);
        const max = (arr: number[]) => Math.max(...arr);
        const p95 = (arr: number[]) => {
            const sorted = [...arr].sort((a, b) => a - b);
            return sorted[Math.floor(sorted.length * 0.95)];
        };

        const positionCalc = this.metrics.map(m => m.positionCalculation);
        const sorting = this.metrics.map(m => m.sorting);
        const gpuRender = this.metrics.map(m => m.gpuRender);
        const preRender = this.metrics.map(m => m.preRender);
        const postRender = this.metrics.map(m => m.postRender);
        const totalFrame = this.metrics.map(m => m.totalFrame);

        console.log('📊 Performance Summary:');
        console.log(`  Frames measured: ${this.metrics.length}`);
        console.log('');
        console.log('Position Calculation (ms):');
        console.log(`  Avg: ${avg(positionCalc).toFixed(2)}, Min: ${min(positionCalc).toFixed(2)}, Max: ${max(positionCalc).toFixed(2)}, P95: ${p95(positionCalc).toFixed(2)}`);
        console.log('Sorting (ms):');
        console.log(`  Avg: ${avg(sorting).toFixed(2)}, Min: ${min(sorting).toFixed(2)}, Max: ${max(sorting).toFixed(2)}, P95: ${p95(sorting).toFixed(2)}`);
        console.log('GPU Render (ms):');
        console.log(`  Avg: ${avg(gpuRender).toFixed(2)}, Min: ${min(gpuRender).toFixed(2)}, Max: ${max(gpuRender).toFixed(2)}, P95: ${p95(gpuRender).toFixed(2)}`);
        console.log('Pre-Render (ms):');
        console.log(`  Avg: ${avg(preRender).toFixed(2)}, Min: ${min(preRender).toFixed(2)}, Max: ${max(preRender).toFixed(2)}, P95: ${p95(preRender).toFixed(2)}`);
        console.log('Post-Render (ms):');
        console.log(`  Avg: ${avg(postRender).toFixed(2)}, Min: ${min(postRender).toFixed(2)}, Max: ${max(postRender).toFixed(2)}, P95: ${p95(postRender).toFixed(2)}`);
        console.log('Total Frame (ms):');
        console.log(`  Avg: ${avg(totalFrame).toFixed(2)}, Min: ${min(totalFrame).toFixed(2)}, Max: ${max(totalFrame).toFixed(2)}, P95: ${p95(totalFrame).toFixed(2)}`);
    }

    exportJSON(): string {
        return JSON.stringify(this.metrics, null, 2);
    }

    downloadJSON(filename: string = `performance-${new Date().toISOString().replace(/:/g, '-')}.json`) {
        const json = this.exportJSON();
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}

export const performanceProfiler = new PerformanceProfiler();

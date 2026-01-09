import { EventHandler } from 'playcanvas';

type FunctionCallback = (...args: any[]) => any;

class Events extends EventHandler {
    functions = new Map<string, FunctionCallback>();

    // declare an editor function
    function(name: string, fn: FunctionCallback) {
        if (this.functions.has(name)) {
            throw new Error(`error: function ${name} already exists`);
        }
        this.functions.set(name, fn);
    }

    // invoke an editor function
    invoke(name: string, ...args: any[]) {
        const fn = this.functions.get(name);
        if (!fn) {
            console.error(`❌ ERROR: function not found '${name}'`);
            console.error('📍 Call stack:', new Error().stack);
            console.error('📋 Available functions:', Array.from(this.functions.keys()));
            return;
        }
        return fn(...args);
    }
}

export { Events };

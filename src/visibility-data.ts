import { GSplatData } from 'playcanvas';

type VisibilityNoneData = {
    mode: 'none';
};

type VisibilityShData = {
    mode: 'sh';
    coeffs: Float32Array[];
};

type VisibilitySvLobe = {
    siteX: Float32Array;
    siteY: Float32Array;
    siteZ: Float32Array;
    value: Float32Array;
    tau: Float32Array;
};

type VisibilitySvData = {
    mode: 'sv';
    numLobes: number;
    lobes: VisibilitySvLobe[];
};

type VisibilityData = VisibilityNoneData | VisibilityShData | VisibilitySvData;

const readVisibilityShData = (splatData: GSplatData): VisibilityShData | null => {
    const coeff0 = splatData.getProp('v_sh_0') as Float32Array | null;
    if (!coeff0) {
        return null;
    }

    const coeffs: Float32Array[] = [];
    for (let i = 0; i < 16; i++) {
        const arr = splatData.getProp(`v_sh_${i}`) as Float32Array | null;
        if (!arr) {
            throw new Error(`Missing visibility SH property v_sh_${i}`);
        }
        coeffs.push(arr);
    }

    return { mode: 'sh', coeffs };
};

const readVisibilitySvData = (splatData: GSplatData): VisibilitySvData | null => {
    const vertexElement = splatData.getElement('vertex');
    const lobeMap = new Map<number, Partial<VisibilitySvLobe>>();

    for (const property of vertexElement.properties as Array<{ name: string }>) {
        const siteMatch = property.name.match(/^v_site_(\d+)_([xyz])$/);
        if (siteMatch) {
            const lobeIndex = parseInt(siteMatch[1], 10);
            const axis = siteMatch[2] as 'x' | 'y' | 'z';
            const storage = splatData.getProp(property.name) as Float32Array | null;
            if (!storage) {
                throw new Error(`Missing SV property storage for ${property.name}`);
            }
            const lobe = lobeMap.get(lobeIndex) ?? {};
            if (axis === 'x') {
                lobe.siteX = storage;
            } else if (axis === 'y') {
                lobe.siteY = storage;
            } else {
                lobe.siteZ = storage;
            }
            lobeMap.set(lobeIndex, lobe);
            continue;
        }

        const valueMatch = property.name.match(/^v_val_(\d+)$/);
        if (valueMatch) {
            const lobeIndex = parseInt(valueMatch[1], 10);
            const storage = splatData.getProp(property.name) as Float32Array | null;
            if (!storage) {
                throw new Error(`Missing SV property storage for ${property.name}`);
            }
            const lobe = lobeMap.get(lobeIndex) ?? {};
            lobe.value = storage;
            lobeMap.set(lobeIndex, lobe);
            continue;
        }

        const tauMatch = property.name.match(/^v_tau_(\d+)$/);
        if (tauMatch) {
            const lobeIndex = parseInt(tauMatch[1], 10);
            const storage = splatData.getProp(property.name) as Float32Array | null;
            if (!storage) {
                throw new Error(`Missing SV property storage for ${property.name}`);
            }
            const lobe = lobeMap.get(lobeIndex) ?? {};
            lobe.tau = storage;
            lobeMap.set(lobeIndex, lobe);
        }
    }

    if (lobeMap.size === 0) {
        return null;
    }

    const lobes = Array.from(lobeMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([lobeIndex, lobe]) => {
        if (!lobe.siteX || !lobe.siteY || !lobe.siteZ || !lobe.value || !lobe.tau) {
            throw new Error(`Incomplete SV visibility lobe ${lobeIndex}`);
        }
        return {
            siteX: lobe.siteX,
            siteY: lobe.siteY,
            siteZ: lobe.siteZ,
            value: lobe.value,
            tau: lobe.tau
        };
    });

    return {
        mode: 'sv',
        numLobes: lobes.length,
        lobes
    };
};

const readVisibilityData = (splatData: GSplatData): VisibilityData => {
    const svData = readVisibilitySvData(splatData);
    if (svData) {
        return svData;
    }

    const shData = readVisibilityShData(splatData);
    if (shData) {
        return shData;
    }

    return { mode: 'none' };
};

export { readVisibilityData };
export type { VisibilityData, VisibilityShData, VisibilitySvData, VisibilitySvLobe };

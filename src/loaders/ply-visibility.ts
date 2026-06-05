import { GSplatData } from 'playcanvas';

type PlyScalarType = 'char' | 'uchar' | 'short' | 'ushort' | 'int' | 'uint' | 'float' | 'double';

type PlyProperty = {
    type: PlyScalarType;
    name: string;
    byteSize: number;
};

const typeAliases = new Map<string, PlyScalarType>([
    ['int8', 'char'],
    ['uint8', 'uchar'],
    ['uchar', 'uchar'],
    ['char', 'char'],
    ['int16', 'short'],
    ['uint16', 'ushort'],
    ['short', 'short'],
    ['ushort', 'ushort'],
    ['int32', 'int'],
    ['uint32', 'uint'],
    ['int', 'int'],
    ['uint', 'uint'],
    ['float32', 'float'],
    ['float', 'float'],
    ['float64', 'double'],
    ['double', 'double']
]);

const typeByteSize: Record<PlyScalarType, number> = {
    char: 1,
    uchar: 1,
    short: 2,
    ushort: 2,
    int: 4,
    uint: 4,
    float: 4,
    double: 8
};

const isVisibilityPropertyName = (name: string) => {
    return name.startsWith('v_sh_') ||
        name.startsWith('v_site_') ||
        name.startsWith('v_val_') ||
        name.startsWith('v_tau_');
};

const readScalar = (view: DataView, offset: number, type: PlyScalarType) => {
    switch (type) {
        case 'char': return view.getInt8(offset);
        case 'uchar': return view.getUint8(offset);
        case 'short': return view.getInt16(offset, true);
        case 'ushort': return view.getUint16(offset, true);
        case 'int': return view.getInt32(offset, true);
        case 'uint': return view.getUint32(offset, true);
        case 'float': return view.getFloat32(offset, true);
        case 'double': return view.getFloat64(offset, true);
    }
};

const findHeaderEndOffset = (headerText: string, endHeaderIndex: number) => {
    let offset = endHeaderIndex + 'end_header'.length;
    if (headerText[offset] === '\r' && headerText[offset + 1] === '\n') {
        offset += 2;
    } else if (headerText[offset] === '\n' || headerText[offset] === '\r') {
        offset += 1;
    }
    return offset;
};

const parseVertexElement = (header: string) => {
    const lines = header.split(/\r?\n/);
    let vertexCount = 0;
    const properties: PlyProperty[] = [];
    let inVertex = false;

    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
            continue;
        }

        const elementMatch = line.match(/^element\s+(\w+)\s+(\d+)/);
        if (elementMatch) {
            inVertex = elementMatch[1] === 'vertex';
            if (inVertex) {
                vertexCount = parseInt(elementMatch[2], 10);
            }
            continue;
        }

        if (!inVertex) {
            continue;
        }

        const propertyMatch = line.match(/^property\s+(\w+)\s+(\w+)$/);
        if (!propertyMatch) {
            continue;
        }

        const type = typeAliases.get(propertyMatch[1].toLowerCase());
        if (!type) {
            continue;
        }

        properties.push({
            type,
            name: propertyMatch[2],
            byteSize: typeByteSize[type]
        });
    }

    return { vertexCount, properties };
};

const plyHasCustomVisibilityProperties = (data: ArrayBuffer): boolean => {
    const headerBytes = new Uint8Array(data, 0, Math.min(1024 * 1024, data.byteLength));
    const headerText = new TextDecoder('ascii').decode(headerBytes);
    const endHeaderIndex = headerText.indexOf('end_header');
    const header = endHeaderIndex === -1 ? headerText : headerText.substring(0, endHeaderIndex);
    const { properties } = parseVertexElement(header);

    if (properties.length === 0) {
        return /^\s*property\s+(float|float32|double|float64)\s+v_(?:sh|site|val|tau)_/mi.test(header);
    }

    return properties.some(prop => (prop.type === 'float' || prop.type === 'double') && isVisibilityPropertyName(prop.name));
};

const parseAndAddCustomVisibilityProperties = (splatData: GSplatData, rawData: ArrayBuffer): number => {
    const headerBytes = new Uint8Array(rawData, 0, Math.min(1024 * 1024, rawData.byteLength));
    const headerText = new TextDecoder('ascii').decode(headerBytes);
    const endHeaderIndex = headerText.indexOf('end_header');

    if (endHeaderIndex === -1) {
        throw new Error('Invalid PLY: missing end_header');
    }

    const header = headerText.substring(0, endHeaderIndex);
    if (!/format\s+binary_little_endian\s+1\.0/i.test(header)) {
        return 0;
    }

    const headerEndOffset = findHeaderEndOffset(headerText, endHeaderIndex);
    const { vertexCount, properties } = parseVertexElement(header);
    if (!vertexCount || properties.length === 0) {
        return 0;
    }

    const customProps = properties.filter(prop => {
        return (prop.type === 'float' || prop.type === 'double') && isVisibilityPropertyName(prop.name);
    });
    if (customProps.length === 0) {
        return 0;
    }

    const bytesPerVertex = properties.reduce((sum, prop) => sum + prop.byteSize, 0);
    const requiredBytes = headerEndOffset + vertexCount * bytesPerVertex;
    if (requiredBytes > rawData.byteLength) {
        throw new Error('Invalid PLY: vertex data is shorter than declared header');
    }

    const propData = new Map<string, Float32Array>();
    for (const prop of customProps) {
        propData.set(prop.name, new Float32Array(vertexCount));
    }

    const dataView = new DataView(rawData, headerEndOffset);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
        let offset = vertex * bytesPerVertex;

        for (const prop of properties) {
            const storage = propData.get(prop.name);
            if (storage) {
                storage[vertex] = readScalar(dataView, offset, prop.type);
            }
            offset += prop.byteSize;
        }
    }

    const vertexElement = splatData.getElement('vertex');
    let added = 0;
    for (const [propName, storage] of propData.entries()) {
        if (!splatData.getProp(propName)) {
            vertexElement.properties.push({
                type: 'float',
                name: propName,
                storage,
                byteSize: 4
            });
            added++;
        }
    }

    return added;
};

export { parseAndAddCustomVisibilityProperties, plyHasCustomVisibilityProperties };

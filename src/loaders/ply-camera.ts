import { DefaultCameraPose } from '../camera-default';

type CameraProperty = {
    name: string;
    type: 'float' | 'uint';
    byteSize: number;
};

// Binary layout of the camera element in PLY
const cameraProperties: CameraProperty[] = [
    { name: 'px', type: 'float', byteSize: 4 },
    { name: 'py', type: 'float', byteSize: 4 },
    { name: 'pz', type: 'float', byteSize: 4 },
    { name: 'r00', type: 'float', byteSize: 4 },
    { name: 'r01', type: 'float', byteSize: 4 },
    { name: 'r02', type: 'float', byteSize: 4 },
    { name: 'r10', type: 'float', byteSize: 4 },
    { name: 'r11', type: 'float', byteSize: 4 },
    { name: 'r12', type: 'float', byteSize: 4 },
    { name: 'r20', type: 'float', byteSize: 4 },
    { name: 'r21', type: 'float', byteSize: 4 },
    { name: 'r22', type: 'float', byteSize: 4 },
    { name: 'fx', type: 'float', byteSize: 4 },
    { name: 'fy', type: 'float', byteSize: 4 },
    { name: 'width', type: 'uint', byteSize: 4 },
    { name: 'height', type: 'uint', byteSize: 4 }
];

const CAMERA_BYTE_SIZE = cameraProperties.reduce((sum, p) => sum + p.byteSize, 0);

const CAMERA_HEADER_LINES = [
    'element camera 1',
    'property float px',
    'property float py',
    'property float pz',
    'property float r00',
    'property float r01',
    'property float r02',
    'property float r10',
    'property float r11',
    'property float r12',
    'property float r20',
    'property float r21',
    'property float r22',
    'property float fx',
    'property float fy',
    'property uint width',
    'property uint height'
];

// Parse a PLY buffer that was written with CAMERA_HEADER_LINES and extract camera pose.
// Returns null if the PLY does not contain a compatible camera element.
const parsePlyCamera = (rawData: ArrayBuffer): DefaultCameraPose | null => {
    if (!rawData || rawData.byteLength === 0) {
        return null;
    }

    const headerBytes = new Uint8Array(rawData, 0, Math.min(65536, rawData.byteLength));
    const headerText = new TextDecoder('ascii').decode(headerBytes);

    const endHeaderIndex = headerText.indexOf('end_header');
    if (endHeaderIndex === -1) {
        return null;
    }

    const header = headerText.substring(0, endHeaderIndex);
    const headerEndOffset = endHeaderIndex + 'end_header'.length + 1;

    const lines = header.split('\n').map((l) => l.replace('\r', ''));

    let vertexCount = 0;
    let bytesPerVertex = 0;
    let foundVertex = false;
    let foundCamera = false;

    let i = 0;
    while (i < lines.length) {
        const line = lines[i].trim();

        if (line.startsWith('element vertex')) {
            const parts = line.split(/\s+/);
            vertexCount = parseInt(parts[2], 10);
            foundVertex = Number.isFinite(vertexCount) && vertexCount > 0;

            // collect vertex properties to compute bytesPerVertex
            i++;
            bytesPerVertex = 0;
            while (i < lines.length) {
                const propLine = lines[i].trim();
                if (!propLine.startsWith('property')) {
                    break;
                }
                const tokens = propLine.split(/\s+/);
                const type = tokens[1];
                if (type === 'uchar') {
                    bytesPerVertex += 1;
                } else if (type === 'float' || type === 'double' || type === 'int' || type === 'uint' || type === 'short' || type === 'ushort' || type === 'char') {
                    // For our viewer PLY we only expect float / uchar, but be conservative.
                    bytesPerVertex += (type === 'double') ? 8 : 4;
                } else {
                    // Unknown type, bail out
                    return null;
                }
                i++;
            }
            continue;
        }

        if (line.startsWith('element camera')) {
            foundCamera = true;

            // Verify that camera properties match CAMERA_HEADER_LINES
            const expectedProps = CAMERA_HEADER_LINES.slice(1);
            for (let p = 0; p < expectedProps.length; p++) {
                const idx = i + 1 + p;
                if (idx >= lines.length) {
                    return null;
                }
                const propLine = lines[idx].trim();
                if (propLine !== expectedProps[p]) {
                    return null;
                }
            }

            break;
        }

        i++;
    }

    if (!foundVertex || !foundCamera || bytesPerVertex <= 0 || vertexCount <= 0) {
        return null;
    }

    const vertexDataSize = vertexCount * bytesPerVertex;
    const cameraOffset = headerEndOffset + vertexDataSize;
    if (cameraOffset + CAMERA_BYTE_SIZE > rawData.byteLength) {
        return null;
    }

    const dv = new DataView(rawData, cameraOffset, CAMERA_BYTE_SIZE);
    let offset = 0;

    const readFloat = () => {
        const v = dv.getFloat32(offset, true);
        offset += 4;
        return v;
    };

    const readUint = () => {
        const v = dv.getUint32(offset, true);
        offset += 4;
        return v;
    };

    const values: Record<string, number> = {};
    for (const prop of cameraProperties) {
        values[prop.name] = (prop.type === 'float') ? readFloat() : readUint();
    }

    const pose: DefaultCameraPose = {
        position: [values.px, values.py, values.pz],
        rotation: [
            [values.r00, values.r01, values.r02],
            [values.r10, values.r11, values.r12],
            [values.r20, values.r21, values.r22]
        ],
        fx: values.fx,
        fy: values.fy,
        width: values.width,
        height: values.height
    };

    return pose;
};

// Serialize a DefaultCameraPose into a binary blob matching the camera element layout.
const writeCameraBinary = (pose: DefaultCameraPose): Uint8Array => {
    const buf = new Uint8Array(CAMERA_BYTE_SIZE);
    const dv = new DataView(buf.buffer);
    let offset = 0;

    const writeFloat = (v: number) => {
        dv.setFloat32(offset, v, true);
        offset += 4;
    };

    const writeUint = (v: number) => {
        dv.setUint32(offset, v >>> 0, true);
        offset += 4;
    };

    const px = pose.position[0];
    const py = pose.position[1];
    const pz = pose.position[2];

    writeFloat(px);
    writeFloat(py);
    writeFloat(pz);

    writeFloat(pose.rotation[0][0]);
    writeFloat(pose.rotation[0][1]);
    writeFloat(pose.rotation[0][2]);
    writeFloat(pose.rotation[1][0]);
    writeFloat(pose.rotation[1][1]);
    writeFloat(pose.rotation[1][2]);
    writeFloat(pose.rotation[2][0]);
    writeFloat(pose.rotation[2][1]);
    writeFloat(pose.rotation[2][2]);

    writeFloat(pose.fx ?? 0);
    writeFloat(pose.fy ?? 0);

    writeUint(pose.width ?? 0);
    writeUint(pose.height ?? 0);

    return buf;
};

export {
    CAMERA_BYTE_SIZE,
    CAMERA_HEADER_LINES,
    parsePlyCamera,
    writeCameraBinary
};


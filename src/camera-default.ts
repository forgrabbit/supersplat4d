import { Mat4, Quat, Vec3 } from 'playcanvas';

import { Events } from './events';

type DefaultCameraPose = {
    position: [number, number, number];
    rotation: [
        [number, number, number],
        [number, number, number],
        [number, number, number]
    ];
    fx?: number;
    fy?: number;
    width?: number;
    height?: number;
};

type SibrCameraPose = {
    worldPosition: Vec3;
    worldRotation: Quat;
    fovYDeg: number;
    aspect: number;
};

const EXTRA_ROTATION_DEGREES = 180;

const createExtraRotationQuat = () => {
    const q = new Quat();
    q.setFromEulerAngles(0, 0, EXTRA_ROTATION_DEGREES);
    return q;
};

const fromDefaultCameraPose = (pose: DefaultCameraPose, applyDefaultOrientation: boolean): SibrCameraPose => {
    const { position, rotation, fy, width, height } = pose;

    const p = new Vec3(position[0], position[1], position[2]);

    // SIBR-style column flip
    const R = rotation;
    const orientation = [
        [R[0][0], -R[0][1], -R[0][2]],
        [R[1][0], -R[1][1], -R[1][2]],
        [R[2][0], -R[2][1], -R[2][2]]
    ];

    const m = new Mat4();
    m.set([
        orientation[0][0], orientation[0][1], orientation[0][2], 0,
        orientation[1][0], orientation[1][1], orientation[1][2], 0,
        orientation[2][0], orientation[2][1], orientation[2][2], 0,
        0, 0, 0, 1
    ]);

    const worldRotation = new Quat();
    worldRotation.setFromMat4(m);

    const worldPosition = p.clone();

    if (applyDefaultOrientation) {
        const extra = createExtraRotationQuat();
        extra.transformVector(worldPosition, worldPosition);
        worldRotation.mul2(extra, worldRotation);
    }

    let fovYDeg = 60;
    if (typeof fy === 'number' && typeof height === 'number' && fy > 0 && height > 0) {
        fovYDeg = 2 * Math.atan(0.5 * height / fy) * 180 / Math.PI;
    }

    let aspect = 1;
    if (typeof width === 'number' && typeof height === 'number' && width > 0 && height > 0) {
        aspect = width / height;
    }

    return {
        worldPosition,
        worldRotation,
        fovYDeg,
        aspect
    };
};

/**
 * Apply a default camera pose (in training space) to the viewer camera.
 * The pose uses the same convention as cameras.json / PLY camera:
 * - position: camera center in training coordinates
 * - rotation: 3x3 rotation matrix (world-to-camera)
 * This function performs the same SIBR-style column flip as loadCameraPoses
 * and then delegates to camera.setFromPoseMatrix for final application.
 */
const applyDefaultCameraPose = (events: Events, pose: DefaultCameraPose) => {
    if (!pose) {
        return;
    }

    const { position, rotation, fx, fy, width, height } = pose;

    if (!Array.isArray(position) || position.length !== 3) {
        throw new Error('Camera JSON must contain position[3].');
    }
    if (!Array.isArray(rotation) || rotation.length !== 3 ||
        !rotation.every(r => Array.isArray(r) && r.length === 3)) {
        throw new Error('Camera JSON must contain rotation[3][3].');
    }

    const sibrExact = !!events.invoke('camera.isSibrExactMode');

    if (sibrExact) {
        const poseSibr = fromDefaultCameraPose(pose, true);
        events.fire('camera.setFromSibrPose', poseSibr);
    } else {
        const p = new Vec3(position[0], position[1], position[2]);

        const R = rotation;
        const orientation = [
            [R[0][0], -R[0][1], -R[0][2]],
            [R[1][0], -R[1][1], -R[1][2]],
            [R[2][0], -R[2][1], -R[2][2]]
        ];

        events.fire('camera.setFromPoseMatrix', {
            position: p,
            rotation: orientation,
            fx,
            fy,
            width,
            height
        });
    }
};

/**
 * Build a single-element camera JSON array from a DefaultCameraPose.
 * This is used only for UI presentation and editing.
 */
const buildCameraJsonFromPose = (pose: DefaultCameraPose, splatName?: string) => {
    const entry: any = {
        id: 0,
        img_name: splatName ?? 'cam00',
        width: pose.width ?? 0,
        height: pose.height ?? 0,
        position: [
            pose.position[0],
            pose.position[1],
            pose.position[2]
        ],
        rotation: [
            [...pose.rotation[0]],
            [...pose.rotation[1]],
            [...pose.rotation[2]]
        ],
        fy: pose.fy ?? 0,
        fx: pose.fx ?? 0
    };

    return [entry];
};

/**
 * Parse and validate a camera JSON string coming from the UI.
 * The expected format is an array with exactly one entry.
 */
const parseCameraJson = (jsonText: string): DefaultCameraPose => {
    let parsed: any;
    try {
        parsed = JSON.parse(jsonText);
    } catch (error) {
        throw new Error('Camera JSON is not valid JSON.');
    }

    if (!Array.isArray(parsed) || parsed.length !== 1) {
        throw new Error('Camera JSON must be an array with exactly one entry.');
    }

    const entry = parsed[0];
    if (!entry || typeof entry !== 'object') {
        throw new Error('Camera JSON entry must be an object.');
    }

    const position = entry.position;
    const rotation = entry.rotation;

    if (!Array.isArray(position) || position.length !== 3 ||
        !position.every((v: any) => typeof v === 'number' && Number.isFinite(v))) {
        throw new Error('Camera JSON position must be an array of 3 finite numbers.');
    }

    if (!Array.isArray(rotation) || rotation.length !== 3 ||
        !rotation.every((row: any) =>
            Array.isArray(row) &&
            row.length === 3 &&
            row.every((v: any) => typeof v === 'number' && Number.isFinite(v)))) {
        throw new Error('Camera JSON rotation must be a 3x3 array of finite numbers.');
    }

    const fx = entry.fx;
    const fy = entry.fy;
    const width = entry.width;
    const height = entry.height;

    const pose: DefaultCameraPose = {
        position: [position[0], position[1], position[2]],
        rotation: [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]]
        ]
    };

    if (typeof fx === 'number' && Number.isFinite(fx)) {
        pose.fx = fx;
    }
    if (typeof fy === 'number' && Number.isFinite(fy)) {
        pose.fy = fy;
    }
    if (typeof width === 'number' && Number.isFinite(width)) {
        pose.width = width;
    }
    if (typeof height === 'number' && Number.isFinite(height)) {
        pose.height = height;
    }

    return pose;
};

export {
    DefaultCameraPose,
    SibrCameraPose,
    fromDefaultCameraPose,
    applyDefaultCameraPose,
    buildCameraJsonFromPose,
    parseCameraJson
};


const vertexShader = /* glsl*/`
#include "gsplatCommonVS"

uniform sampler2D splatState;

uniform vec4 selectedClr;
uniform vec4 lockedClr;

uniform vec3 clrOffset;
uniform vec4 clrScale;

varying mediump vec3 texCoordIsLocked;          // store locked flat in z
varying mediump vec4 color;

#if PICK_PASS
    uniform uint pickMode;                      // 0: add, 1: remove, 2: set
    uniform bool uFrameOnlyMode;                // If true, only pick visible splats at current frame
#endif

mediump vec4 discardVec = vec4(0.0, 0.0, 2.0, 1.0);

uniform float saturation;

vec3 applySaturation(vec3 color) {
    vec3 grey = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    return grey + (color - grey) * saturation;
}

#ifdef HAS_VISIBILITY
uniform vec3 uCameraPosition; // Camera position in the same space as modelCenter

float sigmoid(float v) {
    if (v >= 0.0) {
        return 1.0 / (1.0 + exp(-v));
    }
    float t = exp(v);
    return t / (1.0 + t);
}

float evalVisibilitySHDeg3(vec3 d) {
    vec4 sh0 = loadSplatVisibilitySH0();
    vec4 sh1 = loadSplatVisibilitySH1();
    vec4 sh2 = loadSplatVisibilitySH2();
    vec4 sh3 = loadSplatVisibilitySH3();

    float x = d.x;
    float y = d.y;
    float z = d.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    const float C0 = 0.28209479177387814;
    const float C1 = 0.4886025119029199;
    const float C2_0 = 1.0925484305920792;
    const float C2_1 = -1.0925484305920792;
    const float C2_2 = 0.31539156525252005;
    const float C2_3 = -1.0925484305920792;
    const float C2_4 = 0.5462742152960396;
    const float C3_0 = -0.5900435899266435;
    const float C3_1 = 2.890611442640554;
    const float C3_2 = -0.4570457994644658;
    const float C3_3 = 0.3731763325901154;
    const float C3_4 = -0.4570457994644658;
    const float C3_5 = 1.445305721320277;
    const float C3_6 = -0.5900435899266435;

    float r = 0.0;

    // Degree 0
    r += C0 * sh0.x;

    // Degree 1
    r += (-C1 * y) * sh0.y;
    r += ( C1 * z) * sh0.z;
    r += (-C1 * x) * sh0.w;

    // Degree 2
    r += C2_0 * (x * y) * sh1.x;
    r += C2_1 * (y * z) * sh1.y;
    r += C2_2 * (2.0 * zz - xx - yy) * sh1.z;
    r += C2_3 * (x * z) * sh1.w;
    r += C2_4 * (xx - yy) * sh2.x;

    // Degree 3
    r += C3_0 * y * (3.0 * xx - yy) * sh2.y;
    r += C3_1 * (x * y * z) * sh2.z;
    r += C3_2 * y * (4.0 * zz - xx - yy) * sh2.w;
    r += C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh3.x;
    r += C3_4 * x * (4.0 * zz - xx - yy) * sh3.y;
    r += C3_5 * z * (xx - yy) * sh3.z;
    r += C3_6 * x * (xx - 3.0 * yy) * sh3.w;

    return r;
}
#endif

void main(void) {
    // read gaussian details
    SplatSource source;
    if (!initSource(source)) {
        gl_Position = discardVec;
        return;
    }

    // get per-gaussian edit state, discard if deleted
    uint vertexState = uint(texelFetch(splatState, splat.uv, 0).r * 255.0 + 0.5) & 7u;

    #if OUTLINE_PASS
        if (vertexState != 1u) {
            gl_Position = discardVec;
            return;
        }
    #elif UNDERLAY_PASS
        if (vertexState != 1u) {
            gl_Position = discardVec;
            return;
        }
    #elif PICK_PASS
        if (pickMode == 0u) {
            // add: skip deleted, locked and selected splats
            if (vertexState != 0u) {
                gl_Position = discardVec;
                return;
            }
        } else if (pickMode == 1u) {
            // remove: skip deleted, locked and unselected splats
            if (vertexState != 1u) {
                gl_Position = discardVec;
                return;
            }
        } else {
            // set: skip deleted and locked splats
            if ((vertexState & 6u) != 0u) {
                gl_Position = discardVec;
                return;
            }
        }
        
        // Filter by visibility at current frame if frame-only mode is enabled
        // Note: We use a simplified check here (gaussian weight only) for early discard.
        // Full opacity check (including base opacity) is done in CPU-side selection logic.
        #ifdef DYNAMIC_MODE
        if (uFrameOnlyMode && uIsDynamic > 0.5) {
            ivec2 uv = splat.uv;
            vec2 trbfData = texelFetch(splatTrbf, uv, 0).rg;
            float trbfCenter = trbfData.r;
            float trbfScale = trbfData.g;
            
            // Calculate gaussian weight (temporal kernel)
            // We can't read opacity here safely (readColor may need center), so we use
            // a conservative threshold on gaussian weight alone.
            // This is a performance optimization - full check happens in CPU selection logic.
            float dt = (uCurrentTime - trbfCenter) / max(trbfScale, 1e-6);
            float gaussian = exp(-dt * dt);
            
            // Use selection threshold (0.05) for picker - conservative since we only check gaussian weight
            // Full opacity check (baseOp * gaussian > 0.05) is done in CPU-side selection logic
            if (gaussian < 0.05) {
                gl_Position = discardVec;
                return;
            }
        }
        #endif
    #else
        if ((vertexState & 4u) != 0u) {
            gl_Position = discardVec;
            return;
        }
    #endif

    // get center (getCenter() must be called before getColor() and loads shared data)
    vec3 modelCenter = getCenter();

    SplatCenter center;
    if (!initCenter(modelCenter, center)) {
        gl_Position = discardVec;
        return;
    }

    SplatCorner corner;
    if (!initCorner(source, center, corner)) {
        gl_Position = discardVec;
        return;
    }

    gl_Position = center.proj + vec4(corner.offset.xyz, 0.0);

    // store texture coord and locked state
    texCoordIsLocked = vec3(corner.uv, (vertexState & 2u) != 0u ? 1.0 : 0.0);

    #if UNDERLAY_PASS
        color = getColor();
        color.xyz = mix(color.xyz, selectedClr.xyz * 0.2, selectedClr.a) * selectedClr.a;
    #elif PICK_PASS
        uvec4 bits = (uvec4(splat.index) >> uvec4(0u, 8u, 16u, 24u)) & uvec4(255u);
        color = vec4(bits) / 255.0;
    // handle splat color
    #elif FORWARD_PASS
        // read color
        color = getColor();

        // Apply dynamic opacity for dynamic gaussians
        #ifdef DYNAMIC_MODE
        if (uIsDynamic > 0.5) {
            ivec2 uv = splat.uv;
            vec2 trbfData = texelFetch(splatTrbf, uv, 0).rg;
            float trbfCenter = trbfData.r;
            float trbfScale = trbfData.g;
            
            // Calculate time offset and gaussian weight
            // Match SIBR viewer CUDA: exp(-dt_scaled^2)
            float dt = (uCurrentTime - trbfCenter) / max(trbfScale, 1e-6);
            float gaussian = exp(-dt * dt);
            
            // Apply gaussian weight to opacity
            color.a *= gaussian;
            
            // Discard splats with low opacity (improves performance and visual quality)
            if (color.a < 0.005) {
                gl_Position = discardVec;
                return;
            }
        }
        #endif

        #ifdef HAS_VISIBILITY
            #ifdef FROZEN_OPACITY
                color.a = loadSplatFrozenOpacity().r;
            #else
                vec3 centerForVis = uIsDynamic > 0.5 ? computeDynamicPosition(modelCenter) : modelCenter;
                vec3 D_view = normalize(centerForVis - uCameraPosition);
                float visRaw = evalVisibilitySHDeg3(D_view);
                float visibility = sigmoid(visRaw);
                color.a *= visibility;
            #endif

            if (color.a < 0.005) {
                gl_Position = discardVec;
                return;
            }
        #endif

        // evaluate spherical harmonics
        #if SH_BANDS > 0
        // calculate the model-space view direction
            vec3 dir = normalize(center.view * mat3(center.modelView));

            // read sh coefficients
            vec3 sh[SH_COEFFS];
            float scale;
            readSHData(sh, scale);

            // evaluate
            color.xyz += evalSH(sh, dir) * scale;
        #endif

        // apply tint/brightness
        color = color * clrScale + vec4(clrOffset, 0.0);

        // apply saturation
        color.xyz = applySaturation(color.xyz);

        // don't allow out-of-range alpha
        color.a = clamp(color.a, 0.0, 1.0);

        // apply tonemapping
        color = vec4(prepareOutputFromGamma(max(color.xyz, 0.0)), color.w);

        // apply locked/selected colors
        if ((vertexState & 2u) != 0u) {
            // locked
            color *= lockedClr;
        } else if ((vertexState & 1u) != 0u) {
            // selected
            color.xyz = mix(color.xyz, selectedClr.xyz * 0.8, selectedClr.a);
        }
    #endif
}
`;

const fragmentShader = /* glsl*/`
varying mediump vec3 texCoordIsLocked;
varying mediump vec4 color;

uniform int mode;               // 0: centers, 1: rings
uniform float pickerAlpha;
uniform float ringSize;

const float EXP4 = exp(-4.0);
const float INV_EXP4 = 1.0 / (1.0 - EXP4);

float normExp(float x) {
    return (exp(x * -4.0) - EXP4) * INV_EXP4;
}

void main(void) {
    mediump float A = dot(texCoordIsLocked.xy, texCoordIsLocked.xy);

    if (A > 1.0) {
        discard;
    }

    #if OUTLINE_PASS
        gl_FragColor = vec4(1.0, 1.0, 1.0, mode == 0 ? exp(-A * 4.0) * color.a : 1.0);
    #else
        #ifdef PICK_PASS
            gl_FragColor = color;
        #else
            mediump float alpha = normExp(A) * color.a;

            if (texCoordIsLocked.z == 0.0 && ringSize > 0.0) {
                // rings mode
                if (A < 1.0 - ringSize) {
                    alpha = max(0.05, alpha);
                } else {
                    alpha = 0.6;
                }
            }

            gl_FragColor = vec4(color.xyz * alpha, alpha);
        #endif
    #endif
}
`;

const gsplatCenter = /* glsl*/`
uniform mat4 matrix_model;
uniform mat4 matrix_view;
uniform mat4 matrix_projection;
uniform vec4 camera_params;                     // 1/far, far, near, isOrtho (required by gsplatCornerVS)

uniform usampler2D splatTransform;              // per-splat index into transform palette
uniform sampler2D transformPalette;             // palette of transform matrices

uniform float uCurrentTime;                     // current absolute time for dynamic gaussians
uniform float uIsDynamic;                       // 1.0 = dynamic, 0.0 = static (bool as float for WGSL compat)
#ifdef DYNAMIC_MODE
uniform sampler2D splatMotion;                 // For dynamic: motion_0, motion_1, motion_2 (RGB)
uniform sampler2D splatTrbf;                    // For dynamic: trbf_center, trbf_scale (RG)
#endif

mat4 applyPaletteTransform(mat4 model) {
    uint transformIndex = texelFetch(splatTransform, splat.uv, 0).r;
    if (transformIndex == 0u) {
        return model;
    }

    // read transform matrix
    int u = int(transformIndex % 512u) * 3;
    int v = int(transformIndex / 512u);

    mat4 t;
    t[0] = texelFetch(transformPalette, ivec2(u, v), 0);
    t[1] = texelFetch(transformPalette, ivec2(u + 1, v), 0);
    t[2] = texelFetch(transformPalette, ivec2(u + 2, v), 0);
    t[3] = vec4(0.0, 0.0, 0.0, 1.0);

    return model * transpose(t);
}

// Compute dynamic position for a gaussian
// pos(t) = base_pos + motion * (t - trbf_center)  [Linear motion, as in SIBR]
vec3 computeDynamicPosition(vec3 basePos) {
    #ifndef DYNAMIC_MODE
    return basePos;
    #else
    if (uIsDynamic < 0.5) {
        return basePos;
    }
    
    // Read motion and trbf from textures (use global splat.uv)
    ivec2 uv = splat.uv;
    vec4 motionData = texelFetch(splatMotion, uv, 0);
    vec2 trbfData = texelFetch(splatTrbf, uv, 0).rg;
    
    vec3 motion = motionData.rgb;
    float trbfCenter = trbfData.r;
    
    // Calculate time offset (dt = t - trbf_center)
    float dt = uCurrentTime - trbfCenter;
    
    // Apply linear motion (no gaussian weight for position!)
    return basePos + motion * dt;
    #endif
}

// project the model space gaussian center to view and clip space
bool initCenter(vec3 modelCenter, inout SplatCenter center) {
    // Apply dynamic position transformation if this is a dynamic gaussian
    vec3 dynamicCenter = uIsDynamic > 0.5 ? computeDynamicPosition(modelCenter) : modelCenter;
    
    mat4 modelView = matrix_view * applyPaletteTransform(matrix_model);
    vec4 centerView = modelView * vec4(dynamicCenter, 1.0);

    // early out if splat is behind the camera
    if (centerView.z > 0.0) {
        return false;
    }

    vec4 centerProj = matrix_projection * centerView;

    // ensure gaussians are not clipped by camera near and far
    centerProj.z = clamp(centerProj.z, -abs(centerProj.w), abs(centerProj.w));

    center.view = centerView.xyz / centerView.w;
    center.proj = centerProj;
    center.projMat00 = matrix_projection[0][0];
    center.modelView = modelView;
    return true;
}
`;

/**
 * WGSL override for gsplatModifyVS — injected into the engine's gsplatVS via shaderChunks.
 *
 * The engine's gsplatVS (WGSL) calls three hooks we override here:
 *   modifySplatCenter()         — model-space position offset (called before initCenter)
 *   modifySplatRotationScale()  — rotation/scale tweak (stub, unused)
 *   modifySplatColor()          — alpha/color tweak (called after getColor + SH)
 *
 * On WebGL the full custom vertexShader GLSL chunk handles everything.
 * On WebGPU only this WGSL chunk is active, so ALL per-splat effects must live here.
 *
 * TRBF (Temporal Radial Basis Function) model:
 *   position: p(t) = p0 + motion * (t - trbf_center)          (linear)
 *   alpha:    a(t) = base_alpha * exp(-((t - trbf_center) / trbf_scale)^2)
 *
 * Flow in engine gsplatVS (WGSL):
 *   getCenter()              → reads model-space position p0 from engine texture
 *   modifySplatCenter()      ← applies linear motion offset in model space
 *   initCenter()             → engine applies matrix_model * matrix_view * projection
 *   getColor() + evalSH()    → base colour + spherical harmonics
 *   modifySplatColor()       ← applies TRBF gaussian kernel to alpha
 *   clipCorner() + output    → engine discards if alpha too small
 */
const gsplatModifyWGSL = /* wgsl */`
#ifdef DYNAMIC_MODE
uniform uCurrentTime: f32;
// texture_2d<uff> uses sampleType:'unfilterable-float' in the bind-group layout,
// which is correct for rgba32float on ALL WebGPU devices (including those that lack
// the optional float32-filterable feature).  textureLoad() still returns vec4<f32>.
var splatMotion: texture_2d<uff>;
var splatTrbf: texture_2d<uff>;
#endif

#ifdef HAS_VISIBILITY
uniform uCameraPosition: vec3f;
var splatVisibilitySH0: texture_2d<uff>;
var splatVisibilitySH1: texture_2d<uff>;
var splatVisibilitySH2: texture_2d<uff>;
var splatVisibilitySH3: texture_2d<uff>;
#ifdef FROZEN_OPACITY
var splatFrozenOpacity: texture_2d<uff>;
#endif

fn visSigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Same layout as GLSL evalVisibilitySHDeg3 / compute cull SH (deg 3).
fn evalVisibilitySHDeg3WGSL(d: vec3f, sh0: vec4f, sh1: vec4f, sh2: vec4f, sh3: vec4f) -> f32 {
    let x = d.x;
    let y = d.y;
    let z = d.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;
    let xx_yy = xx - yy;
    var r = 0.28209479177387814 * sh0.x;
    r += -0.4886025119029199 * y * sh0.y;
    r += 0.4886025119029199 * z * sh0.z;
    r += -0.4886025119029199 * x * sh0.w;
    r += 1.0925484305920792 * xy * sh1.x;
    r += -1.0925484305920792 * yz * sh1.y;
    r += 0.31539156525252005 * (2.0 * zz - xx - yy) * sh1.z;
    r += -1.0925484305920792 * xz * sh1.w;
    r += 0.5462742152960396 * xx_yy * sh2.x;
    r += -0.5900435899266435 * y * (3.0 * xx - yy) * sh2.y;
    r += 2.890611442640554 * x * y * z * sh2.z;
    r += -0.4570457994644658 * y * (4.0 * zz - xx - yy) * sh2.w;
    r += 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh3.x;
    r += -0.4570457994644658 * x * (4.0 * zz - xx - yy) * sh3.y;
    r += 1.445305721320277 * z * xx_yy * sh3.z;
    r += -0.5900435899266435 * x * (xx - 3.0 * yy) * sh3.w;
    return r;
}
#endif

// Linear motion model: pos(t) = p0 + motion * (t - trbf_center)
fn modifySplatCenter(center: ptr<function, vec3f>) {
    #ifdef DYNAMIC_MODE
    let motionData: vec4f = textureLoad(splatMotion, splat.uv, 0);
    let trbfData:   vec4f = textureLoad(splatTrbf,   splat.uv, 0);
    let dt: f32 = uniform.uCurrentTime - trbfData.r;
    *center += motionData.rgb * dt;
    #endif
}

// Stub: rotation/scale are not modified for 4DGS
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
}

// Apply TRBF temporal gaussian kernel to alpha.
// Matches the WebGL path in vertexShader GLSL (FORWARD_PASS block).
// Without this, off-peak splats render at full opacity → over-bright glow.
// NOTE: No runtime uIsDynamic check needed here – DYNAMIC_MODE is the compile-time
// gate and is only set when isDynamic=true.  Removing the runtime check eliminates
// one failure mode (the float uniform not being written to the mesh UB in time).
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    #ifdef DYNAMIC_MODE
    let trbfData: vec4f  = textureLoad(splatTrbf, splat.uv, 0);
    let trbfCenter: f32  = trbfData.r;
    let trbfScale: f32   = max(trbfData.g, 1e-6);
    let dt: f32          = (uniform.uCurrentTime - trbfCenter) / trbfScale;
    let gaussian: f32    = exp(-dt * dt);
    (*color).a = (*color).a * gaussian;
    #endif

    #ifdef HAS_VISIBILITY
        #ifdef FROZEN_OPACITY
        (*color).a = textureLoad(splatFrozenOpacity, splat.uv, 0).r;
        #else
        let sh0v = textureLoad(splatVisibilitySH0, splat.uv, 0);
        let sh1v = textureLoad(splatVisibilitySH1, splat.uv, 0);
        let sh2v = textureLoad(splatVisibilitySH2, splat.uv, 0);
        let sh3v = textureLoad(splatVisibilitySH3, splat.uv, 0);
        let D_view = normalize(center - uniform.uCameraPosition);
        let visRaw = evalVisibilitySHDeg3WGSL(D_view, sh0v, sh1v, sh2v, sh3v);
        (*color).a = (*color).a * visSigmoid(visRaw);
        #endif
    #endif
}
`;

export { vertexShader, fragmentShader, gsplatCenter, gsplatModifyWGSL };

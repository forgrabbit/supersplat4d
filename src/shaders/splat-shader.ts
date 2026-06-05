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
uniform float uVisibilityCullThreshold; // effective alpha cull threshold from cfg_args culling

vec3 applySaturation(vec3 color) {
    vec3 grey = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    return grey + (color - grey) * saturation;
}

#ifdef HAS_VISIBILITY
uniform vec3 uCameraPosition; // Camera position in the same space as modelCenter
uniform sampler2D splatFrozenOpacity;

    #ifdef HAS_VISIBILITY_SH
uniform sampler2D splatVisibilitySH0;
uniform sampler2D splatVisibilitySH1;
uniform sampler2D splatVisibilitySH2;
uniform sampler2D splatVisibilitySH3;
    #endif

    #ifdef HAS_VISIBILITY_SV
uniform sampler2D splatVisibilitySVSiteValue;
uniform sampler2D splatVisibilitySVTau;
uniform float uVisibilitySVLobeStride;
uniform float uVisibilitySVPackAxis;
    #endif

float sigmoid(float v) {
    if (v >= 0.0) {
        return 1.0 / (1.0 + exp(-v));
    }
    float t = exp(v);
    return t / (1.0 + t);
}

float softplus(float v) {
    return log(1.0 + exp(-abs(v))) + max(v, 0.0);
}

vec3 safeNormalizeVec3(vec3 v) {
    return v * inversesqrt(max(dot(v, v), 1e-12));
}

#ifdef HAS_VISIBILITY_SH
float evalVisibilitySHDeg3(vec3 d) {
    vec4 sh0 = texelFetch(splatVisibilitySH0, splat.uv, 0);
    vec4 sh1 = texelFetch(splatVisibilitySH1, splat.uv, 0);
    vec4 sh2 = texelFetch(splatVisibilitySH2, splat.uv, 0);
    vec4 sh3 = texelFetch(splatVisibilitySH3, splat.uv, 0);

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

#ifdef HAS_VISIBILITY_SV
ivec2 visibilitySVUv(int lobe) {
    int stride = int(uVisibilitySVLobeStride + 0.5);
    if (uVisibilitySVPackAxis < 0.5) {
        return ivec2(splat.uv.x, splat.uv.y + lobe * stride);
    }

    return ivec2(splat.uv.x + lobe * stride, splat.uv.y);
}

float evalVisibilitySVDeg3(vec3 d) {
    vec3 dir = safeNormalizeVec3(d);
    float logits[VISIBILITY_SV_LOBES];
    float values[VISIBILITY_SV_LOBES];
    float maxLogit = -1.0e30;

    for (int lobe = 0; lobe < VISIBILITY_SV_LOBES; ++lobe) {
        ivec2 uv = visibilitySVUv(lobe);
        vec4 siteValue = texelFetch(splatVisibilitySVSiteValue, uv, 0);
        float tauRaw = texelFetch(splatVisibilitySVTau, uv, 0).r;
        vec3 site = safeNormalizeVec3(siteValue.xyz);
        float logit = -softplus(tauRaw) * length(site - dir);
        logits[lobe] = logit;
        values[lobe] = siteValue.w;
        maxLogit = max(maxLogit, logit);
    }

    float weightedValue = 0.0;
    float totalWeight = 0.0;
    for (int lobe = 0; lobe < VISIBILITY_SV_LOBES; ++lobe) {
        float weight = exp(logits[lobe] - maxLogit);
        weightedValue += weight * values[lobe];
        totalWeight += weight;
    }

    return weightedValue / max(totalWeight, 1e-6);
}
#endif

float evalVisibilityRaw(vec3 d) {
    #ifdef HAS_VISIBILITY_SV
        return evalVisibilitySVDeg3(d);
    #else
        return evalVisibilitySHDeg3(d);
    #endif
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
        if (uFrameOnlyMode && uIsDynamic) {
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
        if (uIsDynamic) {
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
            if (color.a < uVisibilityCullThreshold) {
                gl_Position = discardVec;
                return;
            }
        }
        #endif

        #ifdef HAS_VISIBILITY
            #ifdef FROZEN_OPACITY
                color.a = texelFetch(splatFrozenOpacity, splat.uv, 0).r;
            #else
                vec3 centerForVis = uIsDynamic ? computeDynamicPosition(modelCenter) : modelCenter;
                vec3 D_view = safeNormalizeVec3(centerForVis - uCameraPosition);
                float visRaw = evalVisibilityRaw(D_view);
                float visibility = sigmoid(visRaw);
                color.a *= visibility;
            #endif

            if (color.a < uVisibilityCullThreshold) {
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

uniform highp usampler2D splatTransform;        // per-splat index into transform palette
uniform sampler2D transformPalette;             // palette of transform matrices

uniform float uCurrentTime;                     // current absolute time for dynamic gaussians
uniform bool uIsDynamic;                        // whether this is a dynamic gaussian splat
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
    if (!uIsDynamic) {
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
    vec3 dynamicCenter = uIsDynamic ? computeDynamicPosition(modelCenter) : modelCenter;
    
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

export { vertexShader, fragmentShader, gsplatCenter };

#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day90

// https://www.shadertoy.com/view/3dSXD1 paper city

#define MARCH_STEPS 26
#define SHADOW_STEPS 1

float2x2 rot90(float a) {
    float ca = cos(a);
    float sa = sin(a);
    return float2x2(ca, sa, -sa, ca);
}

float box90(float3 p, float3 s) {
    float3 ap = abs(p) - s;
    return max(ap.x, max(ap.y, ap.z));
}

float tri(float3 p, float3 s) {
    p.y = -p.y;
    p.xz = abs(p.xz);
    return max(max(-p.y - s.y, dot(p.xy, float2(0.7)) - s.x), p.z - s.z);
}

float cone(float3 p, float a, float b) {
    return max(length(p.xz) - p.y * a, p.y - b);
}

float3 rep(float3 p, float3 s) {
    return (fract(p / s + 0.5) - 0.5) * s;
}

float2 rep(float2 p, float2 s) {
    return (fract(p / s + 0.5) - 0.5) * s;
}

float rep(float p, float s) {
    return (fract(p / s + 0.5) - 0.5) * s;
}

float house(float3 p, float s) {
    float t = tri(p + float3(0, 3, 0) * s, float3(1, 1, 3.5) * s);
    t = min(t, box90(p, float3(2, 2, 3) * s));
    return t;
}

float minitower(float3 p) {
    p.y += 5.0;
    float3 p2 = p;
    if (abs(p2.x) < abs(p2.z)) p2.xz = p2.zx;
    float t = min(house(p + float3(0, 3, 0), 0.5), house(p2, 1.0));
    t = min(t, house(p - float3(0, 5, 0), 1.5));
    return t;
}

float tower(float3 p) {
    p.y += 15.0;
    float3 p2 = p;
    if (abs(p2.x) < abs(p2.z)) p2.xz = p2.zx;
    float t = min(house(p + float3(0, 3, 0), 0.5), house(p2, 1.0));
    t = min(t, house(p - float3(0, 5, 0), 1.5));
    p2.x -= sign(p2.x) * 5.0;
    p2.x = abs(p2.x);
    p2.z = abs(p2.z);
    t = min(t, house(p2.zyx - float3(2, 8, 2), 0.3));
    t = min(t, house(p2 - float3(0, 12, 0), 1.5));
    return t;
}

float wall(float3 p) {
    p.x -= cos(p.z * 0.1) * 2.0;
    p.x -= sin(p.z * 0.03) * 3.0;
    float3 rp = p;
    rp.z = rep(rp.z, 5.0);
    float w = box90(rp + float3(0, 1, 0), float3(2, 1, 50));
    rp.x = abs(rp.x) - 2.0;
    float m = box90(rp - float3(0, 2, 0), float3(0.25, 5, 1.6));
    return min(w, m);
}

float field(float3 p) {
    float3 p2 = p;
    if (abs(p2.x) < abs(p2.z)) p2.xz = p2.zx;

    float tmp = box90(p2, float3(5, 5, 5));
    float f = max(abs(tmp - 4.0), -p.y - 2.0);
    f = min(f, box90(p, float3(7, 0.5, 7)));

    float3 p3 = p;
    p3.xz = rep(p3.xz, float2(2.5));

    float a = box90(p3, float3(0.2, 2, 0.2));
    a = min(a, cone(p3 + float3(0, 4, 0), 0.3, 3.0));
    f = min(f, max(a, tmp - 3.8));

    return f;
}

float village(float3 p) {
    float3 p2 = p;
    p2.xz = abs(p2.xz);
    float w = wall(p);
    p2.xz -= 23.0;
    float t = tower(p2);
    float3 p3 = p;
    p3.z = p3.z - 4.5 * sign(p.x);
    p3.x = abs(p3.x) - 25.0;
    float f = field(p3);

    float res = t;
    res = min(res, w);
    res = min(res, f);

    p.z = p.z + 10.0 * sign(p.x);
    p.x = -abs(p.x);
    res = min(res, minitower(p + float3(29, 1, 0)));

    return res;
}

float map(float3 p) {
    float t1 = sin(length(p.xz) * 0.009);
    float s = 12.0;
    for (int i = 0; i < 6; ++i) {
        p.xz = abs(p.xz) - s;
        p.xz = p.xz * rot90(0.55 + t1 + float(i) * 0.34);
        s /= 0.85;
    }
    p.x += 3.0;

    return min(village(p), -p.y);
}

float getao(float3 p, float3 n, float dist) {
    return clamp(map(p + n * dist) / dist, 0.0, 1.0);
}

float noise90(float2 p) {
    float2 ip = floor(p);
    p = smoothstep(0.0, 1.0, fract(p));
    float2 st = float2(67, 137);
    float2 v = dot(ip, st) + float2(0, st.y);
    float2 val = mix(fract(sin(v) * 9875.565), fract(sin(v + st.x) * 9875.565), p.x);
    return mix(val.x, val.y, p.y);
}

float fractal(float2 p) {
    float d = 0.5;
    float v = 0.0;
    for (int i = 0; i < 5; ++i) {
        v += noise90(p / d) * d;
        d *= 0.5;
    }
    return v;
}

float3 sky(float3 r, float3 l, float time) {
    float v = pow(max(dot(r, l), 0.0), 3.0);

    float2 sphereuv = float2(abs(atan2(r.z, r.x)) + time * 0.03, atan2(r.y, length(r.xz)));

    float skyn = fractal(sphereuv * float2(5, 10));
    float skyn2 = fractal(sphereuv * float2(5, 10) * 0.3 - float2(time * 0.06, 0));
    skyn2=smoothstep(0.3, 0.7, skyn2);

    float3 blue = mix(float3(0.5, 0.5, 0.8), float3(0.0), skyn2 * skyn);

    return mix(blue * 0.2, float3(1, 0.7, 0.4) * (skyn2 * 0.8 + 0.2), v);
}

float3 sky2(float3 r, float3 l) {
    float v = pow(max(dot(r, l), 0.0), 3.0);

    float3 blue = float3(0.5, 0.5, 0.8);

    return mix(blue * 0.2, float3(1, 0.7, 0.4), v);
}

fragment float4 shader_day90(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = float2(pixPos.x / res.x, pixPos.y / res.y);
    uv -= 0.5;
    uv /= float2(res.y / res.x, 1.0);
    uv.y *= -1.0;

    float t2 = time + 10.0;
    float3 s = float3(0, 0, -100);
    s.yz = s.yz * rot90(sin(t2 * 0.3) * 0.2 + 0.5);
    s.xz = s.xz * rot90(t2 * 0.2);
    float3 t = float3(0, 30, 60);
    t.yz = t.yz * rot90(sin(t2) * 0.3 - 0.2);
    t.xz = t.xz * rot90(t2 * 0.32);

    float3 cz = normalize(t - s);
    float3 cx = normalize(cross(cz, float3(0, 1, 0)));
    float3 cy = normalize(cross(cz, cx));
    float3 r = normalize(uv.x * cx + uv.y * cy + cz * 0.7);

    float3 p = s;
    float dd = 0.0;
    for (int i = 0; i < MARCH_STEPS; ++i) {
        float d = map(p);
        if (abs(d) < 0.001) break;
        if (dd > 500.0) { dd = 500.0; break; }
        p += d * r * 0.8;
        dd += d;
    }

    float fog = 1.0 - clamp(dd / 500.0, 0.0, 1.0);

    float3 col = float3(0);
    float2 off = float2(0.01, 0);
    float3 n = normalize(map(p) - float3(map(p - off.xyy), map(p - off.yxy), map(p - off.yyx)));

    float ao = (getao(p, n, 12.0) * 0.5 + 0.5) * (getao(p, n, 2.0) * 0.3 + 0.7) * (getao(p, n, 0.5) * 0.8 + 0.2);

    float3 l = normalize(float3(-1, -2, -2.5));
    float f = pow(1.0 - abs(dot(n, r)), 3.0);

    float shad = 1.0;
    float3 sp = p + n * 0.5 - r * 0.2;
    for (int i = 0; i < SHADOW_STEPS; ++i) {
        float d = map(sp);
        if (d < 0.2) { shad = 0.0; break; }
        sp += d * l * 3.0;
    }

    col += max(0.0, dot(n, l)) * fog * float3(1, 0.7, 0.4) * 1.5 * mix(0.0, ao * 0.5 + 0.5, shad);
    col += (-n.y * 0.5 + 0.5) * ao * fog * float3(0.5, 0.5, 0.8) * 0.5;
    col += sky2(reflect(r, n), l) * f * 10.0 * fog * (0.5 + 0.5 * shad);

    col += sky(r, l, time) * pow(dd * 0.01, 1.4);

    col = 1.0 - exp(-col * 2.5);
    col = pow(col, float3(2.3));
    col = pow(col, float3(0.4545));

    return float4(col, 1.0);
}

// MARK: - Day91

// https://www.shadertoy.com/view/tslyRf Supernova

float2x2 Rotate2D(float angle) {
    return float2x2(cos(angle), sin(angle), -sin(angle), cos(angle));
}

float TextureCoordinate(float2 position, sampler s, texture2d<float, access::sample> texture) {
    float zoomingFactor = 60.0;
    return texture.sample(s, position / zoomingFactor, 0.0).x;
}

float FBM(float2 uv, sampler s, texture2d<float, access::sample> texture) {
    float innerPower = 2.0;
    float noiseValue = 0.0;
    float brightness = 2.0;
    float dampeningFactor = 2.0;
    float offset = 0.5;
    float difference = 2.0;
    for (int i = 0; i < 5; ++i) {
        noiseValue += abs((TextureCoordinate(uv, s, texture) - offset) * difference) / brightness;
        brightness *= dampeningFactor;
        uv *= innerPower;
    }
    return noiseValue;
}

float Turbulence(float2 uv, float globalTime, sampler s, texture2d<float, access::sample> texture) {
    float activityLevel = 3.0;
    float2 noiseBasisDiag = float2(FBM(uv - globalTime * activityLevel, s, texture), FBM(uv + globalTime * activityLevel, s, texture));
    uv += noiseBasisDiag;
    float rotationSpeed = 2.0;
    return FBM(uv * Rotate2D(globalTime * rotationSpeed), s, texture);
}

float Ring(float2 uv) {
    float circleRadius = sqrt(length(uv));
    float range = 2.3;
    float functionSlope = 15.0;
    float offset = 0.5;
    return abs(mod(circleRadius, range) - range / 2.0) * functionSlope + offset;
}

fragment float4 shader_day91(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(2)]]) {

    constexpr sampler s(address::repeat, filter::linear);

    float2 uv = float2(pixPos.x / res.x, pixPos.y / res.y) - 0.5;
    uv /= float2(res.y / res.x, 1.0);

    float globalTime = time * 0.1;

    float distanceFromCenter = length(uv);
    float radius = 0.4;
    float alpha = 1.0;
    float alphaFalloffSpeed = 0.08;

    if (distanceFromCenter > radius) {
        alpha = max(0.0, 1.0 - (distanceFromCenter - radius) / alphaFalloffSpeed);
    }

    if (alpha == 0.0) {
        discard_fragment();
    }

    float2 uvZoomed = uv * 4.0;

    float fractalColor = Turbulence(uvZoomed, globalTime, s, texture);
    fractalColor *= Ring(uvZoomed);
    float3 col = normalize(float3(0.721, 0.311, 0.165)) / fractalColor;
    col *= alpha;
    return float4(col, 1.0);
}

// MARK: - Day92

// https://www.shadertoy.com/view/XdlGz8 emboss

struct C_Sample {
    float3 vAlbedo;
    float3 vNormal;
};

C_Sample SampleMaterial(const float2 vUV, sampler s, texture2d<float, access::sample> texture, const float2 vTextureSize, const float fNormalScale) {
    C_Sample result;

    float2 vInvTextureSize = float2(1.0) / vTextureSize;

    float3 cSampleNegXNegY = texture.sample(s, vUV + (float2(-1.0, -1.0)) * vInvTextureSize.xy).rgb;
    float3 cSampleZerXNegY = texture.sample(s, vUV + (float2( 0.0, -1.0)) * vInvTextureSize.xy).rgb;
    float3 cSamplePosXNegY = texture.sample(s, vUV + (float2( 1.0, -1.0)) * vInvTextureSize.xy).rgb;

    float3 cSampleNegXZerY = texture.sample(s, vUV + (float2(-1.0, 0.0)) * vInvTextureSize.xy).rgb;
    float3 cSampleZerXZerY = texture.sample(s, vUV + (float2( 0.0, 0.0)) * vInvTextureSize.xy).rgb;
    float3 cSamplePosXZerY = texture.sample(s, vUV + (float2( 1.0, 0.0)) * vInvTextureSize.xy).rgb;

    float3 cSampleNegXPosY = texture.sample(s, vUV + (float2(-1.0,  1.0)) * vInvTextureSize.xy).rgb;
    float3 cSampleZerXPosY = texture.sample(s, vUV + (float2( 0.0,  1.0)) * vInvTextureSize.xy).rgb;
    float3 cSamplePosXPosY = texture.sample(s, vUV + (float2( 1.0,  1.0)) * vInvTextureSize.xy).rgb;

    // convert to linear
    float3 cLSampleNegXNegY = cSampleNegXNegY * cSampleNegXNegY;
    float3 cLSampleZerXNegY = cSampleZerXNegY * cSampleZerXNegY;
    float3 cLSamplePosXNegY = cSamplePosXNegY * cSamplePosXNegY;

    float3 cLSampleNegXZerY = cSampleNegXZerY * cSampleNegXZerY;
    float3 cLSampleZerXZerY = cSampleZerXZerY * cSampleZerXZerY;
    float3 cLSamplePosXZerY = cSamplePosXZerY * cSamplePosXZerY;

    float3 cLSampleNegXPosY = cSampleNegXPosY * cSampleNegXPosY;
    float3 cLSampleZerXPosY = cSampleZerXPosY * cSampleZerXPosY;
    float3 cLSamplePosXPosY = cSamplePosXPosY * cSamplePosXPosY;

    // Average samples to get albdeo colour
    result.vAlbedo = ( cLSampleNegXNegY + cLSampleZerXNegY + cLSamplePosXNegY
                      + cLSampleNegXZerY + cLSampleZerXZerY + cLSamplePosXZerY
                      + cLSampleNegXPosY + cLSampleZerXPosY + cLSamplePosXPosY ) / 9.0;

    float3 vScale = float3(0.3333);

    float fSampleNegXNegY = dot(cSampleNegXNegY, vScale);
    float fSampleZerXNegY = dot(cSampleZerXNegY, vScale);
    float fSamplePosXNegY = dot(cSamplePosXNegY, vScale);

    float fSampleNegXZerY = dot(cSampleNegXZerY, vScale);
    //float fSampleZerXZerY = dot(cSampleZerXZerY, vScale);
    float fSamplePosXZerY = dot(cSamplePosXZerY, vScale);

    float fSampleNegXPosY = dot(cSampleNegXPosY, vScale);
    float fSampleZerXPosY = dot(cSampleZerXPosY, vScale);
    float fSamplePosXPosY = dot(cSamplePosXPosY, vScale);

    float2 vEdge;
    vEdge.x = (fSampleNegXNegY - fSamplePosXNegY) * 0.25
    + (fSampleNegXZerY - fSamplePosXZerY) * 0.5
    + (fSampleNegXPosY - fSamplePosXPosY) * 0.25;

    vEdge.y = (fSampleNegXNegY - fSampleNegXPosY) * 0.25
    + (fSampleZerXNegY - fSampleZerXPosY) * 0.5
    + (fSamplePosXNegY - fSamplePosXPosY) * 0.25;

    result.vNormal = normalize(float3(vEdge * fNormalScale, 1.0));

    return result;
}

fragment float4 shader_day92(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::repeat, filter::linear);

    float2 vUV = pixPos.xy / res.xy;

    C_Sample materialSample;

    float fNormalScale = 10.0;
    materialSample = SampleMaterial(vUV, s, texture, res, fNormalScale);

    float fLightHeight = 0.2;
    float fViewHeight = 2.0;
    float3 vSurfacePos = float3(vUV, 0.0);
    float3 vViewPos = float3(0.5, 0.5, fViewHeight);
    float3 vLightPos = float3(float2(sin(time), cos(time)) * 0.25 + 0.5, fLightHeight);

    float3 vDirToView = normalize(vViewPos - vSurfacePos);
    float3 vDirToLight = normalize(vLightPos - vSurfacePos);

    float fNDotL = clamp(dot(materialSample.vNormal, vDirToLight), 0.0, 1.0);
    float fDiffuse = fNDotL;

    float3 vHalf = normalize(vDirToView + vDirToLight);
    float fNDotH = clamp(dot(materialSample.vNormal, vHalf), 0.0, 1.0);
    float fSpec = pow(fNDotH, 10.0) * fNDotL * 0.5;

    float3 vResult = materialSample.vAlbedo * fDiffuse + fSpec;
    vResult = sqrt(vResult);

    return float4(vResult, 1.0);
}

// MARK: - Day93

// https://www.shadertoy.com/view/wslcWf Probabilistic quadtree filter

// the number of divisions at the start
#define MIN_DIVISIONS 4.0

// the numer of possible quad divisions
#define MAX_ITERATIONS 5

// the number of samples picked fter each quad division
#define SAMPLES_PER_ITERATION 30
#define F_SAMPLES_PER_ITERATION 30.

// useless, kept it for reference for a personal usage
#define MAX_SAMPLES 200

// threshold min, max given the mouse.x
#define THRESHOLD_MIN 0.01
#define THRESHOLD_MAX 0.05

float2 hash22(float2 p) {
    float n = sin(dot(p, float2(41, 289)));
    return fract(float2(262144, 32768) * n);
}

float gscale(float4 color) {
    return (color.r + color.g + color.b + color.a) / 4.0;
}

float quadColorVariation (float2 center, float size, sampler sa, texture2d<float, access::sample> texture) {
    float samplesBuffer[SAMPLES_PER_ITERATION];

    float avg = 0.0;

    for (int i = 0; i < SAMPLES_PER_ITERATION; i++) {
        float fi = float(i);
        float2 r = hash22(center.xy + float2(fi, 0.0)) - 0.5;
        float sp = gscale(texture.sample(sa, center + r * size));
        avg+= sp;
        samplesBuffer[i] = sp;
    }

    avg /= F_SAMPLES_PER_ITERATION;

    float var = 0.0;
    for (int i = 0; i < SAMPLES_PER_ITERATION; i++) {
        var+= abs(samplesBuffer[i] - avg);
    }
    return var / F_SAMPLES_PER_ITERATION;
}

fragment float4 shader_day93(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler sa(address::clamp_to_edge, filter::linear);

    float2 uv = pixPos.xy / res;

    float threshold = mix(THRESHOLD_MIN, THRESHOLD_MAX, 1.0 / res.x);

    float divs = MIN_DIVISIONS;

    float2 quadCenter = (floor(uv * divs) + 0.5) / divs;
    float quadSize = 1.0 / divs;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        float var = quadColorVariation(quadCenter, quadSize, sa, texture);
        if (var < threshold) break;
        divs *= 2.0;
        quadCenter = (floor(uv * divs) + 0.5) / divs;
        quadSize /= 2.0;
    }

    float4 color = texture.sample(sa, uv);

    float2 nUv = fract(uv * divs);

    float2 lWidth = float2(1.0 / res.x, 1.0 / res.y);
    float2 uvAbs = abs(nUv - 0.5);
    float s = step(0.5 - uvAbs.x, lWidth.x * divs) + step(0.5 - uvAbs.y, lWidth.y * divs);
    color-= s;

    return color;
}

// MARK: - Day94

// https://www.shadertoy.com/view/3sXyRN Corridor

float2x2 rot94(float a) {
    float ca = cos(a);
    float sa = sin(a);
    return float2x2(ca, sa, -sa, ca);
}

float3 cam(float3 p, float t) {
    t *= 0.3;
    p.xz = p.xz * rot94(sin(t) * 0.3);
    p.xy = p.xy * rot94(sin(t * 0.7) * 0.4);
    return p;
}

float hash94(float t) {
    return fract(sin(t * 788.874));
}

float curve(float t, float d) {
    t /= d;
    return mix(hash94(floor(t)), hash94(floor(t) + 1.0), pow(smoothstep(0.0, 1.0, fract(t)), 10.0));
}

float tick(float t, float d) {
    t /= d;
    float m = fract(t);
    m = smoothstep(0.0, 1.0, m);
    m = smoothstep(0.0, 1.0, m);
    return (floor(t) + m) * d;
}

float hash2_94(float2 uv) {
    return fract(dot(sin(uv * 425.215 + uv.yx * 714.388), float2(522.877)));
}

float2 hash22_94(float2 uv) {
    return fract(sin(uv * 425.215 + uv.yx * 714.388) * float2(522.877));
}

float3 hash3_94(float2 id) {
    return fract(sin(id.xyy * float3(427.544, 224.877, 974.542) + id.yxx * float3(947.544, 547.847, 652.454)) * 342.774);
}

float camtime(float t) {
    return t * 1.9 + tick(t, 1.9) * 1.0;
}

fragment float4 shader_day94(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float timeX = mod(time, 300.0);

    float2 uv = float2(pixPos.x / res.x, pixPos.y / res.y);
    uv -= 0.5;
    uv /= float2(res.y / res.x, 1.0);

    float3 col = float3(0.0);

    float3 size = float3(0.9, 0.9, 1000.0);

    float dof = 0.02;
    float dofdist = 1.0 / 5.0;

    for (float j = 0.0; j < 10.0; ++j) {

        // DOF offset
        float2 off = hash22_94(uv + j * 74.542 + 35.877) * 2.0 - 1.0;

        // Motion blur offset
        float t2 = camtime(timeX + j * 0.05 / 10.0);

        float3 s = float3(0.0, 0.0, -1.0);
        s.xy += off * dof;
        float3 r = normalize(float3(-uv - off * dof * dofdist, 2.0));

        cam(s, t2);
        cam(r, t2);

        float3 alpha = float3(1.0);

        // Bounces
        for (float i = 0.0; i < 3.0; ++i) {
            // box collision
            float3 boxmin = (size - s) / r;
            float3 boxmax = (-size - s) / r;

            float3 box = max(boxmin, boxmax);

            // only check box x and y axis
            float d = min(box.x, box.y);
            float3 p = s + r * d;
            float2 cuv = p.xz;
            float3 n = float3(0.0, sign(box.y), 0.0);

            if (box.x < box.y) {
                cuv = p.yz;
                cuv.x += 1.0;
                n = float3(sign(box.x), 0.0, 0.0);
            }

            float3 p2 = p;
            p2.z += t2 * 3.0;
            cuv.y += t2 * 3.0;
            cuv *= 3.0;
            float2 id = floor(cuv);

            float rough = min(1.0, 0.85 + 0.2 * hash2_94(id + 100.5));

            float3 addcol = float3(0);
            addcol += float3(1.0+max(0.0,cos(cuv.y*0.025)*0.9),0.5,0.2+max(0.0,sin(cuv.y*0.05)*0.5))*2.0;
            addcol *= smoothstep(0.5*curve(time+id.y*0.01+id.x*0.03, 0.3),0.0,hash2_94(id));
            addcol *= step(0.5,sin(p2.x)*sin(p2.z*0.4));
            addcol += float3(0.7,0.5,1.2)*step(p2.y,-0.9)*max(0.0,curve(time,0.2)*2.0-1.0)*step(hash2_94(id+.7),0.2);
            col += addcol * alpha;

            float fre = pow(1.0 - max(0.0, dot(n, r)), 3.0);
            alpha *= fre * 0.9;

            float3 pure = reflect(r,n);

            r = normalize(hash3_94(uv + j * 74.524 + i * 35.712) - 0.5);
            float dr = dot(r,n);
            if (dr < 0.0) r = -r;
            r = normalize(mix(r, pure,rough));

            s = p;
        }
    }
    col /= 10.0;
    col *= 2.0;
    col = smoothstep(0.0, 1.0, col);
    col = pow(col, float3(0.4545));
    return float4(col, 1.0);
}

// MARK: - Day95

// https://www.shadertoy.com/view/4dBGDy Green disks

float3 rotateX(float a, float3 v) {
    return float3(v.x, cos(a) * v.y + sin(a) * v.z, cos(a) * v.z - sin(a) * v.y);
}

float3 rotateY(float a, float3 v) {
    return float3(cos(a) * v.x + sin(a) * v.z, v.y, cos(a) * v.z - sin(a) * v.x);
}

float orbIntensity(float3 p) {
    if (length(p) < 4.0)
        return 1.0;
    return smoothstep(0.25, 1.0, cos(p.x * 10.0) * sin(p.y * 5.0) * cos(p.z * 7.0)) * 0.2 * step(length(p), 17.0);
}

float3 project(float3 p, float3 cam_origin, float3x3 cam_rotation) {
    float3x3 cam_rotation_t = float3x3(float3(cam_rotation[0].x, cam_rotation[1].x, cam_rotation[2].x),
                                       float3(cam_rotation[0].y, cam_rotation[1].y, cam_rotation[2].y),
                                       float3(cam_rotation[0].z, cam_rotation[1].z, cam_rotation[2].z));
    p = cam_rotation_t * (p - cam_origin);
    return float3(p.xy / p.z, p.z);
}

float orb(float rad, float3 cd, float2 pixPos) {
    return 1.0 - smoothstep(0.5, 0.55, distance(cd.xy, pixPos) / rad);
}

float orbShadow(float rad, float3 cd, float2 pixPos) {
    return 1.0 - smoothstep(0.4, 1.1, distance(cd.xy, pixPos) / rad) * mix(1.0, 0.99, orb(rad, cd, pixPos));
}

float3 traverseUniformGrid(float3 ro, float3 rd, float2 pixPos, float time, float3 cam_origin, float3x3 cam_rotation) {
    float3 increment = float3(1.0) / rd;
    float3 intersection = ((floor(ro) + round(rd * 0.5 + float3(0.5))) - ro) * increment;

    increment = abs(increment);
    ro += rd * 1e-3;

    float4 accum = float4(0.0, 0.0, 0.0, 1.0);

    for (int i = 0; i < 24; i += 1) {
        float3 rp = floor(ro + rd * min(intersection.x, min(intersection.y, intersection.z)));

        float orb_intensity = orbIntensity(rp);

        if (orb_intensity > 1e-3) {
            float3 cd = project(rp + float3(0.5), cam_origin, cam_rotation);
            if (cd.z > 1.0) {
                float rad = 0.55 / cd.z;
                rad *= 1.0 + 0.5 * sin(rp.x + time * 1.0) * cos(rp.y + time * 2.0) * cos(rp.z);
                float dist = distance(rp + float3(0.5), ro);
                float c = smoothstep(1.0, 2.5, dist);
                float a = orb(rad, cd, pixPos) * c;
                float b = orbShadow(rad, cd, pixPos) * c;
                accum.rgb += accum.a * a * 1.5 *
                mix(float3(1.0), float3(0.4, 1.0, 0.5) * 0.5, 0.5 + 0.5 * cos(rp.x)) * exp(-dist * dist * 0.008);
                accum.a *= 1.0 - b;
            }
        }

        intersection += increment * step(intersection.xyz, intersection.yxy) *
        step(intersection.xyz, intersection.zzx);
    }

    accum.rgb += accum.a * float3(0.02);
    return accum.rgb;
}

fragment float4 shader_day95(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(4)]]) {

    constexpr sampler s(address::repeat, filter::linear);

    float2 uv = pixPos.xy / res.xy;
    float timeX = time;
    float3 cam_origin;
    float3x3 cam_rotation;

    float2 frag_coord = uv * 2.0 - float2(1.0);
    frag_coord.x *= res.x / res.y;

    float time0 = time;
    float time1 = time0 + 0.04;

    float jitter = texture.sample(s, uv * res.xy / 256.0).r;

    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);

    for (int n = 0; n < 2; n += 1) {
        timeX = mix(time0, time1, (float(n) + jitter) / 4.0) * 0.7;

        cam_origin = rotateX(timeX * 0.3, rotateY(timeX * 0.5, float3(0.0, 0.0, -10.0)));

        float3 cam_w = normalize(float3(cos(timeX) * 10.0, 0.0, 0.0) - cam_origin);
        float3 cam_u = normalize(cross(cam_w, float3(0.0, 1.0, 0.0)));
        float3 cam_v = normalize(cross(cam_u, cam_w));

        cam_rotation = float3x3(cam_u, cam_v, cam_w);

        float3 ro = cam_origin,rd = cam_rotation * float3(frag_coord, 1.0);

        fragColor.rgb += traverseUniformGrid(ro, rd, frag_coord, timeX, cam_origin, cam_rotation);
    }

    fragColor.rgb = sqrt(fragColor.rgb / 4.0 * 0.8);

    return fragColor;
}

// MARK: - Day96

// https://www.shadertoy.com/view/ltfGD7 Wind walker Ocean

#define WATER_COL float3(0.0, 0.4453, 0.7305)
#define WATER2_COL float3(0.0, 0.4180, 0.6758)
#define FOAM_COL float3(0.8125, 0.9609, 0.9648)
#define FOG_COL float3(0.6406, 0.9453, 0.9336)
#define SKY_COL float3(0.0, 0.8203, 1.0)

#define M_2PI 6.283185307
#define M_6PI 18.84955592

float circ(float2 pos, float2 c, float s) {
    c = abs(pos - c);
    c = min(c, 1.0 - c);
    return dot(c, c) < s ? -1.0 : 0.0;
}

float2 mod96(float2 x, float2 y) {
    return x - y * floor(x / y);
}

float waterlayer(float2 uv) {
    uv = mod96(uv, 1.0);
    float ret = 1.0;
    ret += circ(uv, float2(0.37378, 0.277169), 0.0268181);
    ret += circ(uv, float2(0.0317477, 0.540372), 0.0193742);
    ret += circ(uv, float2(0.430044, 0.882218), 0.0232337);
    ret += circ(uv, float2(0.641033, 0.695106), 0.0117864);
    ret += circ(uv, float2(0.0146398, 0.0791346), 0.0299458);
    ret += circ(uv, float2(0.43871, 0.394445), 0.0289087);
    ret += circ(uv, float2(0.909446, 0.878141), 0.028466);
    ret += circ(uv, float2(0.310149, 0.686637), 0.0128496);
    ret += circ(uv, float2(0.928617, 0.195986), 0.0152041);
    ret += circ(uv, float2(0.0438506, 0.868153), 0.0268601);
    ret += circ(uv, float2(0.308619, 0.194937), 0.00806102);
    ret += circ(uv, float2(0.349922, 0.449714), 0.00928667);
    ret += circ(uv, float2(0.0449556, 0.953415), 0.023126);
    ret += circ(uv, float2(0.117761, 0.503309), 0.0151272);
    ret += circ(uv, float2(0.563517, 0.244991), 0.0292322);
    ret += circ(uv, float2(0.566936, 0.954457), 0.00981141);
    ret += circ(uv, float2(0.0489944, 0.200931), 0.0178746);
    ret += circ(uv, float2(0.569297, 0.624893), 0.0132408);
    ret += circ(uv, float2(0.298347, 0.710972), 0.0114426);
    ret += circ(uv, float2(0.878141, 0.771279), 0.00322719);
    ret += circ(uv, float2(0.150995, 0.376221), 0.00216157);
    ret += circ(uv, float2(0.119673, 0.541984), 0.0124621);
    ret += circ(uv, float2(0.629598, 0.295629), 0.0198736);
    ret += circ(uv, float2(0.334357, 0.266278), 0.0187145);
    ret += circ(uv, float2(0.918044, 0.968163), 0.0182928);
    ret += circ(uv, float2(0.965445, 0.505026), 0.006348);
    ret += circ(uv, float2(0.514847, 0.865444), 0.00623523);
    ret += circ(uv, float2(0.710575, 0.0415131), 0.00322689);
    ret += circ(uv, float2(0.71403, 0.576945), 0.0215641);
    ret += circ(uv, float2(0.748873, 0.413325), 0.0110795);
    ret += circ(uv, float2(0.0623365, 0.896713), 0.0236203);
    ret += circ(uv, float2(0.980482, 0.473849), 0.00573439);
    ret += circ(uv, float2(0.647463, 0.654349), 0.0188713);
    ret += circ(uv, float2(0.651406, 0.981297), 0.00710875);
    ret += circ(uv, float2(0.428928, 0.382426), 0.0298806);
    ret += circ(uv, float2(0.811545, 0.62568), 0.00265539);
    ret += circ(uv, float2(0.400787, 0.74162), 0.00486609);
    ret += circ(uv, float2(0.331283, 0.418536), 0.00598028);
    ret += circ(uv, float2(0.894762, 0.0657997), 0.00760375);
    ret += circ(uv, float2(0.525104, 0.572233), 0.0141796);
    ret += circ(uv, float2(0.431526, 0.911372), 0.0213234);
    ret += circ(uv, float2(0.658212, 0.910553), 0.000741023);
    ret += circ(uv, float2(0.514523, 0.243263), 0.0270685);
    ret += circ(uv, float2(0.0249494, 0.252872), 0.00876653);
    ret += circ(uv, float2(0.502214, 0.47269), 0.0234534);
    ret += circ(uv, float2(0.693271, 0.431469), 0.0246533);
    ret += circ(uv, float2(0.415, 0.884418), 0.0271696);
    ret += circ(uv, float2(0.149073, 0.41204), 0.00497198);
    ret += circ(uv, float2(0.533816, 0.897634), 0.00650833);
    ret += circ(uv, float2(0.0409132, 0.83406), 0.0191398);
    ret += circ(uv, float2(0.638585, 0.646019), 0.0206129);
    ret += circ(uv, float2(0.660342, 0.966541), 0.0053511);
    ret += circ(uv, float2(0.513783, 0.142233), 0.00471653);
    ret += circ(uv, float2(0.124305, 0.644263), 0.00116724);
    ret += circ(uv, float2(0.99871, 0.583864), 0.0107329);
    ret += circ(uv, float2(0.894879, 0.233289), 0.00667092);
    ret += circ(uv, float2(0.246286, 0.682766), 0.00411623);
    ret += circ(uv, float2(0.0761895, 0.16327), 0.0145935);
    ret += circ(uv, float2(0.949386, 0.802936), 0.0100873);
    ret += circ(uv, float2(0.480122, 0.196554), 0.0110185);
    ret += circ(uv, float2(0.896854, 0.803707), 0.013969);
    ret += circ(uv, float2(0.292865, 0.762973), 0.00566413);
    ret += circ(uv, float2(0.0995585, 0.117457), 0.00869407);
    ret += circ(uv, float2(0.377713, 0.00335442), 0.0063147);
    ret += circ(uv, float2(0.506365, 0.531118), 0.0144016);
    ret += circ(uv, float2(0.408806, 0.894771), 0.0243923);
    ret += circ(uv, float2(0.143579, 0.85138), 0.00418529);
    ret += circ(uv, float2(0.0902811, 0.181775), 0.0108896);
    ret += circ(uv, float2(0.780695, 0.394644), 0.00475475);
    ret += circ(uv, float2(0.298036, 0.625531), 0.00325285);
    ret += circ(uv, float2(0.218423, 0.714537), 0.00157212);
    ret += circ(uv, float2(0.658836, 0.159556), 0.00225897);
    ret += circ(uv, float2(0.987324, 0.146545), 0.0288391);
    ret += circ(uv, float2(0.222646, 0.251694), 0.00092276);
    ret += circ(uv, float2(0.159826, 0.528063), 0.00605293);
    return max(ret, 0.0);
}

float3 water(float2 uv, float3 cdir, float time) {
    uv *= float2(0.25);

    float2 a = 0.025 * cdir.xz / cdir.y;
    float h = sin(uv.x + time);
    uv += a * h;
    h = sin(0.841471 * uv.x - 0.540302 * uv.y + time);
    uv += a * h;

    float d1 = mod(uv.x + uv.y, M_2PI);
    float d2 = mod((uv.x + uv.y + 0.25) * 1.3, M_6PI);
    d1 = time * 0.07 + d1;
    d2 = time * 0.5 + d2;
    float2 dist = float2(
                     sin(d1) * 0.15 + sin(d2) * 0.05,
                     cos(d1) * 0.15 + cos(d2) * 0.05
                     );

    float3 ret = mix(WATER_COL, WATER2_COL, waterlayer(uv + dist.xy));
    ret = mix(ret, FOAM_COL, waterlayer(float2(1.0) - uv - dist.yx));
    return ret;
}

float3 pixtoray(float2 uv, float2 res) {
    float3 pixpos;
    pixpos.x = uv.x - 0.5;
    pixpos.y = uv.y + 0.5;
    pixpos.y *= res.y / res.x;
    pixpos.z = -0.6;
    return normalize(pixpos);
}

float3 quatmul(float4 q, float3 v) {
    float3 qfloat = q.xyz;
    float3 uv = cross(qfloat, v);
    float3 uuv = cross(qfloat, uv);
    uv *= (2.0 * q.w);
    uuv *= 2.0;
    return v + uv + uuv;
}

fragment float4 shader_day96(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float4 ret = float4(0.0, 0.0, 0.0, 1.0);
    float2 offset = float2(0.0);

    float2 uv = (pixPos.xy + offset) / res.xy;
    uv.y *= -1.0;

    float3 cpos = float3(0.0, 4.0, 10.0);
    float3 cdir = pixtoray(uv, res);
    cdir = quatmul(float4(-0.19867, 0.0, 0.0, 0.980067), cdir);

    float cost = cos(time * -0.05);
    float sint = sin(time * -0.05);
    cdir.xz = cost * cdir.xz + sint * float2(-cdir.z, cdir.x);
    cpos.xz = cost * cpos.xz + sint * float2(-cpos.z, cpos.x);

    const float3 ocean = float3(0.0, 1.0, 0.0);
    float dist = -dot(cpos, ocean) / dot(cdir, ocean);
    float3 pos = cpos + dist * cdir;

    float3 pix;
    if (dist > 0.0 && dist < 100.0) {
        float3 wat = water(pos.xz, cdir, time);
        pix = mix(wat, FOG_COL, min(dist * 0.01, 1.0));
    } else {
        pix = mix(FOG_COL, SKY_COL, min(cdir.y * 4.0, 1.0));
    }
    ret.rgb += pix;

    return ret;
}

// MARK: - Day97

// https://www.shadertoy.com/view/XtlSD7 Mario 1-1

#define SPRITE_DEC(x, i)  mod( floor( i / pow( 4.0, mod( x, 8.0 ) ) ), 4.0 )
#define SPRITE_DEC2(x, i) mod( floor( i / pow( 4.0, mod( x, 11.0 ) ) ), 4.0 )
#define RGB(r, g, b)      float3( float( r ) / 255.0, float( g ) / 255.0, float( b ) / 255.0 )

#define MARIO_SPEED 89.0
#define GOOMBA_SPEED 32.0

float3 SpriteBlock(float3 color, float x, float y) {
    // black
    float idx = 1.0;

    // light orange
    idx = x < y ? 3.0 : idx;

    // dark orange
    idx = x > 3.0 && x < 12.0 && y > 3.0 && y < 12.0 ? 2.0 : idx;
    idx = x == 15.0 - y ? 2.0 : idx;

    color = RGB( 0, 0, 0 );
    color = idx == 2.0 ? RGB( 231,  90,  16 ) : color;
    color = idx == 3.0 ? RGB( 247, 214, 181 ) : color;

    return color;
}

float3 SpriteHill(float3 color, float x, float y) {
    float idx = 0.0;

    // dark green
    idx = ( x > y && 79.0 - x > y ) && y < 33.0 ? 2.0 : idx;
    idx = ( x >= 37.0 && x <= 42.0 ) && y == 33.0 ? 2.0 : idx;

    // black
    idx = ( x == y || 79.0 - x == y ) && y < 33.0 ? 1.0 : idx;
    idx = ( x == 33.0 || x == 46.0 ) && y == 32.0 ? 1.0 : idx;
    idx = ( x >= 34.0 && x <= 36.0 ) && y == 33.0 ? 1.0 : idx;
    idx = ( x >= 43.0 && x <= 45.0 ) && y == 33.0 ? 1.0 : idx;
    idx = ( x >= 37.0 && x <= 42.0 ) && y == 34.0 ? 1.0 : idx;
    idx = ( x >= 25.0 && x <= 26.0 ) && ( y >= 8.0  && y <= 11.0 ) ? 1.0 : idx;
    idx = ( x >= 41.0 && x <= 42.0 ) && ( y >= 24.0 && y <= 27.0 ) ? 1.0 : idx;
    idx = ( x >= 49.0 && x <= 50.0 ) && ( y >= 8.0  && y <= 11.0 ) ? 1.0 : idx;
    idx = ( x >= 28.0 && x <= 30.0 ) && ( y >= 11.0 && y <= 14.0 ) ? 1.0 : idx;
    idx = ( x >= 28.0 && x <= 30.0 ) && ( y >= 11.0 && y <= 14.0 ) ? 1.0 : idx;
    idx = ( x >= 44.0 && x <= 46.0 ) && ( y >= 27.0 && y <= 30.0 ) ? 1.0 : idx;
    idx = ( x >= 44.0 && x <= 46.0 ) && ( y >= 27.0 && y <= 30.0 ) ? 1.0 : idx;
    idx = ( x >= 52.0 && x <= 54.0 ) && ( y >= 11.0 && y <= 14.0 ) ? 1.0 : idx;
    idx = ( x == 29.0 || x == 53.0 ) && ( y >= 10.0 && y <= 15.0 ) ? 1.0 : idx;
    idx = x == 45.0 && ( y >= 26.0 && y <= 31.0 ) ? 1.0 : idx;

    color = idx == 1.0 ? RGB( 0,     0,  0 ) : color;
    color = idx == 2.0 ? RGB( 0,   173,  0 ) : color;

    return color;
}

float3 SpritePipe(float3 color, float x, float y, float h) {
    float offset = h * 16.0;

    // light green
    float idx = 3.0;

    // dark green
    idx = ( ( x > 5.0 && x < 8.0 ) || ( x == 13.0 ) || ( x > 15.0 && x < 23.0 ) ) && y < 17.0 + offset ? 2.0 : idx;
    idx = ( ( x > 4.0 && x < 7.0 ) || ( x == 12.0 ) || ( x > 14.0 && x < 24.0 ) ) && ( y > 17.0 + offset && y < 30.0 + offset ) ? 2.0 : idx;
    idx = ( x < 5.0 || x > 11.0 ) && y == 29.0 + offset ? 2.0 : idx;
    idx = fract( x * 0.5 + y * 0.5 ) == 0.5 && x > 22.0 && ( ( x < 26.0 && y < 17.0 + offset ) || ( x < 28.0 && y > 17.0 + offset && y < 30.0 + offset ) ) ? 2.0 : idx;

    // black
    idx = y == 31.0 + offset || x == 0.0 || x == 31.0 || y == 17.0 + offset ? 1.0 : idx;
    idx = ( x == 2.0 || x == 29.0 ) && y < 18.0 + offset ? 1.0 : idx;
    idx = ( x > 1.0 && x < 31.0 ) && y == 16.0 + offset ? 1.0 : idx;

    // transparent
    idx = ( x < 2.0 || x > 29.0 ) && y < 17.0 + offset ? 0.0 : idx;

    color = idx == 1.0 ? RGB( 0,     0,  0 ) : color;
    color = idx == 2.0 ? RGB( 0,   173,  0 ) : color;
    color = idx == 3.0 ? RGB( 189, 255, 24 ) : color;

    return color;
}

float3 SpriteCloud(float3 color, float x, float y, float isBush) {
    float idx = 0.0;

    idx = y == 23.0 ? ( x <= 10.0 ? 0.0 : ( x <= 21.0 ? 5440.0 : 0.0 ) ) : idx;
    idx = y == 22.0 ? ( x <= 10.0 ? 0.0 : ( x <= 21.0 ? 32720.0 : 0.0 ) ) : idx;
    idx = y == 21.0 ? ( x <= 10.0 ? 0.0 : ( x <= 21.0 ? 131061.0 : 0.0 ) ) : idx;
    idx = y == 20.0 ? ( x <= 10.0 ? 1048576.0 : ( x <= 21.0 ? 1179647.0 : 0.0 ) ) : idx;
    idx = y == 19.0 ? ( x <= 10.0 ? 1048576.0 : ( x <= 21.0 ? 3670015.0 : 1.0 ) ) : idx;
    idx = y == 18.0 ? ( x <= 10.0 ? 1048576.0 : ( x <= 21.0 ? 4190207.0 : 7.0 ) ) : idx;
    idx = y == 17.0 ? ( x <= 10.0 ? 3407872.0 : ( x <= 21.0 ? 4177839.0 : 7.0 ) ) : idx;
    idx = y == 16.0 ? ( x <= 10.0 ? 3997696.0 : ( x <= 21.0 ? 4194299.0 : 7.0 ) ) : idx;
    idx = y == 15.0 ? ( x <= 10.0 ? 4150272.0 : ( x <= 21.0 ? 4194303.0 : 1055.0 ) ) : idx;
    idx = y == 14.0 ? ( x <= 10.0 ? 4193536.0 : ( x <= 21.0 ? 4194303.0 : 7455.0 ) ) : idx;
    idx = y == 13.0 ? ( x <= 10.0 ? 4194112.0 : ( x <= 21.0 ? 4194303.0 : 8063.0 ) ) : idx;
    idx = y == 12.0 ? ( x <= 10.0 ? 4194240.0 : ( x <= 21.0 ? 4194303.0 : 73727.0 ) ) : idx;
    idx = y == 11.0 ? ( x <= 10.0 ? 4194260.0 : ( x <= 21.0 ? 4194303.0 : 491519.0 ) ) : idx;
    idx = y == 10.0 ? ( x <= 10.0 ? 4194301.0 : ( x <= 21.0 ? 4194303.0 : 524287.0 ) ) : idx;
    idx = y == 9.0 ? ( x <= 10.0 ? 4194301.0 : ( x <= 21.0 ? 4194303.0 : 524287.0 ) ) : idx;
    idx = y == 8.0 ? ( x <= 10.0 ? 4194292.0 : ( x <= 21.0 ? 4194303.0 : 131071.0 ) ) : idx;
    idx = y == 7.0 ? ( x <= 10.0 ? 4193232.0 : ( x <= 21.0 ? 4194303.0 : 32767.0 ) ) : idx;
    idx = y == 6.0 ? ( x <= 10.0 ? 3927872.0 : ( x <= 21.0 ? 4193279.0 : 131071.0 ) ) : idx;
    idx = y == 5.0 ? ( x <= 10.0 ? 2800896.0 : ( x <= 21.0 ? 4193983.0 : 524287.0 ) ) : idx;
    idx = y == 4.0 ? ( x <= 10.0 ? 3144960.0 : ( x <= 21.0 ? 3144362.0 : 262143.0 ) ) : idx;
    idx = y == 3.0 ? ( x <= 10.0 ? 4150272.0 : ( x <= 21.0 ? 3845099.0 : 98303.0 ) ) : idx;
    idx = y == 2.0 ? ( x <= 10.0 ? 3997696.0 : ( x <= 21.0 ? 4107775.0 : 6111.0 ) ) : idx;
    idx = y == 1.0 ? ( x <= 10.0 ? 1310720.0 : ( x <= 21.0 ? 4183167.0 : 325.0 ) ) : idx;
    idx = y == 0.0 ? ( x <= 10.0 ? 0.0 : ( x <= 21.0 ? 1392661.0 : 0.0 ) ) : idx;

    idx = SPRITE_DEC2( x, idx );

    float3 colorB = isBush == 1.0 ? RGB( 0,   173,  0 ) : RGB(  57, 189, 255 );
    float3 colorC = isBush == 1.0 ? RGB( 189, 255, 24 ) : RGB( 254, 254, 254 );

    color = idx == 1.0 ? RGB( 0, 0, 0 ) : color;
    color = idx == 2.0 ? colorB         : color;
    color = idx == 3.0 ? colorC         : color;

    return color;
}

float3 SpriteFlag(float3 color, float x, float y) {
    float idx = 0.0;
    idx = y == 15.0 ? 43690.0 : idx;
    idx = y == 14.0 ? ( x <= 7.0 ? 43688.0 : 42326.0 ) : idx;
    idx = y == 13.0 ? ( x <= 7.0 ? 43680.0 : 38501.0 ) : idx;
    idx = y == 12.0 ? ( x <= 7.0 ? 43648.0 : 39529.0 ) : idx;
    idx = y == 11.0 ? ( x <= 7.0 ? 43520.0 : 39257.0 ) : idx;
    idx = y == 10.0 ? ( x <= 7.0 ? 43008.0 : 38293.0 ) : idx;
    idx = y == 9.0 ? ( x <= 7.0 ? 40960.0 : 38229.0 ) : idx;
    idx = y == 8.0 ? ( x <= 7.0 ? 32768.0 : 43354.0 ) : idx;
    idx = y == 7.0 ? ( x <= 7.0 ? 0.0 : 43690.0 ) : idx;
    idx = y == 6.0 ? ( x <= 7.0 ? 0.0 : 43688.0 ) : idx;
    idx = y == 5.0 ? ( x <= 7.0 ? 0.0 : 43680.0 ) : idx;
    idx = y == 4.0 ? ( x <= 7.0 ? 0.0 : 43648.0 ) : idx;
    idx = y == 3.0 ? ( x <= 7.0 ? 0.0 : 43520.0 ) : idx;
    idx = y == 2.0 ? ( x <= 7.0 ? 0.0 : 43008.0 ) : idx;
    idx = y == 1.0 ? ( x <= 7.0 ? 0.0 : 40960.0 ) : idx;
    idx = y == 0.0 ? ( x <= 7.0 ? 0.0 : 32768.0 ) : idx;

    idx = SPRITE_DEC( x, idx );

    color = idx == 1.0 ? RGB(   0, 173,   0 ) : color;
    color = idx == 2.0 ? RGB( 255, 255, 255 ) : color;

    return color;
}

float3 SpriteCastleFlag(float3 color, float x, float y) {
    float idx = 0.0;
    idx = y == 13.0 ? ( x <= 10.0 ? 8.0 : 0.0 ) : idx;
    idx = y == 12.0 ? ( x <= 10.0 ? 42.0 : 0.0 ) : idx;
    idx = y == 11.0 ? ( x <= 10.0 ? 8.0 : 0.0 ) : idx;
    idx = y == 10.0 ? ( x <= 10.0 ? 4194292.0 : 15.0 ) : idx;
    idx = y == 9.0 ? ( x <= 10.0 ? 4161524.0 : 15.0 ) : idx;
    idx = y == 8.0 ? ( x <= 10.0 ? 4161524.0 : 15.0 ) : idx;
    idx = y == 7.0 ? ( x <= 10.0 ? 1398260.0 : 15.0 ) : idx;
    idx = y == 6.0 ? ( x <= 10.0 ? 3495924.0 : 15.0 ) : idx;
    idx = y == 5.0 ? ( x <= 10.0 ? 4022260.0 : 15.0 ) : idx;
    idx = y == 4.0 ? ( x <= 10.0 ? 3528692.0 : 15.0 ) : idx;
    idx = y == 3.0 ? ( x <= 10.0 ? 3667956.0 : 15.0 ) : idx;
    idx = y == 2.0 ? ( x <= 10.0 ? 4194292.0 : 15.0 ) : idx;
    idx = y == 1.0 ? ( x <= 10.0 ? 4.0 : 0.0 ) : idx;
    idx = y == 0.0 ? ( x <= 10.0 ? 4.0 : 0.0 ) : idx;

    idx = SPRITE_DEC2( x, idx );

    color = idx == 1.0 ? RGB( 181,  49,  33 ) : color;
    color = idx == 2.0 ? RGB( 230, 156,  33 ) : color;
    color = idx == 3.0 ? RGB( 255, 255, 255 ) : color;

    return color;
}

float3 SpriteGoomba(float3 color, float x, float y, float frame) {
    float idx = 0.0;

    // second frame is flipped first frame
    x = frame == 1.0 ? 15.0 - x : x;

    if ( frame <= 1.0 )
    {
        idx = y == 15.0 ? ( x <= 7.0 ? 40960.0 : 10.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 43008.0 : 42.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 43520.0 : 170.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 43648.0 : 682.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 43360.0 : 2410.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 42920.0 : 10970.0 ) : idx;
        idx = y ==  9.0 ? ( x <= 7.0 ? 22440.0 : 10965.0 ) : idx;
        idx = y ==  8.0 ? ( x <= 7.0 ? 47018.0 : 43742.0 ) : idx;
        idx = y ==  7.0 ? ( x <= 7.0 ? 49066.0 : 43774.0 ) : idx;
        idx = y ==  6.0 ? 43690.0 : idx;
        idx = y ==  5.0 ? ( x <= 7.0 ? 65192.0 : 10943.0 ) : idx;
        idx = y ==  4.0 ? ( x <= 7.0 ? 65280.0 : 255.0 ) : idx;
        idx = y ==  3.0 ? ( x <= 7.0 ? 65280.0 : 1535.0 ) : idx;
        idx = y ==  2.0 ? ( x <= 7.0 ? 64832.0 : 5471.0 ) : idx;
        idx = y ==  1.0 ? ( x <= 7.0 ? 62784.0 : 5463.0 ) : idx;
        idx = y ==  0.0 ? ( x <= 7.0 ? 5376.0 : 1364.0 ) : idx;
    }
    else
    {
        idx = y == 7.0 ? ( x <= 7.0 ? 40960.0 : 10.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 43648.0 : 682.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 42344.0 : 10586.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 24570.0 : 45045.0 ) : idx;
        idx = y == 3.0 ? 43690.0 : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 65472.0 : 1023.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 65280.0 : 255.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 1364.0 : 5456.0 ) : idx;
    }

    idx = SPRITE_DEC( x, idx );

    color = idx == 1.0 ? RGB( 0,     0,   0 ) : color;
    color = idx == 2.0 ? RGB( 153,  75,  12 ) : color;
    color = idx == 3.0 ? RGB( 255, 200, 184 ) : color;

    return color;
}

float3 SpriteKoopa(float3 color, float x, float y, float frame) {
    float idx = 0.0;

    if ( frame == 0.0 )
    {
        idx = y == 23.0 ? ( x <= 7.0 ? 768.0 : 0.0 ) : idx;
        idx = y == 22.0 ? ( x <= 7.0 ? 4032.0 : 0.0 ) : idx;
        idx = y == 21.0 ? ( x <= 7.0 ? 4064.0 : 0.0 ) : idx;
        idx = y == 20.0 ? ( x <= 7.0 ? 12128.0 : 0.0 ) : idx;
        idx = y == 19.0 ? ( x <= 7.0 ? 12136.0 : 0.0 ) : idx;
        idx = y == 18.0 ? ( x <= 7.0 ? 12136.0 : 0.0 ) : idx;
        idx = y == 17.0 ? ( x <= 7.0 ? 12264.0 : 0.0 ) : idx;
        idx = y == 16.0 ? ( x <= 7.0 ? 11174.0 : 0.0 ) : idx;
        idx = y == 15.0 ? ( x <= 7.0 ? 10922.0 : 0.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 10282.0 : 341.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 30730.0 : 1622.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 31232.0 : 1433.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 24192.0 : 8037.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 24232.0 : 7577.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 28320.0 : 9814.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 40832.0 : 6485.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 26496.0 : 9814.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 23424.0 : 5529.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 22272.0 : 5477.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 24320.0 : 64921.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 65024.0 : 12246.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 59904.0 : 11007.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 43008.0 : 10752.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 40960.0 : 2690.0 ) : idx;
    }
    else
    {
        idx = y == 22.0 ? ( x <= 7.0 ? 192.0 : 0.0 ) : idx;
        idx = y == 21.0 ? ( x <= 7.0 ? 1008.0 : 0.0 ) : idx;
        idx = y == 20.0 ? ( x <= 7.0 ? 3056.0 : 0.0 ) : idx;
        idx = y == 19.0 ? ( x <= 7.0 ? 11224.0 : 0.0 ) : idx;
        idx = y == 18.0 ? ( x <= 7.0 ? 11224.0 : 0.0 ) : idx;
        idx = y == 17.0 ? ( x <= 7.0 ? 11224.0 : 0.0 ) : idx;
        idx = y == 16.0 ? ( x <= 7.0 ? 11256.0 : 0.0 ) : idx;
        idx = y == 15.0 ? ( x <= 7.0 ? 10986.0 : 0.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 10918.0 : 0.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 2730.0 : 341.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 18986.0 : 1622.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 18954.0 : 5529.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 24202.0 : 8037.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 24200.0 : 7577.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 28288.0 : 9814.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 40864.0 : 6485.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 26496.0 : 9814.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 23424.0 : 5529.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 22272.0 : 5477.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 24320.0 : 64921.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 65152.0 : 4054.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 60064.0 : 11007.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 2728.0 : 43520.0 ) : idx;
    }

    idx = SPRITE_DEC( x, idx );

    color = idx == 1.0 ? RGB( 30,  132,   0 ) : color;
    color = idx == 2.0 ? RGB( 215, 141,  34 ) : color;
    color = idx == 3.0 ? RGB( 255, 255, 255 ) : color;

    return color;
}

float3 SpriteQuestion(float3 color, float x, float y, float t) {
    float idx = 0.0;
    idx = y == 15.0 ? ( x <= 7.0 ? 43688.0 : 10922.0 ) : idx;
    idx = y == 14.0 ? ( x <= 7.0 ? 65534.0 : 32767.0 ) : idx;
    idx = y == 13.0 ? ( x <= 7.0 ? 65502.0 : 30719.0 ) : idx;
    idx = y == 12.0 ? ( x <= 7.0 ? 44030.0 : 32762.0 ) : idx;
    idx = y == 11.0 ? ( x <= 7.0 ? 23294.0 : 32745.0 ) : idx;
    idx = y == 10.0 ? ( x <= 7.0 ? 56062.0 : 32619.0 ) : idx;
    idx = y == 9.0 ? ( x <= 7.0 ? 56062.0 : 32619.0 ) : idx;
    idx = y == 8.0 ? ( x <= 7.0 ? 55294.0 : 32618.0 ) : idx;
    idx = y == 7.0 ? ( x <= 7.0 ? 49150.0 : 32598.0 ) : idx;
    idx = y == 6.0 ? ( x <= 7.0 ? 49150.0 : 32758.0 ) : idx;
    idx = y == 5.0 ? ( x <= 7.0 ? 65534.0 : 32757.0 ) : idx;
    idx = y == 4.0 ? ( x <= 7.0 ? 49150.0 : 32766.0 ) : idx;
    idx = y == 3.0 ? ( x <= 7.0 ? 49150.0 : 32758.0 ) : idx;
    idx = y == 2.0 ? ( x <= 7.0 ? 65502.0 : 30709.0 ) : idx;
    idx = y == 1.0 ? ( x <= 7.0 ? 65534.0 : 32767.0 ) : idx;
    idx = y == 0.0 ? 21845.0 : idx;

    idx = SPRITE_DEC( x, idx );

    color = idx == 1.0 ? RGB( 0,     0,   0 ) : color;
    color = idx == 2.0 ? RGB( 231,  90,  16 ) : color;
    color = idx == 3.0 ? mix( RGB( 255,  165, 66 ), RGB( 231,  90,  16 ), t ) : color;

    return color;
}

float3 SpriteMushroom(float3 color, float x, float y) {
    float idx = 0.0;
    idx = y == 15.0 ? ( x <= 7.0 ? 40960.0 : 10.0 ) : idx;
    idx = y == 14.0 ? ( x <= 7.0 ? 43008.0 : 22.0 ) : idx;
    idx = y == 13.0 ? ( x <= 7.0 ? 43520.0 : 85.0 ) : idx;
    idx = y == 12.0 ? ( x <= 7.0 ? 43648.0 : 341.0 ) : idx;
    idx = y == 11.0 ? ( x <= 7.0 ? 43680.0 : 2646.0 ) : idx;
    idx = y == 10.0 ? ( x <= 7.0 ? 42344.0 : 10922.0 ) : idx;
    idx = y == 9.0 ? ( x <= 7.0 ? 38232.0 : 10922.0 ) : idx;
    idx = y == 8.0 ? ( x <= 7.0 ? 38234.0 : 42410.0 ) : idx;
    idx = y == 7.0 ? ( x <= 7.0 ? 38234.0 : 38314.0 ) : idx;
    idx = y == 6.0 ? ( x <= 7.0 ? 42346.0 : 38570.0 ) : idx;
    idx = y == 5.0 ? 43690.0 : idx;
    idx = y == 4.0 ? ( x <= 7.0 ? 64856.0 : 9599.0 ) : idx;
    idx = y == 3.0 ? ( x <= 7.0 ? 65280.0 : 255.0 ) : idx;
    idx = y == 2.0 ? ( x <= 7.0 ? 65280.0 : 239.0 ) : idx;
    idx = y == 1.0 ? ( x <= 7.0 ? 65280.0 : 239.0 ) : idx;
    idx = y == 0.0 ? ( x <= 7.0 ? 64512.0 : 59.0 ) : idx;

    idx = SPRITE_DEC( x, idx );

    color = idx == 1.0 ? RGB( 181, 49,   33 ) : color;
    color = idx == 2.0 ? RGB( 230, 156,  33 ) : color;
    color = idx == 3.0 ? RGB( 255, 255, 255 ) : color;

    return color;
}

float3 SpriteGround(float3 color, float x, float y) {
    float idx = 0.0;
    idx = y == 15.0 ? ( x <= 7.0 ? 65534.0 : 49127.0 ) : idx;
    idx = y == 14.0 ? ( x <= 7.0 ? 43691.0 : 27318.0 ) : idx;
    idx = y == 13.0 ? ( x <= 7.0 ? 43691.0 : 27318.0 ) : idx;
    idx = y == 12.0 ? ( x <= 7.0 ? 43691.0 : 27318.0 ) : idx;
    idx = y == 11.0 ? ( x <= 7.0 ? 43691.0 : 27254.0 ) : idx;
    idx = y == 10.0 ? ( x <= 7.0 ? 43691.0 : 38246.0 ) : idx;
    idx = y == 9.0 ? ( x <= 7.0 ? 43691.0 : 32758.0 ) : idx;
    idx = y == 8.0 ? ( x <= 7.0 ? 43691.0 : 27318.0 ) : idx;
    idx = y == 7.0 ? ( x <= 7.0 ? 43691.0 : 27318.0 ) : idx;
    idx = y == 6.0 ? ( x <= 7.0 ? 43691.0 : 27318.0 ) : idx;
    idx = y == 5.0 ? ( x <= 7.0 ? 43685.0 : 27309.0 ) : idx;
    idx = y == 4.0 ? ( x <= 7.0 ? 43615.0 : 27309.0 ) : idx;
    idx = y == 3.0 ? ( x <= 7.0 ? 22011.0 : 27307.0 ) : idx;
    idx = y == 2.0 ? ( x <= 7.0 ? 32683.0 : 27307.0 ) : idx;
    idx = y == 1.0 ? ( x <= 7.0 ? 27307.0 : 23211.0 ) : idx;
    idx = y == 0.0 ? ( x <= 7.0 ? 38230.0 : 38231.0 ) : idx;

    idx = SPRITE_DEC( x, idx );

    color = RGB( 0, 0, 0 );
    color = idx == 2.0 ? RGB( 231,  90,  16 ) : color;
    color = idx == 3.0 ? RGB( 247, 214, 181 ) : color;

    return color;
}

float3 SpriteFlagpoleEnd(float3 color, float x, float y) {
    float idx = 0.0;

    idx = y == 7.0 ? 1360.0  : idx;
    idx = y == 6.0 ? 6836.0  : idx;
    idx = y == 5.0 ? 27309.0 : idx;
    idx = y == 4.0 ? 27309.0 : idx;
    idx = y == 3.0 ? 27305.0 : idx;
    idx = y == 2.0 ? 27305.0 : idx;
    idx = y == 1.0 ? 6820.0  : idx;
    idx = y == 0.0 ? 1360.0  : idx;

    idx = SPRITE_DEC( x, idx );

    color = idx == 1.0 ? RGB( 0,     0,  0 ) : color;
    color = idx == 2.0 ? RGB( 0,   173,  0 ) : color;
    color = idx == 3.0 ? RGB( 189, 255, 24 ) : color;

    return color;
}

float3 SpriteMario(float3 color, float x, float y, float frame) {
    float idx = 0.0;

    if ( frame == 0.0 )
    {
        idx = y == 14.0 ? ( x <= 7.0 ? 40960.0 : 42.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 43008.0 : 2730.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 21504.0 : 223.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 56576.0 : 4063.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 23808.0 : 16255.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 62720.0 : 1375.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 61440.0 : 1023.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 21504.0 : 793.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 22272.0 : 4053.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 23488.0 : 981.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 43328.0 : 170.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 43584.0 : 170.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 10832.0 : 42.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 16400.0 : 5.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 16384.0 : 21.0 ) : idx;
    }
    else if ( frame == 1.0 )
    {
        idx = y == 15.0 ? ( x <= 7.0 ? 43008.0 : 10.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 43520.0 : 682.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 54528.0 : 55.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 63296.0 : 1015.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 55104.0 : 4063.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 64832.0 : 343.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 64512.0 : 255.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 25856.0 : 5.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 38208.0 : 22.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 42304.0 : 235.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 38208.0 : 170.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 62848.0 : 171.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 62976.0 : 42.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 43008.0 : 21.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 21504.0 : 85.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 21504.0 : 1.0 ) : idx;
    }
    else if ( frame == 2.0 )
    {
        idx = y == 15.0 ? ( x <= 7.0 ? 43008.0 : 10.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 43520.0 : 682.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 54528.0 : 55.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 63296.0 : 1015.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 55104.0 : 4063.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 64832.0 : 343.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 64512.0 : 255.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 42320.0 : 5.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 42335.0 : 16214.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 58687.0 : 15722.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 43535.0 : 1066.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 43648.0 : 1450.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 43680.0 : 1450.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 2708.0 : 1448.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 84.0 : 0.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 336.0 : 0.0 ) : idx;
    }
    else if ( frame == 3.0 )
    {
        idx = y == 15.0 ? ( x <= 7.0 ? 0.0 : 64512.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 40960.0 : 64554.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 43008.0 : 64170.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 21504.0 : 21727.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 56576.0 : 22495.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 23808.0 : 32639.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 62720.0 : 5471.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 61440.0 : 2047.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 38224.0 : 405.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 21844.0 : 16982.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 21855.0 : 17066.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 39487.0 : 23470.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 43596.0 : 23210.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 43344.0 : 23210.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 43604.0 : 42.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 43524.0 : 0.0 ) : idx;
    }
    else if ( frame == 4.0 )
    {
        idx = y == 29.0 ? ( x <= 7.0 ? 32768.0 : 170.0 ) : idx;
        idx = y == 28.0 ? ( x <= 7.0 ? 43008.0 : 234.0 ) : idx;
        idx = y == 27.0 ? ( x <= 7.0 ? 43520.0 : 250.0 ) : idx;
        idx = y == 26.0 ? ( x <= 7.0 ? 43520.0 : 10922.0 ) : idx;
        idx = y == 25.0 ? ( x <= 7.0 ? 54528.0 : 1015.0 ) : idx;
        idx = y == 24.0 ? ( x <= 7.0 ? 57152.0 : 16343.0 ) : idx;
        idx = y == 23.0 ? ( x <= 7.0 ? 24384.0 : 65535.0 ) : idx;
        idx = y == 22.0 ? ( x <= 7.0 ? 24400.0 : 65407.0 ) : idx;
        idx = y == 21.0 ? ( x <= 7.0 ? 65360.0 : 5463.0 ) : idx;
        idx = y == 20.0 ? ( x <= 7.0 ? 64832.0 : 5471.0 ) : idx;
        idx = y == 19.0 ? ( x <= 7.0 ? 62464.0 : 4095.0 ) : idx;
        idx = y == 18.0 ? ( x <= 7.0 ? 43264.0 : 63.0 ) : idx;
        idx = y == 17.0 ? ( x <= 7.0 ? 22080.0 : 6.0 ) : idx;
        idx = y == 16.0 ? ( x <= 7.0 ? 22080.0 : 25.0 ) : idx;
        idx = y == 15.0 ? ( x <= 7.0 ? 22096.0 : 4005.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 22160.0 : 65365.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 23184.0 : 65365.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 23168.0 : 64853.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 27264.0 : 64853.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 43648.0 : 598.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 43648.0 : 682.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 43648.0 : 426.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 43605.0 : 2666.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 43605.0 : 2710.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 43605.0 : 681.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 10837.0 : 680.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 85.0 : 340.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 5.0 : 340.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 1.0 : 5460.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 0.0 : 5460.0 ) : idx;
    }
    else if ( frame == 5.0 )
    {
        idx = y == 30.0 ? ( x <= 7.0 ? 40960.0 : 42.0 ) : idx;
        idx = y == 29.0 ? ( x <= 7.0 ? 43520.0 : 58.0 ) : idx;
        idx = y == 28.0 ? ( x <= 7.0 ? 43648.0 : 62.0 ) : idx;
        idx = y == 27.0 ? ( x <= 7.0 ? 43648.0 : 2730.0 ) : idx;
        idx = y == 26.0 ? ( x <= 7.0 ? 62784.0 : 253.0 ) : idx;
        idx = y == 25.0 ? ( x <= 7.0 ? 63440.0 : 4085.0 ) : idx;
        idx = y == 24.0 ? ( x <= 7.0 ? 55248.0 : 16383.0 ) : idx;
        idx = y == 23.0 ? ( x <= 7.0 ? 55252.0 : 16351.0 ) : idx;
        idx = y == 22.0 ? ( x <= 7.0 ? 65492.0 : 1365.0 ) : idx;
        idx = y == 21.0 ? ( x <= 7.0 ? 65360.0 : 1367.0 ) : idx;
        idx = y == 20.0 ? ( x <= 7.0 ? 64832.0 : 1023.0 ) : idx;
        idx = y == 19.0 ? ( x <= 7.0 ? 43520.0 : 15.0 ) : idx;
        idx = y == 18.0 ? ( x <= 7.0 ? 38464.0 : 22.0 ) : idx;
        idx = y == 17.0 ? ( x <= 7.0 ? 21904.0 : 26.0 ) : idx;
        idx = y == 16.0 ? ( x <= 7.0 ? 21904.0 : 90.0 ) : idx;
        idx = y == 15.0 ? ( x <= 7.0 ? 21904.0 : 106.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 21904.0 : 125.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 21904.0 : 255.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 21920.0 : 767.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 22176.0 : 2815.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 23200.0 : 2751.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 43680.0 : 2725.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 43648.0 : 661.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 27136.0 : 341.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 23040.0 : 85.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 26624.0 : 21.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 41984.0 : 86.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 21504.0 : 81.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 21760.0 : 1.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 21760.0 : 21.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 20480.0 : 21.0 ) : idx;
    }
    else if ( frame == 6.0 )
    {
        idx = y == 31.0 ? ( x <= 7.0 ? 40960.0 : 42.0 ) : idx;
        idx = y == 30.0 ? ( x <= 7.0 ? 43520.0 : 58.0 ) : idx;
        idx = y == 29.0 ? ( x <= 7.0 ? 43648.0 : 62.0 ) : idx;
        idx = y == 28.0 ? ( x <= 7.0 ? 43648.0 : 2730.0 ) : idx;
        idx = y == 27.0 ? ( x <= 7.0 ? 62784.0 : 253.0 ) : idx;
        idx = y == 26.0 ? ( x <= 7.0 ? 63440.0 : 4085.0 ) : idx;
        idx = y == 25.0 ? ( x <= 7.0 ? 55248.0 : 16383.0 ) : idx;
        idx = y == 24.0 ? ( x <= 7.0 ? 55252.0 : 16351.0 ) : idx;
        idx = y == 23.0 ? ( x <= 7.0 ? 65492.0 : 1365.0 ) : idx;
        idx = y == 22.0 ? ( x <= 7.0 ? 65364.0 : 1367.0 ) : idx;
        idx = y == 21.0 ? ( x <= 7.0 ? 64832.0 : 1023.0 ) : idx;
        idx = y == 20.0 ? ( x <= 7.0 ? 21504.0 : 15.0 ) : idx;
        idx = y == 19.0 ? ( x <= 7.0 ? 43520.0 : 12325.0 ) : idx;
        idx = y == 18.0 ? ( x <= 7.0 ? 38208.0 : 64662.0 ) : idx;
        idx = y == 17.0 ? ( x <= 7.0 ? 21840.0 : 64922.0 ) : idx;
        idx = y == 16.0 ? ( x <= 7.0 ? 21844.0 : 65114.0 ) : idx;
        idx = y == 15.0 ? ( x <= 7.0 ? 21844.0 : 30298.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 38228.0 : 5722.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 42325.0 : 1902.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 43605.0 : 682.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 44031.0 : 682.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 44031.0 : 17066.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 43775.0 : 21162.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 43772.0 : 21866.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 43392.0 : 21866.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 42640.0 : 21866.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 23189.0 : 21866.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 43605.0 : 21824.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 2389.0 : 0.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 84.0 : 0.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 84.0 : 0.0 ) : idx;
        idx = y == 0.0 ? ( x <= 7.0 ? 336.0 : 0.0 ) : idx;
    }
    else
    {
        idx = y == 31.0 ? ( x <= 7.0 ? 0.0 : 16128.0 ) : idx;
        idx = y == 30.0 ? ( x <= 7.0 ? 0.0 : 63424.0 ) : idx;
        idx = y == 29.0 ? ( x <= 7.0 ? 40960.0 : 55274.0 ) : idx;
        idx = y == 28.0 ? ( x <= 7.0 ? 43520.0 : 65514.0 ) : idx;
        idx = y == 27.0 ? ( x <= 7.0 ? 43648.0 : 21866.0 ) : idx;
        idx = y == 26.0 ? ( x <= 7.0 ? 43648.0 : 23210.0 ) : idx;
        idx = y == 25.0 ? ( x <= 7.0 ? 62784.0 : 22013.0 ) : idx;
        idx = y == 24.0 ? ( x <= 7.0 ? 63440.0 : 24573.0 ) : idx;
        idx = y == 23.0 ? ( x <= 7.0 ? 55248.0 : 32767.0 ) : idx;
        idx = y == 22.0 ? ( x <= 7.0 ? 55248.0 : 32735.0 ) : idx;
        idx = y == 21.0 ? ( x <= 7.0 ? 65492.0 : 5461.0 ) : idx;
        idx = y == 20.0 ? ( x <= 7.0 ? 64852.0 : 7511.0 ) : idx;
        idx = y == 19.0 ? ( x <= 7.0 ? 64832.0 : 6143.0 ) : idx;
        idx = y == 18.0 ? ( x <= 7.0 ? 43520.0 : 5477.0 ) : idx;
        idx = y == 17.0 ? ( x <= 7.0 ? 38228.0 : 1382.0 ) : idx;
        idx = y == 16.0 ? ( x <= 7.0 ? 21845.0 : 1430.0 ) : idx;
        idx = y == 15.0 ? ( x <= 7.0 ? 21845.0 : 410.0 ) : idx;
        idx = y == 14.0 ? ( x <= 7.0 ? 22005.0 : 602.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 38909.0 : 874.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 43007.0 : 686.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 44031.0 : 682.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 43763.0 : 17066.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 43708.0 : 21162.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 43648.0 : 21930.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 43584.0 : 21930.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 42389.0 : 21930.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 23189.0 : 21930.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 43669.0 : 21920.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 43669.0 : 0.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 10901.0 : 0.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 5.0 : 0.0 ) : idx;
    }

    idx = SPRITE_DEC( x, idx );

    color = idx == 1.0 ? RGB( 106, 107,  4 ) : color;
    color = idx == 2.0 ? RGB( 177,  52, 37 ) : color;
    color = idx == 3.0 ? RGB( 227, 157, 37 ) : color;

    return color;
}

float3 SpriteCoin(float3 color, float x, float y, float frame) {
    float idx = 0.0;
    if ( frame == 0.0 )
    {
        idx = y == 14.0 ? ( x <= 7.0 ? 32768.0 : 1.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 32768.0 : 1.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 24576.0 : 5.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 24576.0 : 5.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 24576.0 : 5.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 24576.0 : 5.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 28672.0 : 5.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 28672.0 : 5.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 24576.0 : 5.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 24576.0 : 5.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 24576.0 : 5.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 24576.0 : 5.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 32768.0 : 1.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 32768.0 : 1.0 ) : idx;
    }
    else if ( frame == 1.0 )
    {
        idx = y == 14.0 ? ( x <= 7.0 ? 32768.0 : 2.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 40960.0 : 10.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 43008.0 : 42.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 59392.0 : 41.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 47616.0 : 166.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 47616.0 : 166.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 47616.0 : 166.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 47616.0 : 166.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 47616.0 : 166.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 47616.0 : 166.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 59392.0 : 41.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 43008.0 : 42.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 40960.0 : 10.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 32768.0 : 2.0 ) : idx;;
    }
    else if ( frame == 2.0 )
    {
        idx = y == 14.0 ? ( x <= 7.0 ? 49152.0 : 1.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 49152.0 : 1.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 61440.0 : 7.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 49152.0 : 1.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 49152.0 : 1.0 ) : idx;
    }
    else
    {
        idx = y == 14.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 13.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 12.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 11.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 10.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 9.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 8.0 ? ( x <= 7.0 ? 0.0 : 3.0 ) : idx;
        idx = y == 7.0 ? ( x <= 7.0 ? 0.0 : 3.0 ) : idx;
        idx = y == 6.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 5.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 4.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 3.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 2.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
        idx = y == 1.0 ? ( x <= 7.0 ? 0.0 : 2.0 ) : idx;
    }

    idx = SPRITE_DEC( x, idx );

    color = idx == 1.0 ? RGB( 181, 49,   33 ) : color;
    color = idx == 2.0 ? RGB( 230, 156,  33 ) : color;
    color = idx == 3.0 ? RGB( 255, 255, 255 ) : color;

    return color;
}

float3 SpriteBrick(float3 color, float x, float y) {
    float ymod4 = floor( mod( y, 4.0 ) );
    float xmod8 = floor( mod( x, 8.0 ) );
    float ymod8 = floor( mod( y, 8.0 ) );

    // dark orange
    float idx = 2.0;

    // black
    idx = ymod4 == 0.0 ? 1.0 : idx;
    idx = xmod8 == ( ymod8 < 4.0 ? 3.0 : 7.0 ) ? 1.0 : idx;

    // light orange
    idx = y == 15.0 ? 3.0 : idx;

    color = idx == 1.0 ? RGB( 0,     0,   0 ) : color;
    color = idx == 2.0 ? RGB( 231,  90,  16 ) : color;
    color = idx == 3.0 ? RGB( 247, 214, 181 ) : color;

    return color;
}

float3 DrawCastle(float3 color, float x, float y) {
    if ( x >= 0.0 && x < 80.0 && y >= 0.0 && y < 80.0 )
    {
        float ymod4    = mod( y, 4.0 );
        float xmod8    = mod( x, 8.0 );
        float xmod16_4 = mod( x + 4.0, 16.0 );
        float xmod16_3 = mod( x + 5.0, 16.0 );
        float ymod8    = mod( y, 8.0 );

        // dark orange
        float idx = 2.0;

        // black
        idx = ymod4 == 0.0 && y <= 72.0 && ( y != 44.0 || xmod16_3 > 8.0 ) ? 1.0 : idx;
        idx = x >= 24.0 && x <= 32.0 && y >= 48.0 && y <= 64.0 ? 1.0 : idx;
        idx = x >= 48.0 && x <= 56.0 && y >= 48.0 && y <= 64.0 ? 1.0 : idx;
        idx = x >= 32.0 && x <= 47.0 && y <= 25.0 ? 1.0 : idx;
        idx = xmod8 == ( ymod8 < 4.0 ? 3.0 : 7.0 ) && y <= 72.0 && ( xmod16_3 > 8.0 || y <= 40.0 || y >= 48.0 ) ? 1.0 : idx;

        // white
        idx = y == ( xmod16_4 < 8.0 ? 47.0 : 40.0 ) ? 3.0 : idx;
        idx = y == ( xmod16_4 < 8.0 ? 79.0 : 72.0 ) ? 3.0 : idx;
        idx = xmod8 == 3.0 && y >= 40.0 && y <= 47.0 ? 3.0 : idx;
        idx = xmod8 == 3.0 && y >= 72.0 ? 3.0 : idx;

        // transparent
        idx = ( x < 16.0 || x >= 64.0 ) && y >= 48.0 ? 0.0 : idx;
        idx = x >= 4.0  && x <= 10.0 && y >= 41.0 && y <= 47.0 ? 0.0 : idx;
        idx = x >= 68.0 && x <= 74.0 && y >= 41.0 && y <= 47.0 ? 0.0 : idx;
        idx = y >= 73.0 && xmod16_3 > 8.0 ? 0.0 : idx;

        color = idx == 1.0 ? RGB(   0,   0,   0 ) : color;
        color = idx == 2.0 ? RGB( 231,  90,  16 ) : color;
        color = idx == 3.0 ? RGB( 247, 214, 181 ) : color;
    }

    return color;
}

float3 DrawKoopa(float3 color, float x, float y, float frame) {
    if (x >= 0.0 && x <= 15.0) {
        color = SpriteKoopa(color, x, y, frame);
    }
    return color;
}

float3 KoopaWalk(float3 color, float worldX, float worldY, float time, float frame, float startX) {
    float x = worldX - startX + floor( time * GOOMBA_SPEED);
    color = DrawKoopa(color, x, worldY - 16.0, frame);
    return color;
}

float3 DrawHitQuestion(float3 color, float questionX, float questionY, float time, float questionT, float questionHitTime) {
    float t = clamp( ( time - questionHitTime ) / 0.25, 0.0, 1.0 );
    t = 1.0 - abs( 2.0 * t - 1.0 );

    questionY -= floor( t * 8.0 );
    if (questionX >= 0.0 && questionX <= 15.0) {
        if (time >= questionHitTime) {
            color = SpriteQuestion(color, questionX, questionY, 1.0);
            if (questionX >= 3.0 && questionX <= 12.0 && questionY >= 1.0 && questionY <= 15.0) {
                color = RGB(231, 90, 16);
            }
        } else {
            color = SpriteQuestion(color, questionX, questionY, questionT);
        }
    }
    return color;
}

float3 DrawW(float3 color, float x, float y) {
    if (x >= 0.0 && x < 14.0 && y >= 0.0 && y < 14.0) {
        if ((    x <= 3.0 || x >= 10.0)
            || ( x >= 4.0 && x <= 5.0 && y >= 2.0 && y <= 7.0 )
            || ( x >= 8.0 && x <= 9.0 && y >= 2.0 && y <= 7.0 )
            || ( x >= 6.0 && x <= 7.0 && y >= 4.0 && y <= 9.0 )
            )
        {
            color = RGB(255, 255, 255);
        }
    }
    return color;
}

float3 DrawO(float3 color, float x, float y) {
    if (x >= 0.0 && x < 14.0 && y >= 0.0 && y < 14.0) {
        if (   (( x <= 1.0 || x >= 12.0 ) && ( y >= 2.0 && y <= 11.0 ))
            || ( x >= 2.0 && x <= 4.0 )
            || ( x >= 9.0 && x <= 11.0 )
            || (( y <= 1.0 || y >= 11.0 ) && ( x >= 2.0 && x <= 11.0 ))
            )
        {
            color = RGB( 255, 255, 255 );
        }
    }
    return color;
}

float3 DrawR(float3 color, float x, float y) {
    if (x >= 0.0 && x < 14.0 && y >= 0.0 && y < 14.0) {
        if (   ( x <= 3.0 )
            || ( y >= 12.0 && x <= 11.0 )
            || ( x >= 10.0 && y >= 6.0 && y <= 11.0 )
            || ( x >= 8.0  && x <= 9.0 && y <= 7.0 )
            || ( x <= 9.0  && y >= 4.0 && y <= 5.0 )
            || ( x >= 8.0  && y <= 1.0 )
            || ( x >= 6.0  && x <= 11.0 && y >= 2.0 && y <= 3.0 )
            )
        {
            color = RGB( 255, 255, 255 );
        }
    }
    return color;
}

float3 DrawL(float3 color, float x, float y) {
    if (x >= 0.0 && x < 14.0 && y >= 0.0 && y < 14.0) {
        if (x <= 3.0 || y <= 1.0)
        {
            color = RGB( 255, 255, 255 );
        }
    }
    return color;
}

float3 DrawD(float3 color, float x, float y) {
    if (x >= 0.0 && x < 14.0 && y >= 0.0 && y < 14.0) {
        color = RGB( 255, 255, 255 );
        if (   ( x >= 4.0 && x <= 7.0 && y >= 2.0 && y <= 11.0 )
            || ( x >= 8.0 && x <= 9.0 && y >= 4.0 && y <= 9.0 )
            || ( x >= 12.0 && ( y <= 3.0 || y >= 10.0 ) )
            || ( x >= 10.0 && ( y <= 1.0 || y >= 12.0 ) )
            )
        {
            color = RGB( 0, 0, 0 );
        }
    }
    return color;
}

float3 Draw1(float3 color, float x, float y) {
    if (x >= 0.0 && x < 14.0 && y >= 0.0 && y < 14.0) {
        if (   ( y <= 1.0 )
            || ( x >= 5.0 && x <= 8.0 )
            || ( x >= 3.0 && x <= 4.0 && y >= 10.0 && y <= 11.0 )
            )
        {
            color = RGB( 255, 255, 255 );
        }
    }
    return color;
}

float3 DrawM(float3 color, float x, float y) {
    if (x >= 0.0 && x < 14.0 && y >= 0.0 && y < 14.0) {
        if (y >= 4.0 && y <= 7.0) {
            color = RGB( 255, 255, 255 );
        }
    }
    return color;
}

float3 DrawIntro(float3 color, float x, float y, float screenWidth, float screenHeight) {
//    color = RGB(0, 0, 0);
//
//    float offset    = 18.0;
//    float textX     = floor( x + (screenWidth - offset * 8.0 - 7.0) / 2.0);
//    float textY     = floor( y + (screenHeight - 7.0 ) / 2.0 - 16.0 * 2.0);
//    float marioX    = textX - offset * 4.0;
//    float marioY    = textY + 16.0 * 3.0;
//
//    color = DrawW(color, textX + offset * 0.0, textY);
//    color = DrawO(color, textX + offset * 1.0, textY);
//    color = DrawR(color, textX + offset * 2.0, textY);
//    color = DrawL(color, textX + offset * 3.0, textY);
//    color = DrawD(color, textX + offset * 4.0, textY);
//    color = Draw1(color, textX + offset * 6.0, textY);
//    color = DrawM(color, textX + offset * 7.0, textY);
//    color = Draw1(color, textX + offset * 8.0, textY);
//
//    if (marioX >= 0.0 && marioX <= 15.0) {
//        color = SpriteMario(color, marioX, marioY, 4.0);
//    }
//
//    return color;

    color = RGB( 0, 0, 0 );

    float offset     = 18.0;
    float textX     = floor( x - ( screenWidth - offset * 8.0 - 7.0 ) / 2.0 );
    float textY     = floor( y - ( screenHeight - 7.0 ) / 2.0 - 16.0 * 2.0 );
    float marioX    = textX - offset * 4.0;
    float marioY    = textY + 16.0 * 3.0;

    color = DrawW( color, textX - offset * 0.0, textY );
    color = DrawO( color, textX - offset * 1.0, textY );
    color = DrawR( color, textX - offset * 2.0, textY );
    color = DrawL( color, textX - offset * 3.0, textY );
    color = DrawD( color, textX - offset * 4.0, textY );
    color = Draw1( color, textX - offset * 6.0, textY );
    color = DrawM( color, textX - offset * 7.0, textY );
    color = Draw1( color, textX - offset * 8.0, textY );

    if (marioX >= 0.0 && marioX <= 15.0) {
        color = SpriteMario( color, marioX, marioY, 4.0 );
    }

    return color;
}

float CoinAnimY(float worldY, float time, float coinTime) {
    return worldY - 4.0 * 16.0 - floor( 64.0 * ( 1.0 - abs( 2.0 * ( clamp( ( time - coinTime ) / 0.8, 0.0, 1.0 ) ) - 1.0 ) ) );
}

float QuestionAnimY(float worldY, float time, float questionHitTime) {
    return worldY - 4.0 * 16.0 - floor( 8.0 * ( 1.0 - abs( 2.0 * clamp( ( time - questionHitTime ) / 0.25, 0.0, 1.0 ) - 1.0 ) ) );
}

float GoombaSWalkX(float worldX, float startX, float time, float goombaLifeTime) {
    return worldX + floor( min( time, goombaLifeTime ) * GOOMBA_SPEED ) - startX;
}

float3 DrawGame(float3 color, float time, float pixelX, float pixelY, float screenWidth, float screenHeight) {
    float mushroomPauseStart = 16.25;
    float mushroomPauseLength = 2.0;
    float flagPauseStart = 38.95;
    float flagPauseLength = 1.5;

    float cameraP1        = clamp( time - mushroomPauseStart, 0.0, mushroomPauseLength );
    float cameraP2        = clamp( time - flagPauseStart,     0.0, flagPauseLength );
    float cameraX         = floor( min( ( time - cameraP1 - cameraP2 ) * MARIO_SPEED - 240.0, 3152.0 ) );
    float worldX         = pixelX + cameraX;
    float worldY          = pixelY - 8.0;
    float tileX            = floor( worldX / 16.0 );
    float tileY            = floor( worldY / 16.0 );
    float worldXMod16    = mod( worldX, 16.0 );
    float worldYMod16     = mod( worldY, 16.0 );

    // default background color
    color = RGB( 92, 148, 252 );

    // draw hills
    float bigHillX   = mod( worldX, 768.0 );
    float smallHillX = mod( worldX - 240.0, 768.0 );
    float hillX      = min( bigHillX, smallHillX );
    float hillY      = worldY - ( smallHillX < bigHillX ? 0.0 : 16.0 );
    color = SpriteHill(color, hillX, hillY);

    // draw clouds and bushes
    float sc1CloudX = mod( worldX - 296.0, 768.0 );
    float sc2CloudX = mod( worldX - 904.0, 768.0 );
    float mcCloudX  = mod( worldX - 584.0, 768.0 );
    float lcCloudX  = mod( worldX - 440.0, 768.0 );
    float scCloudX  = min( sc1CloudX, sc2CloudX );
    float sbCloudX     = mod( worldX - 376.0, 768.0 );
    float mbCloudX  = mod( worldX - 664.0, 768.0 );
    float lbCloudX  = mod( worldX - 184.0, 768.0 );
    float cCloudX    = min( min( scCloudX, mcCloudX ), lcCloudX );
    float bCloudX    = min( min( sbCloudX, mbCloudX ), lbCloudX );
    float sCloudX    = min( scCloudX, sbCloudX );
    float mCloudX    = min( mcCloudX, mbCloudX );
    float cloudX    = min( cCloudX, bCloudX );
    float isBush    = bCloudX < cCloudX ? 1.0 : 0.0;
    float cloudSeg    = cloudX == sCloudX ? 0.0 : ( cloudX == mCloudX ? 1.0 : 2.0 );
    float cloudY    = worldY - ( isBush == 1.0 ? 8.0 : ( ( cloudSeg == 0.0 && sc1CloudX < sc2CloudX ) || cloudSeg == 1.0 ? 168.0 : 152.0 ) );
    if (cloudX >= 0.0 && cloudX < 32.0 + 16.0 * cloudSeg) {
        if (cloudSeg == 1.0 ) {
            cloudX = cloudX < 24.0 ? cloudX : cloudX - 16.0;
        }
        if (cloudSeg == 2.0) {
            cloudX = cloudX < 24.0 ? cloudX : ( cloudX < 40.0 ? cloudX - 16.0 : cloudX - 32.0 );
        }
        color = SpriteCloud(color, cloudX, cloudY, isBush);
    }

    // draw flag pole
    if (worldX >= 3175.0 && worldX <= 3176.0 && worldY <= 176.0) {
        color = RGB( 189, 255, 24 );
    }

    // draw flag
    float flagX = worldX - 3160.0;
    float flagY = worldY - 159.0 + floor( 122.0 * clamp( ( time - 39.0 ) / 1.0, 0.0, 1.0 ) );
    if (flagX >= 0.0 && flagX <= 15.0) {
        color = SpriteFlag(color, flagX, flagY);
    }

    // draw flagpole end
    float flagpoleEndX = worldX - 3172.0;
    float flagpoleEndY = worldY - 176.0;
    if (flagpoleEndX >= 0.0 && flagpoleEndX <= 7.0) {
        color = SpriteFlagpoleEnd(color, flagpoleEndX, flagpoleEndY);
    }

    // draw blocks
    if (   ( tileX >= 134.0 && tileX < 138.0 && tileX - 132.0 > tileY )
        || ( tileX >= 140.0 && tileX < 144.0 && 145.0 - tileX > tileY )
        || ( tileX >= 148.0 && tileX < 153.0 && tileX - 146.0 > tileY && tileY < 5.0 )
        || ( tileX >= 155.0 && tileX < 159.0 && 160.0 - tileX > tileY )
        || ( tileX >= 181.0 && tileX < 190.0 && tileX - 179.0 > tileY && tileY < 9.0 )
        || ( tileX == 198.0 && tileY == 1.0 )
        )
    {
        color = SpriteBlock(color, worldXMod16, worldYMod16);
    }

    // draw pipes
    float pipeY = worldY - 16.0;
    float pipeH    = 0.0;
    float pipeX = worldX - 179.0 * 16.0;
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 163.0 * 16.0;
        pipeH = 0.0;
    }
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 57.0 * 16.0;
        pipeH = 2.0;
    }
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 46.0 * 16.0;
        pipeH = 2.0;
    }
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 38.0 * 16.0;
        pipeH = 1.0;
    }
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 28.0 * 16.0;
        pipeH = 0.0;
    }
    if (pipeX >= 0.0 && pipeX <= 31.0 && pipeY >= 0.0 && pipeY <= 31.0 + pipeH * 16.0) {
        color = SpritePipe(color, pipeX, pipeY, pipeH);
    }

    // draw mushroom
    float mushroomStart = 15.7;
    if (time >= mushroomStart && time <= 17.0) {
        float mushroomX = worldX - 1248.0;
        float mushroomY = worldY - 4.0 * 16.0;
        if ( time >= mushroomStart )
        {
            mushroomY = worldY - 4.0 * 16.0 - floor( 16.0 * clamp( ( time - mushroomStart ) / 0.5, 0.0, 1.0 ) );
        }
        if ( time >= mushroomStart + 0.5 )
        {
            mushroomX -= floor( MARIO_SPEED * ( time - mushroomStart - 0.5 ) );
        }
        if ( time >= mushroomStart + 0.5 + 0.4 )
        {
            mushroomY = mushroomY + floor( sin( ( ( time - mushroomStart - 0.5 - 0.4 ) ) * 3.14 ) * 4.0 * 16.0 );
        }

        if (mushroomX >= 0.0 && mushroomX <= 15.0) {
            color = SpriteMushroom(color, mushroomX, mushroomY);
        }
    }

    // draw coins
    float coinFrame = floor( mod( time * 12.0, 4.0 ) );
    float coinX = worldX - 2720.0;
    float coinTime = 33.9;
    float coinY = CoinAnimY( worldY, time, coinTime );
    if ( coinX < 0.0 )
    {
        coinX         = worldX - 1696.0;
        coinTime     = 22.4;
        coinY         = CoinAnimY( worldY, time, coinTime );
    }
    if ( coinX < 0.0 )
    {
        coinX         = worldX - 352.0;
        coinTime     = 5.4;
        coinY         = CoinAnimY( worldY, time, coinTime );
    }

    if (coinX >= 0.0 && coinX <= 15.0 && time >= coinTime + 0.1) {
        color = SpriteCoin(color, coinX, coinY, coinFrame);
    }

    // draw questions
    float questionT = clamp( sin( time * 6.0 ), 0.0, 1.0 );
    if (    ( tileY == 4.0 && ( tileX == 16.0 || tileX == 20.0 || tileX == 109.0 || tileX == 112.0 ) )
        || ( tileY == 8.0 && ( tileX == 21.0 || tileX == 94.0 || tileX == 109.0 ) )
        || ( tileY == 8.0 && ( tileX >= 129.0 && tileX <= 130.0 ) )
        )
    {
        color = SpriteQuestion(color, worldXMod16, worldYMod16, questionT);
    }

    // draw hitted questions
    float questionHitTime     = 33.9;
    float questionX         = worldX - 2720.0;
    if ( questionX < 0.0 )
    {
        questionHitTime = 22.4;
        questionX        = worldX - 1696.0;
    }
    if ( questionX < 0.0 )
    {
        questionHitTime = 15.4;
        questionX        = worldX - 1248.0;
    }
    if ( questionX < 0.0 )
    {
        questionHitTime = 5.3;
        questionX        = worldX - 352.0;
    }
    questionT        = time >= questionHitTime ? 1.0 : questionT;
    float questionY = QuestionAnimY( worldY, time, questionHitTime );
    if (questionX >= 0.0 && questionX <= 15.0) {
        color = SpriteQuestion(color, questionX, questionY, questionT);
    }
    if (time >= questionHitTime && questionX >= 3.0 && questionX <= 12.0 && questionY >= 1.0 && questionY <= 15.0) {
        color = RGB( 231, 90, 16 );
    }

    // draw bricks
    if (   ( tileY == 4.0 && ( tileX == 19.0 || tileX == 21.0 || tileX == 23.0 || tileX == 77.0 || tileX == 79.0 || tileX == 94.0 || tileX == 118.0 || tileX == 168.0 || tileX == 169.0 || tileX == 171.0 ) )
        || ( tileY == 8.0 && ( tileX == 128.0 || tileX == 131.0 ) )
        || ( tileY == 8.0 && ( tileX >= 80.0 && tileX <= 87.0 ) )
        || ( tileY == 8.0 && ( tileX >= 91.0 && tileX <= 93.0 ) )
        || ( tileY == 4.0 && ( tileX >= 100.0 && tileX <= 101.0 ) )
        || ( tileY == 8.0 && ( tileX >= 121.0 && tileX <= 123.0 ) )
        || ( tileY == 4.0 && ( tileX >= 129.0 && tileX <= 130.0 ) )
        )
    {
        color = SpriteBrick(color, worldXMod16, worldYMod16);
    }

    // draw castle flag
    float castleFlagX = worldX - 3264.0;
    float castleFlagY = worldY - 64.0 - floor( 32.0 * clamp( ( time - 44.6 ) / 1.0, 0.0, 1.0 ) );
    if (castleFlagX > 0.0 && castleFlagX < 14.0) {
        color = SpriteCastleFlag(color, castleFlagX, castleFlagY);
    }

    color = DrawCastle(color, worldX - 3232.0, worldY - 16.0);

    // draw ground
    if ( tileY <= 0.0
        && !( tileX >= 69.0  && tileX < 71.0 )
        && !( tileX >= 86.0  && tileX < 89.0 )
        && !( tileX >= 153.0 && tileX < 155.0 )
        )
    {
        color = SpriteGround(color, worldXMod16, worldYMod16);
    }

    // draw Koopa
    float goombaFrame = floor( mod( time * 5.0, 2.0 ) );
    color = KoopaWalk(color, worldX, worldY, time, goombaFrame, 2370.0);

    // draw stomped walking Goombas
    float goombaY             = worldY - 16.0;
    float goombaLifeTime     = 26.3;
    float goombaX = GoombaSWalkX( worldX, 2850.0 + 24.0, time, goombaLifeTime );
    if ( goombaX < 0.0 )
    {
        goombaLifeTime     = 25.3;
        goombaX = GoombaSWalkX( worldX, 2760.0, time, goombaLifeTime );
    }
    if ( goombaX < 0.0 )
    {
        goombaLifeTime     = 23.5;
        goombaX = GoombaSWalkX( worldX, 2540.0, time, goombaLifeTime );
    }
    if ( goombaX < 0.0 )
    {
        goombaLifeTime     = 20.29;
        goombaX = GoombaSWalkX( worldX, 2150.0, time, goombaLifeTime );
    }
    if ( goombaX < 0.0 )
    {
        goombaLifeTime     = 10.3;
        goombaX = worldX - 790.0 - floor( abs( mod( ( min( time, goombaLifeTime ) + 6.3 ) * GOOMBA_SPEED, 2.0 * 108.0 ) - 108.0 ) );
    }
    goombaFrame = time > goombaLifeTime ? 2.0 : goombaFrame;
    if ( goombaX >= 0.0 && goombaX <= 15.0 )
    {
        color = SpriteGoomba(color, goombaX, goombaY, goombaFrame);
    }

    // draw walking Goombas
    goombaFrame         = floor( mod( time * 5.0, 2.0 ) );
    float goombaWalkX     = worldX + floor( time * GOOMBA_SPEED );
    goombaX             = goombaWalkX - 3850.0 - 24.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 3850.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 2850.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 2760.0 - 24.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 2540.0 - 24.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 2150.0 - 24.0;
    if ( goombaX < 0.0 ) goombaX = worldX - 766.0 - floor( abs( mod( ( time + 6.3 ) * GOOMBA_SPEED, 2.0 * 108.0 ) - 108.0 ) );
    if ( goombaX < 0.0 ) goombaX = worldX - 638.0 - floor( abs( mod( ( time + 6.6 ) * GOOMBA_SPEED, 2.0 * 84.0 ) - 84.0 ) );
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 435.0;
    if ( goombaX >= 0.0 && goombaX <= 15.0 )
    {
        color = SpriteGoomba(color, goombaX, goombaY, goombaFrame);
    }

    // Mario jump
    float marioBigJump1     = 27.1;
    float marioBigJump2     = 29.75;
    float marioBigJump3     = 35.05;
    float marioJumpTime     = 0.0;
    float marioJumpScale    = 0.0;

    if ( time >= 4.2   ) { marioJumpTime = 4.2;   marioJumpScale = 0.45; }
    if ( time >= 5.0   ) { marioJumpTime = 5.0;   marioJumpScale = 0.5;  }
    if ( time >= 6.05  ) { marioJumpTime = 6.05;  marioJumpScale = 0.7;  }
    if ( time >= 7.8   ) { marioJumpTime = 7.8;   marioJumpScale = 0.8;  }
    if ( time >= 9.0   ) { marioJumpTime = 9.0;   marioJumpScale = 1.0;  }
    if ( time >= 10.3  ) { marioJumpTime = 10.3;  marioJumpScale = 0.3;  }
    if ( time >= 11.05 ) { marioJumpTime = 11.05; marioJumpScale = 1.0;  }
    if ( time >= 13.62 ) { marioJumpTime = 13.62; marioJumpScale = 0.45; }
    if ( time >= 15.1  ) { marioJumpTime = 15.1;  marioJumpScale = 0.5;  }
    if ( time >= 18.7  ) { marioJumpTime = 18.7;  marioJumpScale = 0.6;  }
    if ( time >= 19.65 ) { marioJumpTime = 19.65; marioJumpScale = 0.45; }
    if ( time >= 20.29 ) { marioJumpTime = 20.29; marioJumpScale = 0.3;  }
    if ( time >= 21.8  ) { marioJumpTime = 21.8;  marioJumpScale = 0.35; }
    if ( time >= 22.3  ) { marioJumpTime = 22.3;  marioJumpScale = 0.35; }
    if ( time >= 23.0  ) { marioJumpTime = 23.0;  marioJumpScale = 0.40; }
    if ( time >= 23.5  ) { marioJumpTime = 23.5;  marioJumpScale = 0.3;  }
    if ( time >= 24.7  ) { marioJumpTime = 24.7;  marioJumpScale = 0.45; }
    if ( time >= 25.3  ) { marioJumpTime = 25.3;  marioJumpScale = 0.3;  }
    if ( time >= 25.75 ) { marioJumpTime = 25.75; marioJumpScale = 0.4;  }
    if ( time >= 26.3  ) { marioJumpTime = 26.3;  marioJumpScale = 0.25; }
    if ( time >= marioBigJump1 )         { marioJumpTime = marioBigJump1;         marioJumpScale = 1.0; }
    if ( time >= marioBigJump1 + 1.0 )     { marioJumpTime = marioBigJump1 + 1.0;     marioJumpScale = 0.6; }
    if ( time >= marioBigJump2 )         { marioJumpTime = marioBigJump2;         marioJumpScale = 1.0; }
    if ( time >= marioBigJump2 + 1.0 )     { marioJumpTime = marioBigJump2 + 1.0;    marioJumpScale = 0.6; }
    if ( time >= 32.3  ) { marioJumpTime = 32.3;  marioJumpScale = 0.7;  }
    if ( time >= 33.7  ) { marioJumpTime = 33.7;  marioJumpScale = 0.3;  }
    if ( time >= 34.15 ) { marioJumpTime = 34.15; marioJumpScale = 0.45; }
    if ( time >= marioBigJump3 )                 { marioJumpTime = marioBigJump3;                 marioJumpScale = 1.0; }
    if ( time >= marioBigJump3 + 1.2 )             { marioJumpTime = marioBigJump3 + 1.2;             marioJumpScale = 0.89; }
    if ( time >= marioBigJump3 + 1.2 + 0.75 )     { marioJumpTime = marioBigJump3 + 1.2 + 0.75;     marioJumpScale = 0.5; }

    float marioJumpOffset         = 0.0;
    float marioJumpLength         = 1.5  * marioJumpScale;
    float marioJumpAmplitude    = 76.0 * marioJumpScale;
    if ( time >= marioJumpTime && time <= marioJumpTime + marioJumpLength )
    {
        float t = ( time - marioJumpTime ) / marioJumpLength;
        marioJumpOffset = floor( sin( t * 3.14 ) * marioJumpAmplitude );
    }

    // Mario land
    float marioLandTime     = 0.0;
    float marioLandAplitude = 0.0;
    if ( time >= marioBigJump1 + 1.0 + 0.45 )             { marioLandTime = marioBigJump1 + 1.0 + 0.45;             marioLandAplitude = 109.0; }
    if ( time >= marioBigJump2 + 1.0 + 0.45 )             { marioLandTime = marioBigJump2 + 1.0 + 0.45;             marioLandAplitude = 109.0; }
    if ( time >= marioBigJump3 + 1.2 + 0.75 + 0.375 )     { marioLandTime = marioBigJump3 + 1.2 + 0.75 + 0.375;     marioLandAplitude = 150.0; }

    float marioLandLength = marioLandAplitude / 120.0;
    if ( time >= marioLandTime && time <= marioLandTime + marioLandLength )
    {
        float t = 0.5 * ( time - marioLandTime ) / marioLandLength + 0.5;
        marioJumpOffset = floor( sin( t * 3.14 ) * marioLandAplitude );
    }

    // Mario flag jump
    marioJumpTime         = flagPauseStart - 0.3;
    marioJumpLength     = 1.5  * 0.45;
    marioJumpAmplitude    = 76.0 * 0.45;
    if ( time >= marioJumpTime && time <= marioJumpTime + marioJumpLength + flagPauseLength )
    {
        float time2 = time;
        if ( time >= flagPauseStart && time <= flagPauseStart + flagPauseLength )
        {
            time2 = flagPauseStart;
        }
        else if ( time >= flagPauseStart )
        {
            time2 = time - flagPauseLength;
        }
        float t = ( time2 - marioJumpTime ) / marioJumpLength;
        marioJumpOffset = floor( sin( t * 3.14 ) * marioJumpAmplitude );
    }

    // Mario base (ground offset)
    float marioBase = 0.0;
    if ( time >= marioBigJump1 + 1.0 && time < marioBigJump1 + 1.0 + 0.45 )
    {
        marioBase = 16.0 * 4.0;
    }
    if ( time >= marioBigJump2 + 1.0 && time < marioBigJump2 + 1.0 + 0.45 )
    {
        marioBase = 16.0 * 4.0;
    }
    if ( time >= marioBigJump3 + 1.2 && time < marioBigJump3 + 1.2 + 0.75 )
    {
        marioBase = 16.0 * 3.0;
    }
    if ( time >= marioBigJump3 + 1.2 + 0.75 && time < marioBigJump3 + 1.2 + 0.75 + 0.375 )
    {
        marioBase = 16.0 * 7.0;
    }

    float marioX        = pixelX - 112.0;
    float marioY        = pixelY - 16.0 - 8.0 - marioBase - marioJumpOffset;
    float marioFrame     = marioJumpOffset == 0.0 ? floor( mod( time * 10.0, 3.0 ) ) : 3.0;
    if ( time >= mushroomPauseStart && time <= mushroomPauseStart + mushroomPauseLength )
    {
        marioFrame = 1.0;
    }
    if ( time > mushroomPauseStart + 0.7 )
    {
        float t = time - mushroomPauseStart - 0.7;
        if ( mod( t, 0.2 ) <= mix( 0.0, 0.2, clamp( t / 1.3, 0.0, 1.0 ) ) )
        {
            // super mario offset
            marioFrame += 4.0;
        }
    }
    if (marioX >= 0.0 && marioX <= 15.0 && cameraX < 3152.0) {
        color = SpriteMario(color, marioX, marioY, marioFrame);
    }

    return color;
}

float2 CRTCurveUV(float2 uv) {
    uv = uv * 2.0 - 1.0;
    float2 offset = abs( uv.yx ) / float2( 6.0, 4.0 );
    uv = uv + uv * offset * offset;
    uv = uv * 0.5 + 0.5;
    return uv;
}

float3 DrawVignette(float3 color, float2 uv) {
    float vignette = uv.x * uv.y * ( 1.0 - uv.x ) * ( 1.0 - uv.y );
    vignette = clamp( pow( 16.0 * vignette, 0.3 ), 0.0, 1.0 );
    color *= vignette;
    return color;
}

float3 DrawScanline(float3 color, float2 uv, float time) {
    float scanline = clamp(0.95 + 0.05 * cos(3.14 * (uv.y + 0.008 * time ) * 240.0 * 1.0), 0.0, 1.0);
    float grille = 0.85 + 0.15 * clamp(1.5 * cos(3.14 * uv.x * 640.0 * 1.0), 0.0, 1.0);
    color *= scanline * grille * 1.2;
    return color;
}

fragment float4 shader_day97(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    // we want to see at least 224x192 (overscan) and we want multiples of pixel size
    float resMultX = floor(res.x / 224.0);
    float resMultY = floor(res.y / 192.0);
    float resRcp = 1.0 / max(min(resMultX, resMultY), 1.0);

    float screenWidth = floor(res.x * resRcp);
    float screenHeight = floor(res.y * resRcp);
    float pixelX = floor(pixPos.x * resRcp);// * -1.0;
    float pixelY = floor(pixPos.y * resRcp);// * -1.0;

    float3 color = RGB(92, 148, 252);
    color = DrawGame(color, time - 6.0, pixelX, pixelY, screenWidth, screenHeight);
    if (time < 6.0) {
        color = DrawIntro(color, pixelX, pixelY, screenWidth, screenHeight);
    }

    // CRT effects (curvature, vignette, scanlines and CRT grille)
    float2 uv = pixPos.xy / res.xy;
    float2 crtUV = CRTCurveUV(uv);
    if (crtUV.x < 0.0 || crtUV.x > 1.0 || crtUV.y < 0.0 || crtUV.y > 1.0) {
        color = float3(0.0, 0.0, 0.0);
    }
    color = DrawVignette(color, crtUV);
    color = DrawScanline(color, uv, time);

    return float4(color, 1.0);
}

// MARK: - Day98a

// https://www.shadertoy.com/view/3dlcWl torus

// helpers
float hash98a(float s) { return fract(sin(s) * 42422.42); }
float2x2 rot98(float v) { float a = cos(v), b = sin(v); return float2x2(a, b, -b, a); }
float torus(float3 p, float2 q) { return length(float2(length(p.xz) - q.x, p.y)) - q.y; }

struct MapResult {
    float value;
    float3 oGlow;
};

MapResult map98(float3 p, float time, float id, float3 glow) {
    p.xy = p.xy * rot(time * 0.1);
    p.xz = p.xz * rot(time * 0.2);

    float d = length(p);
    glow += float3(1.0) / (0.1 + d * 200.0);

    float s = 0.25;
    for (int i = 0; i < 18; i++) {
        s += 0.25;
        p.xy = p.xy * rot98(time * 0.05);
        p.xz = p.xz * rot98(time * 0.1);

        float d2 = torus(p, float2(s, 0.14));

        float intensity = 1.0 / (1.0 + pow(abs(d2 * 15.0), 1.3));
        if (i == 6 && id == 0.0) {
            glow += float3(1.0, 0.3, 1.0) * intensity;
        } else if(i == 15 && id == 1.0) {
            glow += float3(1.0, 1.0, 0.1) * intensity;
        } else if(i == 20 && id == 2.0) {
            glow += float3(0.1, 1.0, 0.1) * intensity;
        } else if(i == 25 && id == 3.0) {
            glow += float3(0.1, 1.0, 1.0) * intensity;
        }

        d = min(d, d2);
    }

    MapResult ret;
    ret.value = d;
    ret.oGlow = glow;

    return ret;
}

fragment float4 shader_day98a(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              constant float& time[[buffer(1)]]) {

    float timeX = time + 10.0;

    float2 uv = pixPos.xy / res.xy;
    float2 v = uv * 2.0 - 1.0;
    v.x /= res.y / res.x;

    float id = floor(hash98a(floor(time * 5.0 * hash98a(floor(timeX * 0.2)))) * 5.0);

    float3 ro = float3(0.0, 0.0, -10.0);
    float3 rd = normalize(float3(v, 1.0));

    float3 p = ro + rd;
    float3 glow = float3(0.0);
    for (int i = 0; i < 14; i++) {
        MapResult ret = map98(p, timeX, id, glow);
        p += rd * ret.value;
        glow = ret.oGlow;
    }

    float3 col = glow;
    col *= pow(uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.8) * 2.0;
    col = pow(col, float3(1.0 / 2.2));

    return float4(col, 1.0);
}

// MARK: - Day98b

// https://www.shadertoy.com/view/wsfcDM 集中線

#define ANIMATE 10.0
#define INV_ANIMATE_FREQ 0.05
#define RADIUS 1.3
#define FREQ 10.0
#define LENGTH 2.0
#define SOFTNESS 0.1
#define WEIRDNESS 0.1

#define lofi(x,d) (floor((x)/(d))*(d))

float2 mod98(float2 x, float2 y) {
    return x - y * floor(x / y);
}

float hash98b(float2 v) {
    return fract(sin(dot( v, float2(89.44, 19.36))) * 22189.22);
}

float iHash(float2 v, float2 r) {
    float4 h = float4(
                      hash98b( float2( floor( v * r + float2( 0.0, 0.0 ) ) / r ) ),
                      hash98b( float2( floor( v * r + float2( 0.0, 1.0 ) ) / r ) ),
                      hash98b( float2( floor( v * r + float2( 1.0, 0.0 ) ) / r ) ),
                      hash98b( float2( floor( v * r + float2( 1.0, 1.0 ) ) / r ) )
                      );
    float2 ip = float2(smoothstep(
                                  float2(0.0),
                                  float2(1.0),
                                  mod98(v * r, 1.0))
                       );
    return mix(
               mix( h.x, h.y, ip.y ),
               mix( h.z, h.w, ip.y ),
               ip.x
               );
}

float noise98(float2 v) {
    float sum = 0.0;
    for (int i = 1; i < 7; i ++) {
        sum += iHash(
                     v + float2(i),
                     float2(2.0 * pow( 2.0, float(i)))) / pow(2.0, float(i)
                                                              );
    }
    return sum;
}

fragment float4 shader_day98b(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              constant float& time[[buffer(1)]],
                              texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 uv = (pixPos.xy * 2.0 - res.xy) / res.xy;
    float2 puv = float2(
                        WEIRDNESS * length(uv) + ANIMATE * lofi(time, INV_ANIMATE_FREQ),
                        FREQ * atan2(uv.y, uv.x)
                        );
    float value = noise98(puv);
    value = length(uv) - RADIUS - LENGTH * (value - 0.5);
    value = smoothstep(-SOFTNESS, SOFTNESS, value);

    float2 uvColor = pixPos.xy / res.xy;
    float4 color = texture.sample(s, uvColor);
    float3 ret = mix(value, color.rgb, 0.5);
    //float3 ret = (1.0 - value) * color.rgb;
    return float4(ret, 1.0);
}

// MARK: - Day99

// https://www.shadertoy.com/view/ld3Gz2 Snail

float sdSphere(float3 p, float4 s) {
    return length(p - s.xyz) - s.w;
}

float sdEllipsoid(float3 p, float3 c, float3 r) {
    return (length((p - c) / r) - 1.0) * min(min(r.x, r.y), r.z);
}

float sdCircle99(float2 p, float2 c, float r) {
    return length(p - c) - r;
}

float sdTorus99(float3 p, float2 t) {
    return length(float2(length(p.xz) - t.x, p.y)) - t.y;
}

float sdCapsule(float3 p, float3 a, float3 b, float r) {
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float2 udSegment(float3 p, float3 a, float3 b) {
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return float2(length(pa - ba * h), h);
}

float det( float2 a, float2 b ) { return a.x * b.y - b.x * a.y; }

float3 getClosest(float2 b0, float2 b1, float2 b2) {
    float a = det(b0,b2);
    float b = 2.0 * det(b1,b0);
    float d = 2.0 * det(b2,b1);
    float f = b * d - a * a;
    float2  d21 = b2-b1;
    float2  d10 = b1-b0;
    float2  d20 = b2-b0;
    float2  gf = 2.0*(b*d21+d*d10+a*d20); gf = float2(gf.y,-gf.x);
    float2  pp = -f*gf/dot(gf,gf);
    float2  d0p = b0-pp;
    float ap = det(d0p,d20);
    float bp = 2.0*det(d10,d0p);
    float t = clamp( (ap+bp)/(2.0*a+b+d), 0.0 ,1.0 );
    return float3( mix(mix(b0,b1,t), mix(b1,b2,t),t), t );
}

float4 sdBezier(float3 a, float3 b, float3 c, float3 p) {
    float3 w = normalize( cross( c-b, a-b ) );
    float3 u = normalize( c-b );
    float3 v =          ( cross( w, u ) );

    float2 a2 = float2( dot(a-b,u), dot(a-b,v) );
    float2 b2 = float2( 0.0 );
    float2 c2 = float2( dot(c-b,u), dot(c-b,v) );
    float3 p3 = float3( dot(p-b,u), dot(p-b,v), dot(p-b,w) );

    float3 cp = getClosest( a2-p3.xy, b2-p3.xy, c2-p3.xy );

    return float4( sqrt(dot(cp.xy,cp.xy)+p3.z*p3.z), cp.z, length(cp.xy), p3.z );
}

float smin(float a, float b, float k) {
    float h = clamp( 0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float smax99(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(a, b, h) + k * h * (1.0 - h);
}

float3 smax99(float3 a, float3 b, float k) {
    float3 h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(a, b, h) + k * h * (1.0 - h);
}

//---------------------------------------------------------------------------

float hash1_99(float n) {
    return fract(sin(n) * 43758.5453123);
}

float3 hash3_99(float n) {
    return fract(sin(n+float3(0.0,13.1,31.3))*158.5453123);
}

float3 forwardSF(float i, float n) {
    const float PI  = 3.141592653589793238;
    const float PHI = 1.618033988749894848;
    float phi = 2.0*PI*fract(i/PHI);
    float zi = 1.0 - (2.0*i+1.0)/n;
    float sinTheta = sqrt( 1.0 - zi*zi);
    return float3( cos(phi)*sinTheta, sin(phi)*sinTheta, zi);
}

//---------------------------------------------------------------------------

//#define ZERO (min(iFrame, 0))
#define ZERO 0

//---------------------------------------------------------------------------

struct MapShellResult {
    float value;
    float4 matInfo;
};

////float mapShell(float3 p, out float4 matInfo) {
MapShellResult mapShell(float3 p, float4 matInfo) {
    const float sc = 1.0 / 1.0;
    p -= float3(0.05,0.12,-0.09);

    p *= sc;

    float3 q = float3x3(-0.6333234236, -0.7332753384, 0.2474039592,
                        0.7738444477, -0.6034162289, 0.1924931824,
                        0.0081370606,  0.3133626215, 0.9495986813) * p;

    const float b = 0.1759;

    float r = length( q.xy );
    float t = atan2( q.y, q.x );

    // https://swiftcoder.wordpress.com/2010/06/21/logarithmic-spiral-distance-field/
    float n = (log(r)/b - t)/(2.0 * M_PI_F);
    float nm = (log(0.11)/b-t)/(2.0 * M_PI_F);
    n = min(n,nm);

    float ni = floor( n );

    float r1 = exp( b * (t + 2.0 * M_PI_F * ni));
    float r2 = r1 * 3.019863;

    //-------

    float h1 = q.z + 1.5*r1 - 0.5; float d1 = sqrt((r1-r)*(r1-r)+h1*h1) - r1;
    float h2 = q.z + 1.5*r2 - 0.5; float d2 = sqrt((r2-r)*(r2-r)+h2*h2) - r2;

    float d, dx, dy;
    if( d1<d2 ) { d = d1; dx=r1-r; dy=h1; }
    else        { d = d2; dx=r2-r; dy=h2; }

    //float di = texture( iChannel2, float2(t+r,0.5), 0. ).x;
    float di = 0.0;
    d += 0.002 * di;

    matInfo = float4(dx,dy,r/0.4, t / M_PI_F);

    float3 s = q;
    q = q - float3(0.34,-0.1,0.03);
    q.xy = float2x2(0.8,0.6,-0.6,0.8) * q.xy;
    d = smin( d, sdTorus99( q, float2(0.28,0.05) ), 0.06);
    d = smax99( d, -sdEllipsoid(q,float3(0.0,0.0,0.0),float3(0.24,0.36,0.24) ), 0.03 );
    d = smax99( d, -sdEllipsoid(s,float3(0.52,-0.0,0.0),float3(0.42,0.23,0.5) ), 0.05 );

    MapShellResult ret;
    ret.value = d / sc;
    ret.matInfo = matInfo;
    return ret;
}

struct MapSnailResult {
    float2 value;
    float4 matInfo;
};

MapSnailResult mapSnail(float3 p, float4 matInfo, float time) {
    float3 head = float3(-0.76,0.6,-0.3);
    float3 q = p - head;

    float4 b1 = sdBezier( float3(-0.13,-0.65,0.0), float3(0.24,0.9+0.11,0.0), head+float3(0.05,0.01-0.02,0.0), p );
    float d1 = b1.x;
    d1 -= smoothstep(0.0,0.2,b1.y)*(0.16 - 0.75*0.07*smoothstep(0.5,1.0,b1.y));
    matInfo.xyz = b1.yzw;
    float d2;

    d2 = sdSphere( q, float4(0.0,-0.06,0.0,0.085) );
    d1 = smin( d1, d2, 0.03 );

    d1 = smin( d1, sdSphere(p,float4(0.05,0.52,0.0,0.13)), 0.07 );

    q.xz = float2x2(0.8,0.6,-0.6,0.8) * q.xz;

    float3 sq = float3( q.xy, abs(q.z) );

    // top antenas
    float3 af = 0.05 * sin(0.5 * time + float3(0.0,1.0,3.0) + float3(2.0,1.0,0.0)*sign(q.z) );
    float4 b2 = sdBezier( float3(0.0), float3(-0.1,0.2,0.2), float3(-0.3,0.2,0.3)+af, sq );
    float d3 = b2.x;
    d3 -= 0.03 - 0.025*b2.y;
    d1 = smin( d1, d3, 0.04 );
    d3 = sdSphere( sq, float4(-0.3,0.2,0.3,0.016) + float4(af,0.0) );
    d1 = smin( d1, d3, 0.01 );

    // bottom antenas
    float3 bf = 0.02*sin(0.3 * time + float3(4.0,1.0,2.0) + float3(3.0,0.0,1.0)*sign(q.z) );
    float2 b3 = udSegment( sq, float3(0.06,-0.05,0.0), float3(-0.04,-0.2,0.18)+bf );
    d3 = b3.x;
    d3 -= 0.025 - 0.02*b3.y;
    d1 = smin( d1, d3, 0.06 );
    d3 = sdSphere( sq, float4(-0.04,-0.2,0.18,0.008)+float4(bf,0.0) );
    d1 = smin( d1, d3, 0.02 );

    // bottom
    float3 pp = p-float3(-0.17,0.15,0.0);
    float co = 0.988771078;
    float si = 0.149438132;
    pp.xy = float2x2(co,-si,si,co) * pp.xy;
    d1 = smin( d1, sdEllipsoid( pp, float3(0.0,0.0,0.0), float3(0.084,0.3,0.15) ), 0.05 );
    d1 = smax99( d1, -sdEllipsoid( pp, float3(-0.08,-0.0,0.0), float3(0.06,0.55,0.1) ), 0.02 );

    // disp
    //float dis = texture( iChannel1, 5.0*p.xy, 0. ).x;
    float dis = 0.0;
    float dx = 0.5 + 0.5*(1.0-smoothstep(0.5,1.0,b1.y));
    d1 -= 0.005*dis*dx*0.5;

    MapSnailResult ret;
    ret.value = float2(d1, 1.0);
    ret.matInfo = matInfo;
    return ret;
}

float mapDrop(float3 p) {
    p -= float3(-0.26,0.25,-0.02);
    p.x -= 2.5 * p.y * p.y;
    return sdCapsule(p, float3(0.0,-0.06,0.0), float3(0.014,0.06,0.0), 0.037);
}

float mapLeaf(float3 p) {
    p -= float3(-1.8,0.6,-0.75);

    p = float3x3(0.671212, 0.366685, -0.644218,
                 -0.479426, 0.877583,  0.000000,
                 0.565354, 0.308854,  0.764842) * p;

    p.y += 0.2*exp(-abs(2.0*p.z) );

    float ph = 0.25*50.0*p.x - 0.25*75.0*abs(p.z);
    float rr = sin( ph );
    rr = rr*rr;
    rr = rr*rr;
    p.y += 0.005*rr;

    float r = clamp((p.x+2.0)/4.0,0.0,1.0);
    r = 0.0001 + r*(1.0-r)*(1.0-r)*6.0;

    rr = sin( ph*2.0 );
    rr = rr*rr;
    rr *= 0.5+0.5*sin( p.x*12.0 );

    float ri = 0.035*rr;

    float d = sdEllipsoid( p, float3(0.0), float3(2.0,0.25*r,r+ri) );

    float d2 = p.y-0.02;
    d = smax99( d, -d2, 0.02 );

    return d;
}

struct MapOpaqueResult {
    float2 value;
    float4 matInfo;
};

MapOpaqueResult mapOpaque(float3 p, float4 matInfo, float time) {
    matInfo = float4(0.0);

    //--------------
    MapSnailResult mapSnailResult = mapSnail(p, matInfo, time);
    float2 res = mapSnailResult.value;
    matInfo = mapSnailResult.matInfo;

    //---------------
    float4 tmpMatInfo = float4(0.0);
    MapShellResult mapShellResult = mapShell(p, tmpMatInfo);
    float d4 = mapShellResult.value;
    tmpMatInfo = mapShellResult.matInfo;

    if (d4 < res.x) { res = float2(d4, 2.0); matInfo = tmpMatInfo; }

    //---------------

    // plant
    float4 b3 = sdBezier(float3(-0.15, -1.5, 0.0), float3(-0.1, 0.5, 0.0), float3(-0.6, 1.5, 0.0), p);
    d4 = b3.x;
    d4 -= 0.04 - 0.02 * b3.y;
    if (d4 < res.x) { res = float2(d4, 3.0); }

    //----------------------------

//    float d5 = mapLeaf(p);
//    if (d5 < res.x) res = float2(d5, 4.0);

    MapOpaqueResult ret;
    ret.value = res;
    ret.matInfo = matInfo;
    return ret;
}

float3 calcNormalOpaque(float3 pos, float eps, float time) {
    float4 kk = float4(0.0);
    float2 e = float2(1.0, -1.0) * 0.5773 * eps;

    MapOpaqueResult ret1 = mapOpaque(pos + e.xyy, kk, time);
    kk = ret1.matInfo;

    MapOpaqueResult ret2 = mapOpaque(pos + e.yyx, kk, time);
    kk = ret2.matInfo;

    MapOpaqueResult ret3 = mapOpaque(pos + e.yxy, kk, time);
    kk = ret3.matInfo;

    MapOpaqueResult ret4 = mapOpaque(pos + e.xxx, kk, time);
    kk = ret4.matInfo;

    return normalize(e.xyy * ret1.value.x +
                     e.yyx * ret2.value.x +
                     e.yxy * ret3.value.x +
                     e.xxx * ret4.value.x);
}

//=========================================================================

float mapLeafWaterDrops(float3 p) {
    p -= float3(-1.8,0.6,-0.75);
    float3 s = p;
    p = float3x3(0.671212, 0.366685, -0.644218,
                 -0.479426, 0.877583,  0.000000,
                 0.565354, 0.308854,  0.764842) * p;

    float3 q = p;
    p.y += 0.2*exp(-abs(2.0*p.z) );

    float r = clamp((p.x + 2.0) / 4.0, 0.0, 1.0);
    r = r * (1.0 - r) * (1.0 - r) * 6.0;
    float d1 = sdEllipsoid( q, float3(0.5,0.0,0.2), 1.0*float3(0.15,0.13,0.15) );
    float d2 = sdEllipsoid( q, float3(0.8,-0.07,-0.15), 0.5*float3(0.15,0.13,0.15) );
    float d3 = sdEllipsoid( s, float3(0.76,-0.8,0.6), 0.5*float3(0.15,0.2,0.15) );
    float d4 = sdEllipsoid( q, float3(-0.5,0.09,-0.2), float3(0.04,0.03,0.04) );

    d3 = max(d3, p.y - 0.01);

    return min(min(d1, d4), min(d2, d3));
}

struct MapTransparentResult {
    float2 value;
    float4 matInfo;
};

MapTransparentResult mapTransparent(float3 p, float4 matInfo) {
    matInfo = float4(0.0);

    float d5 = mapDrop( p );
    float2  res = float2(d5, 4.0);

    float d6 = mapLeafWaterDrops(p);
    res.x = min( res.x, d6 );

    MapTransparentResult ret;
    ret.value = res;
    ret.matInfo = matInfo;
    return ret;
}

float3 calcNormalTransparent(float3 pos, float eps) {
    float4 kk = float(0.0);
    float2 e = float2(1.0, -1.0) * 0.5773 * eps;

    MapTransparentResult ret1 = mapTransparent(pos + e.xyy, kk);
    kk = ret1.matInfo;

    MapTransparentResult ret2 = mapTransparent(pos + e.yyx, kk);
    kk = ret2.matInfo;

    MapTransparentResult ret3 = mapTransparent(pos + e.yxy, kk);
    kk = ret3.matInfo;

    MapTransparentResult ret4 = mapTransparent(pos + e.xxx, kk);
    kk = ret4.matInfo;

    return normalize(e.xyy * ret1.value.x +
                     e.yyx * ret2.value.x +
                     e.yxy * ret3.value.x +
                     e.xxx * ret4.value.x);
}

//=========================================================================

float calcAO99(float3 pos, float3 nor, float time) {
    float4 kk;
    float ao = 0.0;
    for (int i = ZERO; i < 32; i++) {
        float3 ap = forwardSF(float(i), 32.0);
        float h = hash1_99(float(i));
        ap *= sign(dot(ap, nor)) * h * 0.1;

        //ao += clamp( mapOpaque( pos + nor*0.01 + ap, kk ).x*3.0, 0.0, 1.0 );
        MapOpaqueResult ret = mapOpaque(pos + nor * 0.01 + ap, kk, time);
        kk = ret.matInfo;
        ao += clamp( ret.value.x * 3.0, 0.0, 1.0);
    }
    ao /= 32.0;
    return clamp(ao * 6.0, 0.0, 1.0);
}

float calcSSS(float3 pos, float3 nor, float time) {
    float4 kk;
    float occ = 0.0;
    for (int i = ZERO; i < 8; i++) {
        float h = 0.002 + 0.11 * float(i) / 7.0;
        float3 dir = normalize(sin(float(i) * 13.0 + float3(0.0, 2.1, 4.2)));
        dir *= sign(dot(dir, nor));

        //occ += (h-mapOpaque(pos-h*dir, kk).x);
        MapOpaqueResult ret = mapOpaque(pos - h * dir, kk, time);
        kk = ret.matInfo;
        occ += (h - ret.value.x);
    }
    occ = clamp(1.0 - 11.0 * occ / 8.0, 0.0, 1.0);
    return occ * occ;
}

float calcSoftShadow(float3 ro, float3 rd, float k, float time) {
    float4 kk;
    float res = 1.0;
    float t = 0.01;
    for (int i = ZERO; i < 32; i++) {
        //float h = mapOpaque(ro + rd * t, kk ).x;
        MapOpaqueResult ret = mapOpaque(ro + rd * t, kk, time);
        kk = ret.matInfo;
        float h = ret.value.x;

        res = min(res, smoothstep(0.0, 1.0, k * h / t));
        t += clamp(h, 0.04, 0.1);
        if (res < 0.01) break;
    }
    return clamp(res,0.0,1.0);
}

float3 shadeOpaque(float3 ro, float3 rd, float t, float m, float4 matInfo, float time) {
    float eps = 0.002;

    float3 pos = ro + t * rd;
    float3 nor = calcNormalOpaque(pos, eps, time);

    float3 mateD = float3(0.0);
    float3 mateS = float3(0.0);
    float2 mateK = float2(0.0);
    float3 mateE = float3(0.0);

    float focc = 1.0;
    float fsha = 1.0;

    if( m<1.5 ) // snail body
    {
        //float dis = texture( iChannel1, 5.0*pos.xy ).x;
        float dis = 0.0;

        float be = sdEllipsoid( pos, float3(-0.3,-0.5,-0.1), float3(0.2,1.0,0.5) );
        be = 1.0-smoothstep( -0.01, 0.01, be );

        float ff = abs(matInfo.x-0.20);

        mateS = 6.0*mix( 0.7*float3(2.0,1.2,0.2), float3(2.5,1.8,0.9), ff );
        mateS += 2.0*dis;
        mateS *= 1.5;
        mateS *= 1.0 + 0.5*ff*ff;
        mateS *= 1.0-0.5*be;

        mateD = float3(1.0,0.8,0.4);
        mateD *= dis;
        mateD *= 0.015;
        mateD += float3(0.8,0.4,0.3)*0.15*be;

        mateK = float2( 60.0, 0.7 + 2.0*dis );

        float f = clamp( dot( -rd, nor ), 0.0, 1.0 );
        f = 1.0-pow( f, 8.0 );
        //f = 1.0 - (1.0-f)*(1.0-texture( iChannel2, 0.3*pos.xy ).x);
        f = 1.0 - (1.0 - f);
        mateS *= float3(0.5,0.1,0.0) + f*float3(0.5,0.9,1.0);

        //float b = 1.0-smoothstep( 0.25,0.55,abs(pos.y));
        focc = 0.2 + 0.8*smoothstep( 0.0, 0.15, sdSphere(pos,float4(0.05,0.52,0.0,0.13)) );
    }
    else if( m<2.5 ) // shell
    {
        mateK = float2(0.0);

        float tip = 1.0-smoothstep(0.05,0.4, length(pos-float3(0.17,0.2,0.35)) );
        mateD = mix( 0.7*float3(0.2,0.21,0.22), 0.2*float3(0.15,0.1,0.0), tip );

        float2 uv = float2(0.5 * atan2(matInfo.x, matInfo.y) / 3.1416, 1.5 * matInfo.w);

        //float3 ral = texture( iChannel1, float2(2.0*matInfo.w+matInfo.z*0.5,0.5) ).xxx;
        float3 ral = float3(0.0);
        mateD *= 0.25 + 0.75*ral;

        float pa = smoothstep(-0.2,0.2, 0.3+sin(2.0+40.0*uv.x + 3.0*sin(11.0*uv.x)) );
        float bar = mix(pa,1.0,smoothstep(0.7,1.0,tip));
        bar *= (matInfo.z<0.6) ? 1.0 : smoothstep( 0.17, 0.21, abs(matInfo.w)  );
        mateD *= float3(0.06,0.03,0.0)+float3(0.94,0.97,1.0)*bar;

        mateK = float2( 64.0, 0.2 );
        mateS = 1.5*float3(1.0,0.65,0.6) * (1.0-tip);//*0.5;
    }
    else if( m<3.5 ) // plant
    {
        mateD = float3(0.05,0.1,0.0)*0.2;
        mateS = float3(0.1,0.2,0.02)*25.0;
        mateK = float2(5.0,1.0);

        float fre = clamp(1.0+dot(nor,rd), 0.0, 1.0 );
        mateD += 0.2*fre*float3(1.0,0.5,0.1);

        //float3 te = texture( iChannel2, pos.xy*0.2 ).xyz;
        float3 te = float3(0.0);
        mateS *= 0.5 + 1.5*te;
        mateE = 0.5*float3(0.1,0.1,0.03)*(0.2+0.8*te.x);
    }
    else //if( m<4.5 ) // leave
    {
        float3 p = pos - float3(-1.8,0.6,-0.75);
        //float3 s = p;
        p = float3x3(0.671212, 0.366685, -0.644218,
                     -0.479426, 0.877583,  0.000000,
                     0.565354, 0.308854,  0.764842)*p;

        float3 q = p;
        p.y += 0.2*exp(-abs(2.0*p.z) );

        float v = smoothstep( 0.01, 0.02, abs(p.z));

        float rr = sin( 4.0*0.25*50.0*p.x - 4.0*0.25*75.0*abs(p.z) );

        //float3 te = texture( iChannel2, p.xz*0.35 ).xyz;
        float3 te = float3(0.0);

        float r = clamp((p.x+2.0)/4.0,0.0,1.0);
        r = r*(1.0-r)*(1.0-r)*6.0;
        float ff = length(p.xz/float2(2.0,r));

        mateD = mix( float3(0.07,0.1,0.0), float3(0.05,0.2,0.01)*0.25, v );
        mateD = mix( mateD, float3(0.16,0.2,0.01)*0.25, ff );
        mateD *= 1.0 + 0.25*te;
        mateD *= 0.8;

        mateS = float3(0.15,0.2,0.02)*0.8;
        mateS *= 1.0 + 0.2*rr;
        mateS *= 0.8;

        mateK = float2(64.0,0.25);

        //---------------------

        //nor.xz += v*0.15*(-1.0+2.0*texture( iChannel3, 1.0*p.xz ).xy);
        nor.xz += v * 0.15 * (-1.0 + 2.0 * float2(0.0));
        nor = normalize( nor );

        float d1 = sdEllipsoid( q, float3( 0.5-0.07, 0.0,  0.20), 1.0*float3(1.4*0.15,0.13,0.15) );
        float d2 = sdEllipsoid( q, float3( 0.8-0.05,-0.07,-0.15), 0.5*float3(1.3*0.15,0.13,0.15) );
        float d4 = sdEllipsoid( q, float3(-0.5-0.07, 0.09,-0.20), 1.0*float3(1.4*0.04,0.03,0.04) );
        float dd = min(d1,min(d2,d4));
        fsha = 0.05 + 0.95*smoothstep(0.0,0.05,dd);

        d1 = abs( sdCircle99( q.xz, float2( 0.5, 0.20), 1.0*0.15 ));
        d2 = abs( sdCircle99( q.xz, float2( 0.8,-0.15), 0.5*0.15 ));
        d4 = abs( sdCircle99( q.xz, float2(-0.5,-0.20), 1.0*0.04 ));
        dd = min(d1,min(d2,d4));
        focc *= 0.55 + 0.45*smoothstep(0.0,0.08,dd);

        d1 = distance( q.xz, float2( 0.5-0.07, 0.20) );
        d2 = distance( q.xz, float2( 0.8-0.03,-0.15) );
        fsha += (1.0-smoothstep(0.0,0.10,d1))*1.5;
        fsha += (1.0-smoothstep(0.0,0.05,d2))*1.5;
    }

    const float3 sunDir = normalize(float3(0.2, 0.1, 0.02));

    float3 hal = normalize(sunDir - rd);
    float fre = clamp(1.0 + dot(nor, rd), 0.0, 1.0);
    float occ = calcAO99(pos, nor, time) * focc;
    float sss = calcSSS(pos, nor, time);
    sss = sss * occ + fre * occ + (0.5 + 0.5 * fre) * pow(abs(matInfo.x - 0.2), 1.0) * occ;

    float dif1 = clamp(dot(nor, sunDir), 0.0, 1.0);
    float sha = calcSoftShadow(pos, sunDir, 20.0, time);
    dif1 *= sha * fsha;
    float spe1 = clamp(dot(nor, hal), 0.0, 1.0);

    float bou = clamp(0.3 - 0.7 * nor.y, 0.0, 1.0);

    // illumination

    float3 col = float3(0.0);
    col += 7.0*float3(1.7,1.2,0.6)*dif1*2.0;           // sun
    col += 4.0*float3(0.2,1.2,1.6)*occ*(0.5+0.5*nor.y);    // sky
    col += 1.8*float3(0.1,2.0,0.1)*bou*occ;                // bounce

    col *= mateD;

    col += 0.4*sss*(float3(0.15,0.1,0.05)+float3(0.85,0.9,0.95)*dif1)*(0.05+0.95*occ)*mateS; // sss
    col = pow(col,float3(0.6,0.8,1.0));

    col += float3(1.0,1.0,1.0)*0.2*pow( spe1, 1.0+mateK.x )*dif1*(0.04+0.96*pow(fre,4.0))*mateK.x*mateK.y;   // sun lobe1
    col += float3(1.0,1.0,1.0)*0.1*pow( spe1, 1.0+mateK.x/3.0 )*dif1*(0.1+0.9*pow(fre,4.0))*mateK.x*mateK.y; // sun lobe2
    col += 0.1*float3(1.0,max(1.5-0.7*col.y,0.0),2.0)*occ*occ*smoothstep( 0.0, 0.3, reflect( rd, nor ).y )*mateK.x*mateK.y*(0.04+0.96*pow(fre,5.0)); // sky

    col += mateE;

    return col;
}

float3 shadeTransparent(float3 ro, float3 rd, float t, float m, float4 matInfo, float3 col, float depth) {
    const float3 sunDir = normalize(float3(0.2, 0.1, 0.02));
    float3 oriCol = col;

    float dz = depth - t;
    float ao = clamp(dz*50.0,0.0,1.0);
    float3  pos = ro + t*rd;
    float3  nor = calcNormalTransparent( pos, 0.002 );
    float fre = clamp( 1.0 + dot( rd, nor ), 0.0, 1.0 );
    float3  hal = normalize( sunDir-rd );
    float3  ref = reflect( -rd, nor );
    float spe1 = clamp( dot(nor,hal), 0.0, 1.0 );
    float spe2 = clamp( dot(ref,sunDir), 0.0, 1.0 );

    float ds = 1.6 - col.y;

    col *= mix( float3(0.0,0.0,0.0), float3(0.4,0.6,0.4), ao );

    col += ds*1.5*float3(1.0,0.9,0.8)*pow( spe1, 80.0 );
    col += ds*0.2*float3(0.9,1.0,1.0)*smoothstep(0.4,0.8,fre);
    col += ds*0.9*float3(0.6,0.7,1.0)*smoothstep( -0.5, 0.5, -reflect( rd, nor ).y )*smoothstep(0.2,0.4,fre);
    col += ds*0.5*float3(1.0,0.9,0.8)*pow( spe2, 80.0 );
    col += ds*0.5*float3(1.0,0.9,0.8)*pow( spe2, 16.0 );
    //col += float3(0.8,1.0,0.8)*0.5*smoothstep(0.3,0.6,texture( iChannel1, 0.8*nor.xy ).x)*(0.1+0.9*fre*fre);
    col += float3(0.8, 1.0, 0.8) * 0.5 * smoothstep(0.3, 0.6, 0.0) * (0.1 + 0.9 * fre * fre);

    // hide aliasing a bit
    return mix( col, oriCol, smoothstep(0.6,1.0,fre) );
}

//--------------------------------------------

struct IntersectOpaqueResult {
    float2 value;
    float4 matInfo;
};

IntersectOpaqueResult intersectOpaque(float3 ro, float3 rd, float mindist, float maxdist, float4 matInfo, float time) {
    float2 res = float2(-1.0);

    float t = mindist;

    //for (int i = ZERO; i < 64; i++) {
    for (int i = ZERO; i < 18; i++) {
        float3 p = ro + t * rd;

        MapOpaqueResult ret = mapOpaque(p, matInfo, time);
        matInfo = ret.matInfo;
        float2 h = ret.value;

        res = float2(t, h.y);

        if (h.x < (0.001 * t) || t > maxdist) break;

        t += h.x * 0.9;
    }

    IntersectOpaqueResult ret;
    ret.value = res;
    ret.matInfo = matInfo;
    return ret;
}

struct IntersectTransparentResult {
    float2 value;
    float4 matInfo;
};

IntersectTransparentResult intersectTransparent(float3 ro, float3 rd, float mindist, float maxdist, float4 matInfo) {
    float2 res = float2(-1.0);

    float t = mindist;
    for (int i = ZERO; i < 64; i++) {
        float3 p = ro + t * rd;

        MapTransparentResult ret = mapTransparent(p, matInfo);
        matInfo = ret.matInfo;
        float2 h = ret.value;

        res = float2(t, h.y);

        if (h.x < (0.001 * t) || t > maxdist) break;

        t += h.x;
    }
    IntersectTransparentResult ret;
    ret.value = res;
    ret.matInfo = matInfo;
    return ret;
}

float3 background(float3 d) {
    float3 col = float3(0.0);
    for (int i = ZERO; i < 10; i++) {
        float3 tmp = float3(0.0);
        col = smax99(col, tmp, 0.5);
    }
    return pow(col, float3(3.5, 3.0, 6.0)) * 0.2;
}

float3 render(float3 ro, float3 rd, float2 q, float time) {
    const float3 sunDir = normalize(float3(0.2, 0.1, 0.02));

    float3 col = background(rd);

    float mindist = 1.0;
    float maxdist = 4.0;

    float4 matInfo = float4(0.0);

    IntersectOpaqueResult intersectOpaqueResult1 = intersectOpaque(ro, rd, mindist, maxdist, matInfo, time);
    matInfo = intersectOpaqueResult1.matInfo;
    float2 tm = intersectOpaqueResult1.value;

    if (tm.y > -0.5 && tm.x < maxdist) {
        col = shadeOpaque(ro, rd, tm.x, tm.y, matInfo, time);
        maxdist = tm.x;
    }

    //-----------------------------

//    IntersectTransparentResult intersectTransparentResult = intersectTransparent(ro, rd, mindist, maxdist, matInfo);
//    matInfo = intersectTransparentResult.matInfo;
//    tm = intersectTransparentResult.value;
//
//    if (tm.y > -0.5 && tm.x < maxdist) {
//        col = shadeTransparent(ro, rd, tm.x, tm.y, matInfo, col, maxdist);
//    }

    //-----------------------------

    float sun = clamp(dot(rd, sunDir), 0.0, 1.0);
    col += 1.0 * float3(1.5, 0.8, 0.7) * pow(sun, 4.0);

    //-----------------------------

    col = pow(col, float3(0.45));
    col = float3(1.05, 1.0, 1.0) * col * (0.7 + 0.3 * col * max(3.0 - 2.0 * col, 0.0)) + float3(0.0, 0.0, 0.04);
    col *= 0.3 + 0.7 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.1);

    return clamp(col, 0.0, 1.0);
}

float3x3 setCamera(float3 ro, float3 rt) {
    float3 w = normalize(ro - rt);
    float m = sqrt(1.0 - w.y * w.y);
    return float3x3(w.z, 0.0, -w.x,
                    0.0, m * m, -w.z * w.y,
                    w.x * m, w.y * m, w.z * m);
}

fragment float4 shader_day99(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 p = (2.0 * pixPos.xy - res.xy) / res.y;
    p.y *= -1.0;
    float2 q = pixPos.xy / res.xy;
    q.y *= -1.0;

    float an = 1.87 - 0.04 * (1.0 - cos(0.5 * time));

    float3 ro = float3(-0.4, 0.2, 0.0) + 2.2 * float3(cos(an), 0.0, sin(an));
    float3 ta = float3(-0.6, 0.2, 0.0);
    float3x3 ca = setCamera(ro, ta);
    float3 rd = normalize(ca * float3(p, -2.8));

    float3 col = render(ro, rd, q, time);

    return float4(col, 1.0);
}

// MARK: - Day100

// https://www.shadertoy.com/view/4lfcz4 Snow fall

float2 mod2_100(float2 x, float2 y) {
    return x - y * floor(x / y);
}

float2 mod289_2(float2 x) {
    return float2(mod2_100(x, float2(289.0)));
}

float3 mod289_3(float3 x) {
    return float3(mod(x, 289.0));
}

float3 permute100(float3 x) {
    return mod289_3(((x * 34.0) + 1.0) * x);
}

float snoise100(float2 v) {
    const float4 C = float4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    float2 i  = floor(v + dot(v, C.yy));
    float2 x0 = v - i + dot(i, C.xx);

    float2 i1;
    i1 = (x0.x > x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    i = mod289_2(i);
    float3 p = permute100(permute100( i.y + float3(0.0, i1.y, 1.0 ))
                          + i.x + float3(0.0, i1.x, 1.0 ));

    float3 m = max(0.5 - float3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    float3 x = 2.0 * fract(p * C.www) - 1.0;
    float3 h = abs(x) - 0.5;
    float3 ox = floor(x + 0.5);
    float3 a0 = x - ox;

    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    float3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;

    return 130.0 * dot(m, g);
}

float fbm100(float2 p) {
    float f = 0.0;
    float w = 0.5;
    for (int i = 0; i < 5; i ++) {
        f += w * snoise100(p);
        p *= 2.0;
        w *= 0.5;
    }
    return f;
}

float background(float2 uv, float2 res, float time) {
    uv.x += 1.0 / res.x - 1.0;

    float2 sunCenter = float2(0.3, 0.9);
    float suns = clamp(1.2 - distance(uv, sunCenter), 0.0, 1.0);
    float sunsh = smoothstep(0.85, 0.95, suns);

    float slope = 1.0 - smoothstep(0.55, 0.0, 0.8 + uv.x -2.3 * uv.y);

    float noise = abs(fbm100(uv * 1.5));
    slope = (noise * 0.2) + (slope - ((1.0 - noise) * slope * 0.1)) * 0.6;
    slope = clamp(slope, 0.0, 1.0);

    return 0.35 + (slope * (suns + 0.3)) + (sunsh * 0.6);
}

#define LAYERS 66

#define DEPTH1 .3
#define WIDTH1 .4
#define SPEED1 .6

#define DEPTH2 .1
#define WIDTH2 .3
#define SPEED2 .1

float snowing(float2 uv, float2 fragCoord, float2 res, float time) {
    const float3x3 p = float3x3(13.323122, 23.5112, 21.71123, 21.1212, 28.7312, 11.9312, 21.8112, 14.7212, 61.3934);
    float2 mp = float2(1.0) / res;
    uv.x += mp.x * 4.0;
    mp.y *= 0.25;
    float depth = smoothstep(DEPTH1, DEPTH2, mp.y);
    float width = smoothstep(WIDTH1, WIDTH2, mp.y);
    float speed = smoothstep(SPEED1, SPEED2, mp.y);
    float acc = 0.0;
    float dof = 5.0 * sin(time * 0.1);
    for (int i = 0; i < LAYERS; i++) {
        float fi = float(i);
        float2 q = uv * (1.0 + fi*depth);
        float w = width * mod(fi*7.238917,1.0) - width * 0.1 * sin(time * 2.0 + fi);
        q += float2(q.y * w, speed * time / (1.0 + fi * depth * 0.03));
        float3 n = float3(floor(q),31.189+fi);
        float3 m = floor(n)*0.00001 + fract(n);
        float3 mp = (31415.9+m) / fract(p*m);
        float3 r = fract(mp);
        float2 s = abs(mod2_100(q, float2(1.0)) - 0.5 + 0.9 * r.xy -0.45);
        s += 0.01*abs(2.0*fract(10.*q.yx)-1.);
        float d = 0.6*max(s.x-s.y,s.x+s.y)+max(s.x,s.y)-.01;
        float edge = 0.05 +0.05*min(.5*abs(fi-5.-dof),1.);
        acc += smoothstep(edge,-edge,d)*(r.x/(1.+.02*fi*depth));
    }
    return acc;
}

bool MysteryMountains(float4 c, float2 w, float2 res, float time) {
    float4 p = float4(w / res, 1, 1) - 0.5, d = p, t;
    p.z += time * 2.0;
    for (float i = 1.5; i > 0.3; i-=0.002) {
        c = float4(1.0, 1.0, 0.9, 9.0) + d.x;
        if (t.x > p.y * 0.017 + 1.3) return true;
        p += d;
    }
    return false;
}

fragment float4 shader_day100(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              constant float& time[[buffer(1)]]) {

    float2 uv = pixPos.xy / res.y;

    float bg = background(uv, res, time);
    float4 ret = float4(bg * 0.9, bg, bg * 1.1, 1.0);

    uv.y *= -1.0;

    float snowOut = snowing(uv, pixPos.xy, res, time);
    ret += float4(float3(snowOut), 1.0);

    return ret;
}

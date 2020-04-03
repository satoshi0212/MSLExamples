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

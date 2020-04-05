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

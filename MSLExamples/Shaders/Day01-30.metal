#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day3

fragment float4 fragment_day3(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              constant float& time[[buffer(1)]]) {
    float2 uv = (2.0 * pixPos.xy - res) / min(res.x, res.y);
    uv.y *= -1.0;

    float l = length(uv);
    float ring = abs(step(0.8, l) - step(1.0, l));
    float phase = atan2(uv.y, uv.x) + M_PI_F;
    float h = phase / (2.0 * M_PI_F);
    float s = saturate(l);
    float3 rgb = hsv2rgb(fract(h + 0.2 * time), s, 1.0);

    return float4(rgb, 1.0) * ring;
}

fragment float4 shader_day7(float4 pixPos [[position]],
                            constant float2& res[[buffer(0)]],
                            constant float& time[[buffer(1)]],
                            texture2d<float> tex[[texture(1)]]) {
    float4 col = 0.0;

    float2 texUV = pixPos.xy/res;
    texUV.x += 0.5 * sin(texUV.y + time) + 0.5;
    texUV.y += 0.5 * sin(texUV.x + time) + 0.5;

    constexpr sampler s(address::repeat, filter::linear);
    col = tex.sample(s, texUV);

    return col;
}

fragment float4 shader_day8(float4 pixPos [[position]],
                            constant float2& res [[buffer(0)]],
                            constant float& time [[buffer(1)]],
                            texture2d<float> tex[[texture(1)]]) {

    float2 uv = (2.0 * pixPos.xy - res) / min(res.x, res.y);
    uv.y *= -1.0;

    uv *= 4.0;
    float2 id = floor(uv);
    uv = fract(uv) * 2.0 - 1.0;
    float r = N21(id);
    float4 stepped = step(length(uv), 0.5 + 0.5 * sin(3 * time + 5 * r));

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float4 col = float4(tex.sample(s, pixPos.xy/res));

    col.rgb = dot(col.rgb, stepped.rgb);
    return col;
}

// MARK: - Day11

fragment float4 shader_day11(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {
    float4 col = 0;
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy/res;
    col = float4(texture.sample(s, uv));
    col.rgb = dot(col.rgb, float3(0.2126, 0.7152, 0.0722));
    return col;
}

// MARK: - Day13

fragment float4 shader_day13(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture[[texture(1)]]) {

    float2 uv = (2.0 * pixPos.xy - res) / min(res.x, res.y);
    uv.y *= -1.0;

    float idY = floor(res.y * uv.y / 5.0);
    float horizontalNoise = N11(idY + M_PI_F * fract(time));

    float outputmask = step(1.0 - 2.0 * 1.0, horizontalNoise);
    horizontalNoise *= outputmask;

    constexpr sampler s(address::repeat, filter::linear);
    float2 texUV = pixPos.xy / res;

    return texture.sample(s, texUV + 0.05 * horizontalNoise);
}

// MARK: - Day15

fragment float4 shader_day15(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             texture2d<float> tex[[texture(1)]]) {
    float4 col = 0.0;
    float2 uv = (2.0 * pixPos.xy -res) / min(res.x, res.y);

    float size = 0.7;
    float mask = step(length(uv), size);
    float distortion = -0.9;
    float2 offset = distortion * sqrt(length(uv)) * uv * mask;

    float2 texUV = pixPos.xy / res;
    constexpr sampler s(address::repeat, filter::linear);
    col = tex.sample(s, texUV + offset);

    return col;
}

// MARK: - Day24

fragment float4 shader_day24(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time [[buffer(1)]],
                              texture2d<float> cam[[texture(1)]]) {
    constexpr sampler s(address::repeat, filter::linear);
    float2 uv = (2.0 * pixPos.xy - res) / min(res.x, res.y);
    uv *= 1.3;
    uv.x += 0.5 * sin(uv.y + time);;
    uv.y += 0.5 * sin(uv.x + time);;
    float4 cameraImage = cam.sample(s, uv);
    return length(uv) * cameraImage;
}

// MARK: - Day25

fragment float4 shader_day25(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             constant float& volume[[buffer(2)]]) {

    float2 uv = (2.0 * pixPos.xy - res) / min(res.x, res.y);
    uv.y *= -1.0;

    float v = volume;
    float2 smallUV = 2.0 * fract(uv * 3 * v) - 1.0;
    float t1 = 0.2 * sin(5 * atan2(smallUV.y, smallUV.x) + 3.0 * time) + 0.8;
    float smallStar = 0.6 * v * step(length(smallUV), t1);

    uv *= 0.8 / clamp(v, 0.1, 0.8);

    float t2 = 0.2 * sin(5 * atan2(uv.y, uv.x)) + 0.8;
    float star = step(length(uv), t2);
    float4 yellowy = mix(float4(1.0, 1.0, 1.0, 1.0),
                         float4(1.0, 1.0, 0.0, 1.0),
                         smoothstep(0.5, 0.8, v));

    return mix(smallStar, yellowy, star);
}

// MARK: - Day26

fragment float4 shader_day26(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {
    float2 pos = pixPos.xy / res.xy;
    pos.y -= 0.5;

    float3 c = mix(float3(0.0, 1.0, 1.0), float3(0.0, 0.1, 0.1), pos.y);
    float v = sin((pos.x + time * 0.2) * 5.0) * 0.05 + sin((pos.x * 3.0 + time * 0.1) * 5.0) * 0.05;
    if (pos.y < v) {
        c = mix(c, float3(0.0, 0.5, 0.5), 0.2);
    }
    v = sin((pos.x + time * 0.1) * 5.0) * 0.05 + sin((pos.x * 3.0 + time * 0.05) * 5.0) * 0.05;
    if (pos.y < v) {
        c = mix(c, float3(0.0, 0.5, 0.5), 0.2);
    }
    return float4(c, 13.0);
}

// MARK: - Day28

half noise(half3 p) {
    half3 i = floor(p);
    half4 a = dot(i, half3(1.0, 57.0, 21.0)) + half4(0.0, 57.0, 21.0, 78.0);
    half3 f = cos((p - i) * acos(-1.0)) * (-0.5) + 0.5;
    a = mix(sin(cos(a) * a), sin(cos(1.0 + a) * (1.0 + a)), f.x);
    a.xy = mix(a.xz, a.yw, f.y);
    return mix(a.x, a.y, f.z);
}

half sphere(half3 p, half4 spr) {
    return length(spr.xyz - p) - spr.w;
}

half flame(half3 p, half time) {
    half d = sphere(p * half3(1.0, 0.5, 1.0), half4(0.0, -1.0, 0.0, 1.0));
    return d + (noise(p + half3(0.0, time * 2.0, 0.0)) + noise(p * 3.0) * 0.5) * 0.25 * (p.y);
}

half scene(half3 p, half time) {
    return min(half(100.0 - length(p)), abs(flame(p, time)));
}

half4 raymarch(half3 org, half3 dir, half time) {
    half d = 0.0;
    half glow = 0.0;
    half eps = 0.02;
    half3 p = org;
    bool glowed = false;

    for (int i = 0; i < 64; i++) {
        d = scene(p, time) + eps;
        p += d * dir;
        if (d > eps) {
            if (flame(p, time) < 0.0)
                glowed = true;
            if (glowed)
                glow = half(i) / 64.0;
        }
    }
    return half4(p, glow);
}

fragment half4 shader_day28(float4 pixPos [[position]],
                            constant float2& res[[buffer(0)]],
                            constant float& time[[buffer(1)]]) {

    half2 uv = half2((2.0 * pixPos.xy - res) / min(res.x, res.y));
    uv.y *= -1.0;

    half3 org = half3(0.0, -2.0, 4.0);
    half3 dir = normalize(half3(uv.x * 1.6, -uv.y, -1.5));

    half4 p = raymarch(org, dir, time);
    half glow = p.w;

    half4 col = mix(half4(1.0, 0.5, 0.1, 1.0), half4(0.1, 0.5, 1.0, 1.0), p.y * 0.02 + 0.4);

    return mix(half4(0.0), col, pow(glow * 2.0, 4.0));
}

// MARK: - Day28R

// https://www.shadertoy.com/view/MdX3zr flame

float noiseF(float3 p) {
    float3 i = floor(p);
    float4 a = dot(i, float3(1.0, 57.0, 21.0)) + float4(0.0, 57.0, 21.0, 78.0);
    float3 f = cos((p - i) * acos(-1.0)) * (-0.5) + 0.5;
    a = mix(sin(cos(a) * a), sin(cos(1.0 + a) * (1.0 + a)), f.x);
    a.xy = mix(a.xz, a.yw, f.y);
    return mix(a.x, a.y, f.z);
}

float sphere(float3 p, float4 spr) {
    return length(spr.xyz-p) - spr.w;
}

float flame(float3 p, float time) {
    float d = sphere(p*float3(1.,.5,1.), float4(.0,-1.,.0,1.));
    return d + (noiseF(p+float3(.0, time * 2.0, 0.0)) + noiseF(p*3.)*.5)*.25*(p.y) ;
}

float scene(float3 p, float time) {
    return min(100.0 - length(p), abs(flame(p, time)));
}

float4 raymarch(float3 org, float3 dir, float time) {
    float d = 0.0, glow = 0.0, eps = 0.02;
    float3 p = org;
    bool glowed = false;
    for (int i = 0; i < 64; i++) {
        d = scene(p, time) + eps;
        p += d * dir;
        if (d > eps) {
            if (flame(p, time) < .0)
                glowed = true;
            if (glowed)
                glow = float(i) / 64.0;
        }
    }
    return float4(p,glow);
}

fragment float4 shader_day28R(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              constant float& time[[buffer(1)]],
                              texture2d<float, access::sample> noiseTexture [[texture(2)]]) {

    float2 v = -1.0 + 2.0 * pixPos.xy / res.xy;
    v.x *= res.x / res.y;
    v.y *= -1.0;

    float3 org = float3(0.0, -2.0, 4.0);
    float3 dir = normalize(float3(v.x * 1.6, -v.y, -1.5));

    float limits = 0.5;
    float4 p = (v.x > -limits && v.x < limits) ? raymarch(org, dir, time) : float4(0.0);

    float glow = p.w;
    if (mod(time, 6.0) < 3.0) {
        float4 col = mix(float4(1.0, 0.5, 0.1, 1.0), float4(0.1, 0.5, 1.0, 1.0), p.y * 0.02 + 0.4);
        return mix(float4(0.0), col, pow(glow * 2.0, 4.0));
    } else {
        return mix(float4(1.0),
                   mix(float4(1.0, 0.5, 0.1, 1.0),
                       float4(0.1, 0.5, 1.0, 1.0),
                       p.y * 0.02 + 0.4),
                   pow(glow * 2.0, 4.0));
    }
}

// MARK: - Day29

float2 pmod(float2 p, float r) {
    float a = atan2(p.x, p.y) + M_PI_F / r;
    float n = M_PI_F * 2.0 / r;
    a = floor(a / n) * n;
    return p * rot(-a);
}

float box(float3 p, float3 b) {
    float3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

float ifsBox(float3 p, float time) {
    for (int i = 0; i < 5; i++) {
        p = abs(p) - 1.0;
        p.xy = p.xy * rot(time * 0.3);
        p.xz = p.xz * rot(time * 0.1);
    }
    p.xz = p.xz * rot(time);
    return box(p, float3(0.4, 0.8, 0.3));
}

float map(float3 p, float3 cPos, float time) {
    float3 p1 = p;
    p1.x = mod(p1.x - 5.0, 10.0) - 5.0;
    p1.y = mod(p1.y - 5.0, 10.0) - 5.0;
    p1.z = mod(p1.z, 16.0) - 8.0;
    p1.xy = pmod(p1.xy, 5.0);
    return ifsBox(p1, time);
}

fragment float4 shader_day29(float4 pixPos [[position]],
                            constant float2& res[[buffer(0)]],
                            constant float& time[[buffer(1)]]) {

    float2 p = float2((2.0 * pixPos.xy - res) / min(res.x, res.y));
    p.y *= -1.0;

    float3 cPos = float3(0.0, 0.0, -3.0 * time);
    float3 cDir = normalize(float3(0.0, 0.0, -1.0));
    float3 cUp  = float3(sin(time), 1.0, 0.0);
    float3 cSide = cross(cDir, cUp);

    float3 ray = normalize(cSide * p.x + cUp * p.y + cDir);

    float acc = 0.0;
    float acc2 = 0.0;
    float t = 0.0;
    for (int i = 0; i < 99; i++) {
        float3 pos = cPos + ray * t;
        float dist = map(pos, cPos, time);
        dist = max(abs(dist), 0.02);
        float a = exp(-dist * 3.0);
        if (mod(length(pos) + 24.0 * time, 30.0) < 3.0) {
            a *= 2.0;
            acc2 += a;
        }
        acc += a;
        t += dist * 0.5;
    }

    float3 col = float3(acc * 0.01, acc * 0.011 + acc2 * 0.002, acc * 0.012 + acc2 * 0.005);
    return float4(col, 1.0 - t * 0.03);
}

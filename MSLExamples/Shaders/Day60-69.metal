#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day60

// https://www.shadertoy.com/view/Xd2GR3 Tiled hexagon

float4 hexagon(float2 p) {
    float2 q = float2( p.x * 2.0 * 0.5773503, p.y + p.x * 0.5773503);
    float2 pi = floor(q);
    float2 pf = fract(q);
    float v = mod(pi.x + pi.y, 3.0);
    float ca = step(1.0, v);
    float cb = step(2.0, v);
    float2 ma = step(pf.xy, pf.yx);
    float e = dot(ma, 1.0 - pf.yx + ca * (pf.x + pf.y - 1.0) + cb * (pf.yx - 2.0 * pf.xy));
    p = float2(q.x + floor(0.5 + p.y / 1.5), 4.0 * p.y / 3.0) * 0.5 + 0.5;
    float f = length((fract(p) - 0.5) * float2(1.0, 0.85));
    return float4(pi + ca - cb * ma, e, f);
}

float hash60(float2 p) {
    float n = dot(p, float2(127.1, 311.7));
    return fract(sin(n) * 43758.5453);
}

float noise60(sampler s, float3 x, texture2d<float, access::sample> rgbNoiseTexture) {
    float3 p = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float2 uv = (p.xy + float2(37.0, 17.0) * p.z) + f.xy;
    float2 rg = rgbNoiseTexture.sample(s, (uv + 0.5) / 256.0).yx;
    return mix(rg.x, rg.y, f.z);
}

fragment float4 shader_day60(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> rgbNoiseTexture [[texture(4)]]) {
    constexpr sampler s(address::repeat, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float3 tot = float3(0.0);
    float2 pos = (-res.xy + 2.0 * pixPos.xy) / res.y;

    pos *= 1.0 + 0.3 * length(pos);

    float4 h = hexagon(8.0 * pos + 0.5 * time);
    float n = noise60(s, float3(0.3 * h.xy + time * 0.1, time), rgbNoiseTexture);
    float3 col = 0.15 + 0.15 * hash60(h.xy + 1.2) * float3(1.0);
    col *= smoothstep(0.10, 0.11, h.z);
    col *= smoothstep(0.10, 0.11, h.w);
    col *= 1.0 + 0.15 * sin(40.0 * h.z);
    col *= 0.75 + 0.5 * h.z * n;

    h = hexagon(6.0 * pos + 0.6 * time);
    n = noise60(s, float3(0.3 * h.xy + time * 0.1, time), rgbNoiseTexture);
    float3 colb = 0.9 + 0.8 * sin(hash60(h.xy) * 1.5 + 2.0 + float3(0.0, 1.0, 1.0));
    colb *= smoothstep(0.10, 0.11, h.z);
    colb *= 1.0 + 0.15 * sin(40.0 * h.z);
    colb *= 0.75 + 0.5 * h.z * n;

    h = hexagon(6.0 * (pos + 0.1 * float2(-1.3, 1.0)) + 0.6 * time);
    col *= 1.0 - 0.8 * smoothstep(0.45, 0.451, noise60(s, float3(0.3 * h.xy + time * 0.1, time), rgbNoiseTexture));
    col = mix(col, colb, smoothstep(0.45, 0.451, n));
    col *= pow(16.0 * uv.x * (1.0 - uv.x) * uv.y * (1.0 - uv.y), 0.1);
    tot += col;

    return float4(tot, 1.0);
}

// MARK: - Day61

// https://www.shadertoy.com/view/4lK3zy 独特な色合いのポスタライズ

float3 getGrayscale(float3 origColor) {
    return float3(origColor.r * 0.21 +
                  origColor.g * 0.72 +
                  origColor.b * 0.07);
}

float3 getPosterColor(float3 trueColor) {
    const float posterLevelRange = 1.0 / 3.0;
    float3 grayscale = getGrayscale(trueColor);
    float3 modColor = mix(trueColor, grayscale, -0.5);
    modColor = clamp(modColor * 2.0 - 0.4, 0.0, 1.0);
    return float3((floor(modColor.r / posterLevelRange + 0.5)) * posterLevelRange,
                  (floor(modColor.g / posterLevelRange + 0.5)) * posterLevelRange,
                  (floor(modColor.b / posterLevelRange + 0.5)) * posterLevelRange);
}

float easeIn (float perc) {
    return perc * perc;
}

fragment float4 shader_day61(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]],
                             texture2d<float, access::sample> noiseTexture [[texture(2)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    const int roundDist = 24;
    const float roundAmount = 0.3;
    const int averageDist = 2;

    float2 texelSize = float2(1.0) / res.xy;
    float3 color = float3(0.0, 0.0, 0.0);
    for (int x = -averageDist; x <= averageDist; x++) {
        for (int y = -averageDist; y <= averageDist; y++) {
            float4 texelColor = texture.sample(s, uv + (float2(texelSize.x * float(x), texelSize.y * float(y))));
            color += texelColor.rgb * texelColor.a;
        }
    }
    color /= (pow(float(2 * averageDist + 1), 2.0));

    float3 overlay = getGrayscale(noiseTexture.sample(s, uv).rgb);
    float3 posterColor = getPosterColor(color);

    float brighten = 0.0;
    float darken = 0.0;
    float3 testUp;
    float3 testDown;
    for (int offset = roundDist; offset > 0; offset--) {
        testUp = texture.sample(s, uv + (float2(float(offset) * -0.25, float(offset) * 1.0) / res.xy)).rgb;
        testUp = getPosterColor(testUp);
        testDown = texture.sample(s, uv + (float2(float(offset) * 0.25, float(offset) * -1.0) / res.xy)).rgb;
        testDown = getPosterColor(testDown);
        if (testUp.r != posterColor.r && testUp.g != posterColor.g && testUp.b != posterColor.b) {
            brighten = easeIn(1.0 - float(offset) / float(roundDist));
        }
        if (testDown.r != posterColor.r && testDown.g != posterColor.g && testDown.b != posterColor.b) {
            darken = easeIn(1.0 - float(offset) / float(roundDist));
        }
    }

    return float4(posterColor
                  + float3(brighten * roundAmount)
                  - float3(darken * roundAmount)
                  + (overlay / 14.0 - 0.07)
                  , 1.0);
}

// MARK: - Day62

// https://www.shadertoy.com/view/4tdSWr Clouds

float2 hash62(float2 p) {
    p = float2(dot(p, float2(127.1,311.7)), dot(p, float2(269.5,183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise62(float2 p) {
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;
    float2 i = floor(p + (p.x + p.y) * K1);
    float2 a = p - i + (i.x + i.y) * K2;
    float2 o = (a.x > a.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float2 b = a - o + K2;
    float2 c = a - 1.0 + 2.0 * K2;
    float3 h = max(0.5 - float3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    float3 n = h * h * h * h * float3(dot(a, hash62(i)), dot(b, hash62(i + o)), dot(c, hash62(i + 1.0)));
    return dot(n, float3(70.0));
}

float fbm62(float2 n) {
    const float2x2 m = float2x2(1.6,  1.2, -1.2, 1.6);
    float total = 0.0;
    float amplitude = 0.1;
    for (int i = 0; i < 7; i++) {
        total += noise62(n) * amplitude;
        n = m * n;
        amplitude *= 0.4;
    }
    return total;
}

fragment float4 shader_day62(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    const float cloudscale = 1.1;
    const float speed = 0.03;
    const float2x2 m = float2x2(1.6,  1.2, -1.2, 1.6);

    float2 p = pixPos.xy / res.xy;
    float2 uv = p * float2(res.x / res.y, 1.0);
    uv.y *= -1.0;

    float timeX = time * speed;
    float q = fbm62(uv * cloudscale * 0.5);

    float r = 0.0;
    uv *= cloudscale;
    uv -= q - timeX;
    float weight = 0.8;
    for (int i = 0; i < 8; i++){
        r += abs(weight * noise62(uv));
        uv = m * uv + timeX;
        weight *= 0.7;
    }

    float f = 0.0;
    uv = p * float2(res.x / res.y, 1.0);
    uv *= cloudscale;
    uv -= q - timeX;
    weight = 0.7;
    for (int i = 0; i < 8; i++) {
        f += weight * noise62(uv);
        uv = m * uv + timeX;
        weight *= 0.6;
    }

    f *= r + f;

    float c = 0.0;
    timeX = time * speed * 2.0;
    uv = p * float2(res.x / res.y, 1.0);
    uv *= cloudscale * 2.0;
    uv -= q - timeX;
    weight = 0.4;
    for (int i = 0; i < 7; i++) {
        c += weight * noise62(uv);
        uv = m * uv + timeX;
        weight *= 0.6;
    }

    float c1 = 0.0;
    timeX = time * speed * 3.0;
    uv = p * float2(res.x / res.y, 1.0);
    uv *= cloudscale * 3.0;
    uv -= q - timeX;
    weight = 0.4;
    for (int i = 0; i < 7; i++) {
        c1 += abs(weight * noise62(uv));
        uv = m * uv + timeX;
        weight *= 0.6;
    }

    c += c1;

    float3 skycolour = mix(float3(0.4, 0.7, 1.0), float3(0.2, 0.4, 0.6), p.y);
    float3 cloudcolour = float3(1.1, 1.1, 0.9) * clamp((0.5 + 0.3 * c), 0.0, 1.0);
    f = 0.2 + 8.0 * f * r;
    float3 result = mix(skycolour, clamp(0.5 * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));
    return float4(result, 1.0);
}

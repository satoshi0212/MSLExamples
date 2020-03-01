#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day50

// https://www.shadertoy.com/view/XsfcD8

#define CAMPOW 1.0
#define LOWREZ 4.0
#define CFULL 1.0
#define CHALF 0.8431372549
#define DITHER 1.0

float4 bmap(float3 c) {
    c = pow(c, float3(CAMPOW));
    if ((c.r > CHALF) || (c.g > CHALF) || (c.b > CHALF) ) {
        return float4(floor(c.rgb + float3(0.5)), 1.0);
    } else {
        return float4(min(floor((c.rgb / CHALF) + float3(0.5)), float3(1.0, 1.0, 1.0)), 0.0);
    }
}

float3 fmap(float4 c) {
    if (c.a >= 0.5) {
        return c.rgb * float3(CFULL, CFULL, CFULL);
    } else {
        return c.rgb * float3(CHALF, CHALF, CHALF);
    }
}

fragment float4 shader_day50(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 pv = floor(pixPos.xy / LOWREZ);
    float2 bv = floor(pv / 8.0) * 8.0;
    float2 sv = floor(res.xy / LOWREZ);
    float4 min_cs = float4(1.0, 1.0, 1.0, 1.0);
    float4 max_cs = float4(0.0, 0.0, 0.0, 0.0);
    float bright = 0.0;

    for (int py = 1; py < 8; py++) {
        for (int px = 0; px < 8; px++) {
            float4 cs = bmap((texture.sample(s, (bv + float2(px, py)) / sv).rgb));
            bright += cs.a;
            min_cs = min(min_cs, cs);
            max_cs = max(max_cs, cs);
        }
    }

    if (bright >= 24.0) {
        bright = 1.0;
    } else {
        bright = 0.0;
    }

    if (max_cs.r == min_cs.r && max_cs.g == min_cs.g && max_cs.b == min_cs.b) {
        min_cs.rgb = float3(0.0, 0.0, 0.0);
    }
    if (max_cs.r == 0.0 && max_cs.g == 0.0 && max_cs.b == 0.0) {
        bright = 0.0;
        max_cs.rgb = float3(0.0, 0.0, 1.0);
        min_cs.rgb = float3(0.0, 0.0, 0.0);
    }

    float3 c1 = fmap(float4(max_cs.rgb, bright));
    float3 c2 = fmap(float4(min_cs.rgb, bright));
    float3 cs = texture.sample(s, pv / sv).rgb;
    float3 d = (cs + cs) - (c1 + c2);
    float dd = d.r + d.g + d.b;

    if (mod(pv.x + pv.y, 2.0) == 1.0) {
        return float4(dd >= -(DITHER * 0.5) ? c1.r : c2.r,
                      dd >= -(DITHER * 0.5) ? c1.g : c2.g,
                      dd >= -(DITHER * 0.5) ? c1.b : c2.b,
                      1.0);
    } else {
        return float4(dd >= (DITHER * 0.5) ? c1.r : c2.r,
                      dd >= (DITHER * 0.5) ? c1.g : c2.g,
                      dd >= (DITHER * 0.5) ? c1.b : c2.b,
                      1.0);
    }
}

// MARK: - Day51

// https://www.shadertoy.com/view/MdXyzX

#define DRAG_MULT 0.048
#define ITERATIONS_RAYMARCH 4
#define ITERATIONS_NORMAL 24

float2 wavedx(float2 position, float2 direction, float speed, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift * speed;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return float2(wave, -dx);
}

float getwaves(float2 position, int iterations, float time) {
    float iter = 0.0;
    float phase = 6.0;
    float speed = 2.0;
    float weight = 1.0;
    float w = 0.0;
    float ws = 0.0;
    for (int i = 0; i < iterations; i++){
        float2 p = float2(sin(iter), cos(iter));
        float2 res = wavedx(position, p, speed, phase, time);
        position += normalize(p) * res.y * weight * DRAG_MULT;
        w += res.x * weight;
        iter += 12.0;
        ws += weight;
        weight = mix(weight, 0.0, 0.2);
        phase *= 1.18;
        speed *= 1.07;
    }
    return w / ws;
}

float raymarchwater(float3 camera, float3 start, float3 end, float depth, float time) {
    float3 pos = start;
    float h = 0.0;
    float3 dir = normalize(end - start);
    for (int i = 0; i < 318; i++) {
        h = getwaves(pos.xz * 0.1, ITERATIONS_RAYMARCH, time) * depth - depth;
        if (h + 0.01 > pos.y) {
            return distance(pos, camera);
        }
        pos += dir * (pos.y - h);
    }
    return -1.0;
}

float3 normal(float2 pos, float e, float depth, float time) {
    float2 ex = float2(e, 0);
    float H = getwaves(pos.xy * 0.1, ITERATIONS_NORMAL, time) * depth;
    float3 a = float3(pos.x, H, pos.y);
    return normalize(cross(normalize(a - float3(pos.x - e, getwaves(pos.xy * 0.1 - ex.xy * 0.1, ITERATIONS_NORMAL, time) * depth, pos.y)),
                           normalize(a - float3(pos.x, getwaves(pos.xy * 0.1 + ex.yx * 0.1, ITERATIONS_NORMAL, time) * depth, pos.y + e))));
}

float3x3 rotmat(float3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    return float3x3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

float3 getRay(float2 uv, float2 res, float2 tp) {
    uv = (uv * 2.0 - 1.0) * float2(res.x / res.y, 1.0);
    float3 proj = normalize(float3(uv.x, uv.y, 1.0) + float3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);
    if (res.x < 400.0) return proj;
    float3 ray = rotmat(float3(0.0, -1.0, 0.0), 3.0 * (tp.x * 2.0 - 1.0)) * rotmat(float3(1.0, 0.0, 0.0), 1.5 * (tp.y * 2.0 - 1.0)) * proj;
    return ray;
}

float intersectPlane(float3 origin, float3 direction, float3 point, float3 normal) {
    return clamp(dot(point - origin, normal) / dot(direction, normal), -1.0, 9991999.0);
}

float3 extra_cheap_atmosphere(float3 raydir, float3 sundir) {
    sundir.y = max(sundir.y, -0.07);
    float special_trick = 1.0 / (raydir.y * 1.0 + 0.1);
    float special_trick2 = 1.0 / (sundir.y * 11.0 + 1.0);
    float raysundt = pow(abs(dot(sundir, raydir)), 2.0);
    float sundt = pow(max(0.0, dot(sundir, raydir)), 8.0);
    float mymie = sundt * special_trick * 0.2;
    float3 suncolor = mix(float3(1.0), max(float3(0.0), float3(1.0) - float3(5.5, 13.0, 22.4) / 22.4), special_trick2);
    float3 bluesky= float3(5.5, 13.0, 22.4) / 22.4 * suncolor;
    float3 bluesky2 = max(float3(0.0), bluesky - float3(5.5, 13.0, 22.4) * 0.002 * (special_trick + -6.0 * sundir.y * sundir.y));
    bluesky2 *= special_trick * (0.24 + raysundt * 0.24);
    return bluesky2 * (1.0 + 1.0 * pow(1.0 - raydir.y, 3.0)) + mymie * suncolor;
}

float3 getatm(float3 ray) {
    return extra_cheap_atmosphere(ray, normalize(float3(1.0))) * 0.5;
}

float sun(float3 ray) {
    float3 sd = normalize(float3(1.0));
    return pow(max(0.0, dot(ray, sd)), 528.0) * 110.0;
}

float3 aces_tonemap(float3 color) {
    float3x3 m1 = float3x3(
                   0.59719, 0.07600, 0.02840,
                   0.35458, 0.90834, 0.13383,
                   0.04823, 0.01566, 0.83777
                   );
    float3x3 m2 = float3x3(
                   1.60475, -0.10208, -0.00327,
                   -0.53108,  1.10813, -0.07276,
                   -0.07367, -0.00605,  1.07602
                   );
    float3 v = m1 * color;
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), float3(1.0 / 2.2));
}

fragment float4 shader_day51(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             constant float2& touchedPosition[[buffer(3)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    float2 uv = pixPos.xy / res.xy;
    float2 tp = touchedPosition / res.xy;
    
    float waterdepth = 2.1;
    float3 wfloor = float3(0.0, -waterdepth, 0.0);
    float3 wceil = float3(0.0, 0.0, 0.0);
    float3 orig = float3(0.0, 2.0, 0.0);
    float3 ray = getRay(uv, res, tp);
    float hihit = intersectPlane(orig, ray, wceil, float3(0.0, 1.0, 0.0));
    if (ray.y >= -0.01) {
        float3 C = getatm(ray) * 2.0 + sun(ray);
        C = aces_tonemap(C);
        return float4(C, 1.0);
    }
    float lohit = intersectPlane(orig, ray, wfloor, float3(0.0, 1.0, 0.0));
    float3 hipos = orig + ray * hihit;
    float3 lopos = orig + ray * lohit;
    float dist = raymarchwater(orig, hipos, lopos, waterdepth, time);
    float3 pos = orig + ray * dist;

    float3 N = normal(pos.xz, 0.001, waterdepth, time);
    N = mix(float3(0.0, 1.0, 0.0), N, 1.0 / (dist * dist * 0.01 + 1.0));
    float3 R = reflect(ray, N);
    float fresnel = (0.04 + (1.0 - 0.04) * (pow(1.0 - max(0.0, dot(-N, ray)), 5.0)));

    float3 C = fresnel * getatm(R) * 2.0 + fresnel * sun(R);
    C = aces_tonemap(C);

    return float4(C, 1.0);
}

// MARK: - Day52

// https://www.shadertoy.com/view/4djGzz

float wave(float2 pos, float t, float freq, float numWaves, float2 center) {
    float d = length(pos - center);
    d = log(1.0 + exp(d));
    return 1.0 / (1.0 + 20.0 * d * d) * sin(2.0 * 3.1415 * (-numWaves * d + t * freq));
}

float height(float2 pos, float t) {
    float w = wave(pos, t, 2.5, 10.0, float2(0.5, -0.5));
    w += wave(pos, t, 2.5, 10.0, -float2(0.5, -0.5));
    return w;
//    return wave(pos, t, 2.5, 10.0, float2(0.0, 0.0));
}

float2 normal(float2 pos, float t) {
    return float2(height(pos - float2(0.01, 0), t) - height(pos, t),
                  height(pos - float2(0, 0.01), t) - height(pos, t));
}

fragment float4 shader_day52(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float2 uvn = 2.0 * uv - float2(1.0);
    uv += normal(uvn, time);
    return texture.sample(s, uv);
}

// MARK: - Day53

// https://www.shadertoy.com/view/Xl2cWz

fragment float4 shader_day53(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;
    if (mod(time, 8.0) > 4.0) {
        if (uv.x > 0.5)
            uv.x = 1.0 - uv.x;
        else
            uv.x = uv.x;
        return texture.sample(s, uv);
    } else {
        const float PIXEL_SIZE = 40.0;
        float2 m_uv = float2(uv.x >= 0.5 ? uv.x - 0.5 : uv.x, uv.y >= 0.5 ? uv.y - 0.5 : uv.y) * 2.0;
        float4 col = texture.sample(s, m_uv);
        if (uv.x <= 0.5 && uv.y <= 0.5) {
            float gray = col.r > 0.5
              ? 10.0 * (mod(pixPos.y / 5.0, 1.0) - 0.45)
              : 10.0 * (mod(pixPos.x / 5.0, 1.0) - 0.85);
            col = float4(float3(gray), 1.0);
        } else if (uv.x <= 0.5 && uv.y >= 0.5) {
            col = float4(0.0);
            float dx = 1.0 / res.x;
            float dy = 1.0 / res.y;
            m_uv.x = (dx * PIXEL_SIZE) * floor(m_uv.x / (dx * PIXEL_SIZE));
            m_uv.y = (dy * PIXEL_SIZE) * floor(m_uv.y / (dy * PIXEL_SIZE));
            for (int i = 0; i < int(PIXEL_SIZE); i++)
                for (int j = 0; j < int(PIXEL_SIZE); j++)
                    col += texture.sample(s, float2(m_uv.x + dx * float(i), m_uv.y + dy * float(j)));
            col /= pow(PIXEL_SIZE, 2.0);
        } else if (uv.x >= 0.5 && uv.y <= 0.5) {
            col = float4((col.r + col.g + col.b) / 3.0);
        } else {
            col = float4(1.0) - col;
        }
        return col;
    }
}

// MARK: - Day54

// https://www.shadertoy.com/view/XdfcWN

fragment float4 shader_day54(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]],
                             texture2d<float, access::sample> noiseTexture [[texture(2)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float zr = 1.0 - texture.sample(s, pixPos.xy / res.xy).x;

    float ao = 0.0;
    for (int i = 0; i < 8; i++) {
        float2 off = -1.0 + 2.0 * noiseTexture.sample(s, (pixPos.xy + 23.71 * float(i)) / res.xy).xz;
        float z = 1.0 - texture.sample(s, (pixPos.xy + floor(off * 16.0)) / res.xy).x;
        ao += clamp((zr - z) / 0.1, 0.0, 1.0);
    }
    ao = clamp(1.0 - ao / 8.0, 0.0, 1.0);

    float3 col = float3(ao);
    if (mod(time, 8.0) < 4.0) {
        col *= texture.sample(s, uv).xyz;
    }

    float3 og = texture.sample(s, uv).xyz;
    float3 one = float3(1.0);
    float3 two = float3(2.0);
    float3 point5 = float3(0.5);
    col = col.x > 0.5 ? one - (one - og) * (one - two * (col - point5)) : col * two * og;
    col.r = col.r > 0.5 ? 1.0 - (1.0 - og.r) * (1.0 - 2.0 * (col.r - 0.5)) : col.r * 2.0 * og.r;
    col.g = col.g > 0.5 ? 1.0 - (1.0 - og.g) * (1.0 - 2.0 * (col.g - 0.5)) : col.g * 2.0 * og.g;
    col.b = col.b > 0.5 ? 1.0 - (1.0 - og.b) * (1.0 - 2.0 * (col.b - 0.5)) : col.b * 2.0 * og.b;

    return float4(col, 1.0);
}

// MARK: - Day55

// https://www.shadertoy.com/view/Md2cRt 縦白黒ノイズ　2値化
// https://www.shadertoy.com/view/MtXXRf 斜めの白黒

float randX(float2 co) {
    return fract(sin(dot(co.xy, float2(12.9898,78.233))) * 43758.5453);
}

float randX2(float2 co) {
    return randX(co) - randX(float2(co.x + 1.0, co.y + 1.0)) / 2.0;
}

fragment float4 shader_day55(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;
    float COLOR_DEPTH = 1.5;
    float4 texValue = texture.sample(s, uv);
    float value = (texValue.r * 0.2125 + texValue.g * 0.715 + texValue.b * 0.0721);
    float t = mod(time, 9.0);
    if (t < 3.0) {
        float g = max(texValue.r, max(texValue.g, texValue.b)) * 2.0;
        float f = mod((uv.x + uv.y + 500.0) * (345.678 + 100.0), 1.0);
        (mod(g, 1.0) > f) ? texValue.r = ceil(g) / 2.0 : texValue.r = floor(g) / 2.0;
        return texValue.rrra;
    } else if (3.0 <= t && t < 6.0) {
        return float4(floor(value * (COLOR_DEPTH - 0.5) + randX2(pixPos.xy + time)) / (COLOR_DEPTH - 1.0),
                      floor(value * (COLOR_DEPTH - 0.5) + randX2(pixPos.xy + time)) / (COLOR_DEPTH - 1.0),
                      floor(value * (COLOR_DEPTH - 0.5) + randX2(pixPos.xy + time)) / (COLOR_DEPTH - 1.0),
                      1.0);
    } else {
        return float4(floor(value * COLOR_DEPTH) / (COLOR_DEPTH - 1.0),
                      floor(value * COLOR_DEPTH) / (COLOR_DEPTH - 1.0),
                      floor(value * COLOR_DEPTH) / (COLOR_DEPTH - 1.0),
                      1.0);
    }
}

// MARK: - Day56

// https://www.shadertoy.com/view/Xty3Rh

fragment float4 shader_day56(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float t = mod(time, 3.0);
    float _Speed = 15.0;
    float resx = floor((pow(t, 1.4)) * _Speed) * 2.0 + 0.01;

    uv *= res.xy / resx;
    uv = floor(uv);
    uv /= res.xy / resx;

    uv += resx * 0.002;
    float4 tex = texture.sample(s, uv);
    return tex * clamp(1.4 - t, 0.0, 1.0);
}

// MARK: - Day57

// https://www.shadertoy.com/view/MttXWl

float2 crtCurveUV(float2 uv) {
    uv = uv * 2.0 - 1.0;
    float2 offset = abs( uv.yx ) / float2( 6.0, 4.0 );
    uv = uv + uv * offset * offset;
    uv = uv * 0.5 + 0.5;
    return uv;
}

float3 drawVignette(float3 color, float2 uv) {
    float vignette = uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y);
    vignette = clamp(pow(16.0 * vignette, 0.3), 0.0, 1.0);
    color *= vignette;
    return color;
}

float3 drawScanline(float3 color, float2 uv, float time) {
    float scanline = clamp(0.95 + 0.05 * cos(3.14 * (uv.y + 0.008 * time) * 240.0 * 1.0), 0.0, 1.0);
    float grille = 0.85 + 0.15 * clamp(1.5 * cos(3.14 * uv.x * 640.0 * 1.0), 0.0, 1.0);
    color *= scanline * grille * 1.2;
    return color;
}

fragment float4 shader_day57(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float3 color = float3(92.0/ 255.0, 148.0/ 255.0, 252.0 / 255.0);
    float2 crtUV = crtCurveUV(uv);
    if (crtUV.x < 0.0 || crtUV.x > 1.0 || crtUV.y < 0.0 || crtUV.y > 1.0) {
        color = float3(0.0, 0.0, 0.0);
    }
    color = mix(color, texture.sample(s, uv).xyz, 0.5);
    color = drawVignette(color, crtUV);
    color = drawScanline(color, uv, time);
    return float4(color, 1.0);
}

// MARK: - Day58

// https://www.shadertoy.com/view/tltXWM Planet

#define NUM_NOISE_OCTAVES 10
#define PLANET_SIZE 0.75
#define inf 9999999.9

float hash(float p) { p = fract(p * 0.011); p *= p + 7.5; p *= p + p; return fract(p); }
float square(float x) { return x * x; }
float infIfNegative(float x) { return (x >= 0.0) ? x : inf; }

float noise(float3 x) {
    const float3 step = float3(110.0, 241.0, 171.0);
    float3 i = floor(x);
    float3 f = fract(x);
    float n = dot(i, step);
    float3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(n + dot(step, float3(0, 0, 0))), hash(n + dot(step, float3(1, 0, 0))), u.x),
                   mix(hash(n + dot(step, float3(0, 1, 0))), hash(n + dot(step, float3(1, 1, 0))), u.x), u.y),
               mix(mix(hash(n + dot(step, float3(0, 0, 1))), hash(n + dot(step, float3(1, 0, 1))), u.x),
                   mix(hash(n + dot(step, float3(0, 1, 1))), hash(n + dot(step, float3(1, 1, 1))), u.x), u.y), u.z);
}

float fbm(float3 x) {
    float v = 0.0;
    float a = 0.5;
    float3 shift = float3(100);
    for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
        v += a * noise(x);
        x = x * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

float intersectSphere(float3 C, float r, float3 P, float3 w) {
    float3 v = P - C;
    float b = -dot(w, v);
    float c = dot(v, v) - square(r);
    float d = (square(b) - c);
    if (d < 0.0) { return inf; }
    float dsqrt = sqrt(d);
    return min(infIfNegative((b - dsqrt)), infIfNegative((b + dsqrt)));
}

float max3(float3 v) {
    return max(max(v.x, v.y), v.z);
}

fragment float4 shader_day58(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float3 q = float3(0.0);
    float3 r = float3(0.0);
    float v = 0.0;
    float3 color = float3(0.0);

    float theta = time * 0.15;
    float3x3 rot = float3x3(cos(theta), 0, sin(theta),
                            0, 1, 0,
                            -sin(theta), 0, cos(theta));

    const float verticalFieldOfView = 25.0 * M_PI_F / 180.0;

    float3 P = float3(0.0, 0.0, 5.0);
    float3 w = normalize(float3(pixPos.xy - res.xy * 0.5, (res.y) / (-2.0 * tan(verticalFieldOfView / 2.0))));

    float t = intersectSphere(float3(0, 0, 0), PLANET_SIZE, P, w);

    if (t < inf) {
        float3 X = P + w * t;
        X = rot * X;
        q = float3(fbm(X + 0.025 * time), fbm(X), fbm(X));
        r = float3(fbm(X + 1.0 * q + 0.01 * time), fbm(X + q), fbm(X + q));
        v = fbm(X + 5.0 * r + time * 0.005);
    } else {
        return float4(float3(0.0), 1.0);
    }

    float3 col_top = float3(0.0, 0.5, 0.0);
    float3 col_bot = float3(0.0, 1.0, 1.0);
    float3 col_mid1 = float3(0.0, 1.0, 0.0);
    float3 col_mid2 = float3(0.0, 0.0, 1.0);
    float3 col_mid3 = float3(0.0, 0.0, 1.0);

    float3 col_mid = mix(col_mid1, col_mid2, clamp(r, 0.0, 1.0));
    col_mid = mix(col_mid, col_mid3, clamp(q, 0.0, 1.0));
    col_mid = col_mid;

    float pos = v * 2.0 - 1.0;
    color = mix(col_mid, col_top, clamp(pos, 0.0, 1.0));
    color = mix(color, col_bot, clamp(-pos, 0.0, 1.0));
    color = color / max3(color);
    color = (clamp((0.4 * pow(v, 3.0) + pow(v, 2.0) + 0.5 * v), 0.0, 1.0) * 0.9 + 0.1) * color;

    float diffuse = max(0.0, dot(P + w * t, float3(1.0, sqrt(0.5), 1.0)));
    float ambient = 0.1;
    color *= clamp((diffuse + ambient), 0.0, 1.0);

    color *= (P + w * t).z * 2.0;
    return float4(color, 1.0);
}

// MARK: - Day59

// https://www.shadertoy.com/view/ldX3R2 Worley noise

#define id(i, j, k) (float(128 + i) + 256.0 * float(128 + j) + 65536.0 * float(k))
#define id2cell(id) float3(mod(id, 256.0) - 128.0, mod(floor(id / 256.0), 256.0) - 128.0, id / 65536.0)

float noise59(float3 x) {
    float3 p = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    float res = mix(mix(mix(hash(n +   0.0), hash(n +   1.0), f.x),
                        mix(hash(n +  57.0), hash(n +  58.0), f.x), f.y),
                    mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                        mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
    return res;
}

float2 noise2_2(float2 p, float time) {
    float3x3 m = float3x3( 0.00,  0.80,  0.60,
                          -0.80,  0.36, -0.48,
                          -0.60, -0.48,  0.64);
    float3 pos = float3(p, 0.5);
    pos.z += time;
    pos *= m;
    float fx = noise59(pos);
    float fy = noise59(pos + float3(1345.67, 0.0, 45.67));
    return float2(fx, fy);
}

float4 sortd2(float tmp_d2, int i, int j, int k, float4 d2) {
    if (tmp_d2 < d2.x) {
        d2.yzw = d2.xyz; d2.x = tmp_d2;
    } else if (tmp_d2 < d2.y) {
        d2.zw = d2.yz; d2.y = tmp_d2;
    } else if (tmp_d2 < d2.z) {
        d2.w = d2.z; d2.z = tmp_d2;
    } else {
        d2.w = tmp_d2;
    }
    return d2;
}

float4 sortid(float tmp_d2, int i, int j, int k, float4 d2, float4 id) {
    if (tmp_d2 < d2.x) {
        id.yzw = id.xyz; id.x = id(i, j, k);
    } else if (tmp_d2 < d2.y) {
        id.zw = id.yz; id.y = id(i, j, k);
    } else if (tmp_d2 < d2.z) {
        id.w = id.z; id.z = id(i, j, k);
    } else {
        id.w = id(i, j, k);
    }
    return id;
}

float4 worley(float2 ip, float2 p, float4 id, float time) {
    float4 d2 = float4(1.e30);
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            float2 tmp_p = ip + float2(float(i), float(j));
            float2 tmp_pos = tmp_p + noise2_2(tmp_p, time) - p;
            float tmp_d2 = dot(tmp_pos, tmp_pos);
            d2 = sortd2(tmp_d2, i, j, 0, d2);
            id = sortid(tmp_d2, i, j, 0, d2, id);
        }
    }
    return sqrt(d2);
}

fragment float4 shader_day59(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]],
                             texture2d<float, access::sample> noiseTexture [[texture(2)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float time2 = 1.0 * (2.0 * time + 0.5 * sin(1.0 * time + 10.0 * length(uv)));

    float2 ip = floor(10.0 * uv);
    float4 id = float4(ip.x + 256.0 * ip.y);
    float4 D = worley(ip, 10.0 * uv, id, time2);

    uv -= 0.1 * noise2_2(id2cell(id.x).xy, time2);
    float c = pow(D.y - D.x, 0.1);
    float3 col = c * texture.sample(s, uv).rgb;

    return float4(col, 1.0);
}

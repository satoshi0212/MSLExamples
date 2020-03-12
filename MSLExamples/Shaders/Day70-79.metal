#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day70

// https://www.shadertoy.com/view/MtBGDR 紫メタリック

fragment float4 shader_day70(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float4 c = texture.sample(s, uv);
    c = sin(uv.x * 10.0 + c * cos(c * 6.28 + time + uv.x) * sin(c + uv.y + time) * 6.28) * 0.5 +0.5;
    c.b += length(c.rg);
    return c;
}

// MARK: - Day71

// https://www.shadertoy.com/view/Mdf3zr edge glow

float lookup(sampler s, float d, float2 p, float dx, float dy, float2 res, texture2d<float, access::sample> texture) {
    float2 uv = (p.xy + float2(dx * d, dy * d)) / res.xy;
    float4 c = texture.sample(s, uv);
    return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

fragment float4 shader_day71(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float d = sin(time * 5.0) * 0.5 + 1.5;
    float2 p = pixPos.xy;

    float gx = 0.0;
    gx += -1.0 * lookup(s, d, p, -1.0, -1.0, res, texture);
    gx += -2.0 * lookup(s, d, p, -1.0,  0.0, res, texture);
    gx += -1.0 * lookup(s, d, p, -1.0,  1.0, res, texture);
    gx +=  1.0 * lookup(s, d, p,  1.0, -1.0, res, texture);
    gx +=  2.0 * lookup(s, d, p,  1.0,  0.0, res, texture);
    gx +=  1.0 * lookup(s, d, p,  1.0,  1.0, res, texture);

    float gy = 0.0;
    gy += -1.0 * lookup(s, d, p, -1.0, -1.0, res, texture);
    gy += -2.0 * lookup(s, d, p,  0.0, -1.0, res, texture);
    gy += -1.0 * lookup(s, d, p,  1.0, -1.0, res, texture);
    gy +=  1.0 * lookup(s, d, p, -1.0,  1.0, res, texture);
    gy +=  2.0 * lookup(s, d, p,  0.0,  1.0, res, texture);
    gy +=  1.0 * lookup(s, d, p,  1.0,  1.0, res, texture);

    float g = gx * gx + gy * gy;
    float g2 = g * (sin(time) / 2.0 + 0.5);

    float4 col = texture.sample(s, p / res);
    col += float4(0.0, g, g2, 1.0);

    return col;
}


// MARK: - Day72

// https://www.shadertoy.com/view/4scBW8

#define FOV 2.0
#define HEX float2(1.0, 1.73205080757)

#define LIGHT_FREQ 0.3
#define LIGHT_COLOR float3(0.05, 0.2, 0.8)

float hash13(float3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// hexagonal tiling
float2 hexCenter(float2 p) {
    float2 centerA = (floor(p.xy * HEX) + 0.5) / HEX;
    float2 centerB = (floor((p.xy + HEX * 0.5) * HEX) + 0.5) / HEX - HEX * 0.5;
    float2 a = p.xy - centerA.xy;
    float2 b = p.xy - centerB.xy;
    return dot(a,a) < dot(b,b) ? centerA : centerB;
}

// control sphere height
float3 getSphereCenter(float2 c, float time) {
    return float3(c, sin(c.x - c.y * 4.3 + time) * 0.2);
}

// main distance function, returns distance and color
float4 de(float3 p, float3 dir, float r, float3 color, float time) {

    // translate and get the center
    p.xy += time;
    float2 center = hexCenter(p.xy);
    // find out where the red light is
    float red = floor(time * LIGHT_FREQ) + 0.5;
    float fRed = smoothstep(0.5, 0.0, abs(fract(time * LIGHT_FREQ) - 0.5));
    float3 centerRed = getSphereCenter(hexCenter(red/LIGHT_FREQ + float2(0.5, 1.5)), time);

    // accumulate distance and color
    float d = 9e9;
    color = float3(0.0);
    float colorAcc = 0.0;
    //for (int i = 0; i < 7; i++) {
    for (int i = 0; i < 2; i++) {
        float theta = float(i) * (2.0 * M_PI_F / 6.0);
        float2 offset = float2(sin(theta), cos(theta)) * min(1.0 / HEX.y, float(i));
        float3 sphere = getSphereCenter(center + offset, time);
        float3 inCenter = p - sphere;
        float len = length(inCenter);
        float3 norm = inCenter / len;
        float3 toRed = sphere - centerRed;

        // select the nearest sphere
        float dist = len - 0.3;
        d = min(d, dist);

        // colors and light
        float3 albedo = float3(sin(sphere.x * 90.0 + sphere.y * 80.0) * 0.45 + 0.5);
        float3 colorHere = float3(0);

        if (dot(toRed, toRed) < 0.001) {
            albedo = mix(albedo, float3(0.0), fRed);
            colorHere += LIGHT_COLOR * fRed * 4.0;
        } else {
            float3 lightDir = centerRed - p;
            float len = dot(lightDir, lightDir);
            lightDir *= rsqrt(len);
            float3 col = LIGHT_COLOR * fRed / (len + 1.0) * 2.0;
            colorHere += albedo * max(0.0, dot(norm, lightDir) + 0.5 / len) * col;
            colorHere += albedo * pow(max(0.0, dot(lightDir, reflect(dir, norm))), 8.0) * col;
        }

        const float3 lightDir = normalize(float3(1.0, -1.0, 3.0));
        colorHere += albedo * max(0.0, dot(lightDir, norm));
        colorHere += albedo * pow(max(0.0, dot(lightDir, reflect(dir, norm))), 8.0);

        // accumulate color across neighborhood
        float alpha = max(0.0001, smoothstep(r, -r, dist));
        color += colorHere * alpha;
        colorAcc += alpha;
    }

    color /= colorAcc;
    return float4(color, d);
}

fragment float4 shader_day72(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = (pixPos.xy - res.xy * 0.5) / res.y;
    uv.y *= -1.0;

    float3 from = float3(0, 0, 1.2);
    float3 dir = normalize(float3(uv, -1.0 / tan(FOV * 0.5)));

    float2x2 rot1 = rot(-0.85);
    float2x2 rot2 = rot(0.2);

    float2 diryz = dir.yz;
    diryz *= rot1;
    dir.y = diryz.r;
    dir.z = diryz.g;

    float2 dirxy = dir.xy;
    dirxy *= rot2;
    dir.x = dirxy.r;
    dir.y = dirxy.g;

    float focal = 2.5;
    float sinPix = sin(FOV / res.y);
    float4 acc = float4(0.0, 0.0, 0.0, 1.0);
    float3 dummy = float3(0.0);
    float4 ret1 = de(from, dir, 0.0, dummy, time) * hash13(float3(pixPos.xy, 1.0));
    float totdist = ret1.a;

    for (int i = 0; i < 20; i++) {
        float3 p = from + totdist * dir;
        float r = max(totdist * sinPix, abs((totdist - focal) * 0.1));

        float4 ret = de(p, dir, r, 0.0, time);
        float3 color = ret.rgb;
        float dist = ret.a;

        // cone trace the surface
        float alpha = smoothstep(r, -r, dist);
        acc.rgb += acc.a * (alpha * color);
        acc.a *= (1.0 - alpha);

        // hit a surface, stop
        if (acc.a < 0.01) break;
        totdist += max(abs(dist), r * 0.5);
    }

    float3 ret = 0.0;
    ret.rgb = clamp(acc.rgb, float3(0.0), float3(1.0));
    ret.rgb = pow(ret.rgb, float3(1.0 / 2.2));
    return float4(ret, 1.0);
}


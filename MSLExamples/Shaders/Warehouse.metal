#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Edges

fragment float4 shader_dayAA(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 uv = pixPos.xy / res.xy;
    float4 color = texture.sample(s, uv);
    float gray = length(color.rgb);
    //return float4(float3(step(0.06, length(float2(dfdx(gray), dfdy(gray))))), 1.0);
    return float4(float3(pow(length(float2(dfdx(gray), dfdy(gray))), 0.5)), 1.0);
}

// MARK: - Color wash machine

// http://glslsandbox.com/e#50791.0 Color wash machine

fragment float4 shader_day77_(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              constant float& time[[buffer(1)]]) {

    float3 p = float3(pixPos.xy / res.y, sin(time));
    for (int i = 0; i < 40; i++) {
        p.xzy = float3(1.3, 0.999, 0.7) * (abs((abs(p) / dot(p, p) - float3(1.0, 1.0, cos(time) * 0.5))));
    }
    return float4(p, 1.0);
}

// https://www.shadertoy.com/view/WdsyRM Wave in a box
// https://www.shadertoy.com/view/4dfGzs Voxel Edges

// MARK: - Day81

// https://www.shadertoy.com/view/ltffzl Rain

#define S(a, b, t) smoothstep(a, b, t)

float3 N13(float p) {
    float3 p3 = fract(float3(p) * float3(0.1031, 0.11369, 0.13787));
    p3 += dot(p3, p3.yzx + 19.19);
    return fract(float3((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x));
}

float4 N14(float t) {
    return fract(sin(t * float4(123.0, 1024.0, 1456.0, 264.0)) * float4(6547.0, 345.0, 8799.0, 1564.0));
}
float N(float t) {
    return fract(sin(t * 12345.564) * 7658.76);
}

float Saw(float b, float t) {
    return S(0.0, b, t) * S(1.0, b, t);
}

float2 DropLayer2(float2 uv, float t) {
    float2 UV = uv;

    uv.y += t * 0.75;
    float2 a = float2(6.0, 1.0);
    float2 grid = a * 2.0;
    float2 id = floor(uv * grid);

    float colShift = N(id.x);
    uv.y += colShift;

    id = floor(uv * grid);
    float3 n = N13(id.x * 35.2 + id.y * 2376.1);
    float2 st = fract(uv * grid) - float2(0.5, 0.0);

    float x = n.x - 0.5;

    float y = UV.y * 20.0;
    float wiggle = sin(y + sin(y));
    x += wiggle * (0.5 - abs(x)) * (n.z - 0.5);
    x *= 0.7;
    float ti = fract(t + n.z);
    y = (Saw(0.85, ti) - 0.5) * 0.9 + 0.5;
    float2 p = float2(x, y);

    float d = length((st - p) * a.yx);

    float mainDrop = S(0.4, 0.0, d);

    float r = sqrt(S(1.0, y, st.y));
    float cd = abs(st.x - x);
    float trail = S(0.23 * r, 0.15 * r * r, cd);
    float trailFront = S(-0.02, 0.02, st.y - y);
    trail *= trailFront * r * r;

    y = UV.y;
    float trail2 = S(0.2 * r, 0.0, cd);
    float droplets = max(0.0, (sin(y * (1.0 - y) * 120.0) - st.y)) * trail2 * trailFront * n.z;
    y = fract(y * 10.0) + (st.y - 0.5);
    float dd = length(st - float2(x, y));
    droplets = S(0.3, 0.0, dd);
    float m = mainDrop + droplets * r * trailFront;

    return float2(m, trail);
}

float StaticDrops(float2 uv, float t) {
    uv *= 40.0;

    float2 id = floor(uv);
    uv = fract(uv) - 0.5;
    float3 n = N13(id.x * 107.45 + id.y * 3543.654);
    float2 p = (n.xy - 0.5) * 0.7;
    float d = length(uv - p);

    float fade = Saw(0.025, fract(t + n.z));
    float c = S(0.3, 0.0, d) * fract(n.z * 10.0) * fade;
    return c;
}

float2 Drops(float2 uv, float t, float l0, float l1, float l2) {
    float s = StaticDrops(uv, t) * l0;
    float2 m1 = DropLayer2(uv, t) * l1;
    float2 m2 = DropLayer2(uv * 1.85, t) * l2;

    float c = s + m1.x + m2.x;
    c = S(0.3, 1.0, c);

    return float2(c, max(m1.y * l0, m2.y * l1));
}

fragment float4 shader_day81(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::nearest);

    float2 uv = (pixPos.xy - 0.5 * res.xy) / res.y;
    uv.y *= -1.0;
    float2 UV = pixPos.xy / res.xy;

    float T = time;
    float t = T * 0.2;

    float rainAmount = sin(T * 0.05) * 0.3 + 0.7;

    float maxBlur = mix(3.0, 6.0, rainAmount);
    float minBlur = 2.0;

    float staticDrops = S(-0.5, 1.0, rainAmount) * 2.0;
    float layer1 = S(0.25, 0.75, rainAmount);
    float layer2 = S(0.0, 0.5, rainAmount);

    float2 c = Drops(uv, t, staticDrops, layer1, layer2);
    float2 e = float2(0.001, 0.0);
    float cx = Drops(uv + e, t, staticDrops, layer1, layer2).x;
    float cy = Drops(uv + e.yx, t, staticDrops, layer1, layer2).x;
    float2 n = float2(cx - c.x, cy - c.x);

    float focus = mix(maxBlur - c.y, minBlur, S(0.1, 0.2, c.x));
    float3 col = texture.sample(s, UV + n, focus).rgb;

    t = (T + 3.0) * 0.5;
    float colFade = sin(t * 0.2) * 0.5 + 0.5;
    col *= mix(float3(1.0), float3(0.8, 0.9, 1.3), colFade);
    float fade = S(0.0, 10.0, T);
    //    float lightning = sin(t*sin(t*10.));                // lighting flicker
    //    lightning *= pow(max(0., sin(t+sin(t))), 10.);        // lightning flash
    //    col *= 1.+lightning*fade*mix(1., .1, story*story);    // composite lightning

    col *= fade;

    return float4(col, 1.0);
}

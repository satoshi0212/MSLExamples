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

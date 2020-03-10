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

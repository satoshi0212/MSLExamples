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

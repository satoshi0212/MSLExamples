#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Ex001

// https://www.shadertoy.com/view/lss3R8 Toon

#define FILTER_SIZE 3
#define COLOR_LEVELS 7.0
#define EDGE_FILTER_SIZE 3
#define EDGE_THRESHOLD 0.05

float4 edgeFilter(int px, int py, float2 fragCoord, float2 res, texture2d<float, access::sample> texture, sampler s) {
    float4 color = float4(0.0);
    for (int y = -EDGE_FILTER_SIZE; y <= EDGE_FILTER_SIZE; ++y) {
        for (int x = -EDGE_FILTER_SIZE; x <= EDGE_FILTER_SIZE; ++x) {
            color += texture.sample(s, (fragCoord + float2(px + x, py + y)) / res);
        }
    }
    color /= float((2 * EDGE_FILTER_SIZE + 1) * (2 * EDGE_FILTER_SIZE + 1));
    return color;
}

fragment float4 shader_Ex001(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float4 color = float4(0.0);

    float2 uv = pixPos.xy / res.xy;
    color += texture.sample(s, uv);

//    for (int y = -FILTER_SIZE; y <= FILTER_SIZE; ++y) {
//        for (int x = -FILTER_SIZE; x <= FILTER_SIZE; ++x) {
//            color += texture.sample(s, (pixPos.xy + float2(x, y)) / res);
//        }
//    }
//
//    color /= float((2 * FILTER_SIZE + 1) * (2 * FILTER_SIZE + 1));

    for (int c = 0; c < 3; ++c) {
        color[c] = floor(COLOR_LEVELS * color[c]) / COLOR_LEVELS;
    }

    float4 sum = abs(edgeFilter(0, 1, pixPos.xy, res, texture, s) - edgeFilter(0, -1, pixPos.xy, res, texture, s));
    sum += abs(edgeFilter(1, 0, pixPos.xy, res, texture, s) - edgeFilter(-1, 0, pixPos.xy, res, texture, s));
    sum /= 2.0;

    if (length(sum) > EDGE_THRESHOLD) {
        color.rgb = float3(0.0);
    }

    return color;
}

// MARK: - Ex002

// https://www.shadertoy.com/view/XlS3zt postalize

fragment float4 shader_Ex002(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 uv = pixPos.xy / res.xy;
    uv = float2(floor(uv.x * 320.0) / 320.0, floor(uv.y * 240.0) / 240.0);
    float4 samplrr = texture.sample(s, uv);
    float4 samplrr2 = texture.sample(s, uv + float2(1./ 320.0,0.0));
    float4 samplrr3 = texture.sample(s, uv + float2(0.0,1./ 240.0));
    float4 samplrr4 = texture.sample(s, uv + float2(-1./ 320.0,0.0));
    float4 samplrr5 = texture.sample(s, uv + float2(0.0,-1./ 240.0));

    float I = floor(length(samplrr.rgb) + 0.5) * 0.5 + 1.2;
    float3 C = float3(
                floor(samplrr.r * 3.0) / 3.0 * I,
                floor(samplrr.g * 3.0) / 3.0 * I,
                floor(samplrr.b * 3.0) / 3.0 * I
                );
    float border = floor(distance(samplrr2, samplrr) + distance(samplrr3, samplrr) + distance(samplrr4, samplrr) + distance(samplrr5, samplrr) + 0.73);
    uv.x *= 0.6 + sin(uv.y / 7.0 + time) / 3.0;
    uv.y *= 0.3 + sin(uv.x + time) / 5.0;
    float3 finalColor = C * (1.0 - float3(border));

    return float4(finalColor, 1.0);
}

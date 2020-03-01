#include <metal_stdlib>

using namespace metal;

vertex float4 vertexShader(unsigned int vid [[ vertex_id ]]) {
    const float4x4 vertices = float4x4(float4(-0.5, -0.5, 0.0, 0.5),
                                       float4( 0.5, -0.5, 0.0, 0.5),
                                       float4(-0.5,  0.5, 0.0, 0.5),
                                       float4( 0.5,  0.5, 0.0, 0.5));
    return vertices[vid];
}

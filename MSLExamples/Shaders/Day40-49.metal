#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day40

// https://www.shadertoy.com/view/tttXD8

float rand(float2 p, float time) {
    float t = floor(time * 50.0) / 10.0;
    return fract(sin(dot(p, float2(t * 12.9898, t * 78.233))) * 43758.5453);
}

float noise(float2 uv, float blockiness, float time) {
    float2 lv = fract(uv);
    float2 id = floor(uv);
    float n1 = rand(id, time);
    float n2 = rand(id + float2(1.0, 0.0), time);
    float n3 = rand(id + float2(0.0, 1.0), time);
    float n4 = rand(id + float2(1.0, 1.0), time);
    float2 u = smoothstep(0.0, 1.0 + blockiness, lv);
    return mix(mix(n1, n2, u.x), mix(n3, n4, u.x), u.y);
}

float fbm(float2 uv, int count, float blockiness, float complexity, float time) {
    float val = 0.0;
    float amp = 0.5;
    while(count != 0) {
        val += amp * noise(uv, blockiness, time);
        amp *= 0.5;
        uv *= complexity;
        count--;
    }
    return val;
}

fragment float4 shader_day40(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 uv = pixPos.xy / res.xy;
    float2 a = float2(uv.x * (res.x / res.y), uv.y);
    float2 uv2 = float2(a.x / res.x, exp(a.y));
    float2 id = floor(uv * 20.0);

    float shift = 0.9 * pow(fbm(uv2, int(rand(id) * 6.0), 10000.0, 400.0, time), 10.0);

    float scanline = abs(cos(uv.y * 400.0));
    scanline = smoothstep(0.0, 2.0, scanline);
    shift = smoothstep(0.00001, 0.2, shift);

    float colR = texture.sample(s, float2(uv.x + shift, uv.y)).r * (1. - shift);
    float colG = texture.sample(s, float2(uv.x - shift, uv.y)).g * (1. - shift) + rand(id) * shift;
    float colB = texture.sample(s, float2(uv.x - shift, uv.y)).b * (1. - shift);

    float3 f = float3(colR, colG, colB) - (0.1 * scanline);

    return float4(f, 1.0);
}

// MARK: - Day41

// https://www.shadertoy.com/view/XldBzj

fragment float4 shader_day41(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float2 scaled_uv = uv * 75.0;
    float2 begin = floor(scaled_uv) / 75.0;
    float grey = pow(length(texture.sample(s, begin).rgb) / sqrt(3.0), 0.66);
    grey = 1.0 - grey;
    grey = 75.0 * distance(uv, begin + float2(0.5 / 75.0, 0.5 / 75.0)) > grey ? 1.0 : 0.0;
    return float4(float3(250.0, 242.0, 228.0) / 256.0 * grey, 1.0);
}

// MARK: - Day42

// https://www.shadertoy.com/view/lljBWy

float3 permute(float3 x) {
    return mod(x * x * 34.0 + x, 289.0);
}

float snoise(float2 v) {
    float2 i = floor((v.x + v.y) * 0.36602540378443 + v);
    float2 x0 = (i.x + i.y) * 0.211324865405187 + v - i;
    float s = step(x0.x, x0.y);
    float2 j = float2(1.0 - s, s);
    float2 x1 = x0 - j + 0.211324865405187;
    float2 x3 = x0 - 0.577350269189626;
    float3 p = permute(permute(i.y + float3(0, j.y, 1)) + i.x + float3(0, j.x, 1));
    float3 m = max(0.5 - float3(dot(x0,x0), dot(x1, x1), dot(x3, x3)), 0.0);
    float3 x = fract(p * 0.024390243902439) * 2.0 - 1.0;
    float3 h = abs(x) - 0.5;
    float3 a0 = x - floor(x + 0.5);
    return 0.5 + 65.0 * dot(pow(m, float3(4.0)) * (-0.85373472095314 * (a0 * a0 + h * h) + 1.79284291400159),
                            a0 * float3(x0.x, x1.x, x3.x) + h * float3(x0.y, x1.y, x3.y));
}

fragment float4 shader_day42(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler sa(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float t = time * 0.8;
    float s = smoothstep(0.4, 1.0, uv.x);
    uv.x += s * snoise(uv * (4.3 * (s / 3.7 + 1.2)) - float2(t * 1.2, 0.0));
    return texture.sample(sa, uv);
}

// MARK: - Day43

// https://www.shadertoy.com/view/4dyGWm

fragment float4 shader_day43(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    int random = 3;
    int a = 1103515245;
    int c = 12345;
    float m = 2147483648.0;
    float2 o;
    float minDist = 10000000.0;
    float4 ret = 0.0;
    float dis;

    for (int i = 0; i < 1024; i++) {
        random = a * random + c;
        o.x = (float(random) / m) * res.x;
        random = a * random + c;
        o.y = (float(random) / m) * res.y;
        dis = distance(pixPos.xy, o);
        if (dis < minDist) {
            minDist = dis;
            float2 uv = o / res.xy;
            ret = (texture.sample(s, uv)) * (1.0 - minDist / 200.0);
        }
    }
    return ret;
}

// MARK: - Day44

// https://www.shadertoy.com/view/XdGyDK

fragment float4 shader_day44(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float3 col = 0.5 + 0.5 * cos(time + uv.xyx + float3(0.0, 2.0, 4.0));

    float stheta = sin(0.1 * time);
    float ctheta = cos(0.1 * time);
    float2 rot_uv = float2x2(ctheta, stheta, -stheta, ctheta) * uv;
    float2 warped_uv = uv + float2(0.25 * sin(3.0 * rot_uv.x), 0.05 * cos(4.5 * rot_uv.y));
    float wiggle = sin(400.0 * dot(float2(0.6, 0.8), warped_uv));
    float intense = dot(float3(0.58), texture.sample(s, float2(uv.x, uv.y), 2.0).rgb);

    return float4(col * clamp(wiggle - 1.0 + 2.0 * intense, 0.0, 1.0), 1.0);
}

// MARK: - Day45

// https://www.shadertoy.com/view/ldjXRt

fragment float4 shader_day45(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;
    int intensity_count[8];
    for (int i = 0; i < 8; ++i) {
        intensity_count[i] = 0;
    }
    for (float x = -2.0; x < 2.0; ++x) {
        for (float y = -2.0; y < 2.0; ++y) {
            float2 abs_pos = float2(x, y);
            if (4.0 < dot(abs_pos, abs_pos))
                continue;
            float2 pos = (abs_pos / res.xy) + uv;
            float4 col_element = texture.sample(s, pos);
            int current_intensity = int((dot(col_element, float4(1.0, 1.0, 1.0, 0.0)) / 3.0) * float(8.0));
            current_intensity = (current_intensity >= 8) ? 8 - 1 : current_intensity;
            for (int i = 0; i < 8; ++i) {
                if (i == current_intensity) {
                    intensity_count[i] += 1;
                    break;
                }
            }
        }
    }
    int max_level = 0;
    float val = 0.0;
    float4 col_out = float4(0.0, 0.0, 0.0, 1.0);
    for (int level = 0; level < 8; ++level) {
        if (intensity_count[level] > max_level) {
            max_level = intensity_count[level];
            val = float(max_level) / (3.14 * 4.0);
            col_out = float4(val, val, val, 1.0);
        }
    }
    return col_out;
}

// MARK: - Day46

// https://www.shadertoy.com/view/XssGD7

fragment float4 shader_day46(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float offset = 1.0 / 512.0;
    float3 o = float3(-offset, 0.0, offset);
    float4 gx = float4(0.0);
    float4 gy = float4(0.0);
    float4 t;
    gx += texture.sample(s, uv + o.xz);
    gy += gx;
    gx += 2.0 * texture.sample(s, uv + o.xy);
    t = texture.sample(s, uv + o.xx);
    gx += t;
    gy -= t;
    gy += 2.0 * texture.sample(s, uv + o.yz);
    gy -= 2.0 * texture.sample(s, uv + o.yx);
    t = texture.sample(s, uv + o.zz);
    gx -= t;
    gy += t;
    gx -= 2.0 * texture.sample(s, uv + o.zy);
    t = texture.sample(s, uv + o.zx);
    gx -= t;
    gy -= t;
    float4 grad = sqrt(gx * gx + gy * gy);

    return float4(grad);
}

fragment float4 shader_day46r(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float offset = 1.0 / 512.0;
    float3 o = float3(-offset, 0.0, offset);
    float4 gx = float4(0.0);
    float4 gy = float4(0.0);
    float4 t;
    gx += texture.sample(s, uv + o.xz);
    gy += gx;
    gx += 2.0 * texture.sample(s, uv + o.xy);
    t = texture.sample(s, uv + o.xx);

    gx += t;
    gy -= t;
    gy += 2.0 * texture.sample(s, uv + o.yz);
    gy -= 2.0 * texture.sample(s, uv + o.yx);
    t = texture.sample(s, uv + o.zz);

    gx -= t;
    gy += t;
    gx -= 2.0 * texture.sample(s, uv + o.zy);
    t = texture.sample(s, uv + o.zx);

    gx -= t;
    gy -= t;
    float4 grad = sqrt(gx * gx + gy * gy);

    return float4(grad);
}

// MARK: - Day47

// https://www.shadertoy.com/view/3lcSRs

fragment float4 shader_day47(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    float3 colors[] = {
                       float3(0.2, 0.0, 0.0),
                       float3(1.0, 0.0, 0.0),
                       float3(0.0, 1.0, 0.0),
                       float3(0.3, 0.3, 0.6),
                       float3(0.0, 0.0, 0.8)
    };

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float4 col = texture.sample(s, uv);
    float color = (col.x + col.y + col.z) / 3.0;
    int index = int(color * 5.0);
    col = float4(colors[index].xyz, 1);
    return float4(col.xyz, 1.0);
}

// MARK: - Day48

// https://www.shadertoy.com/view/4tKyR1

#define OUTLINE_COLOR float4(0.2)
#define OUTLINE_STRENGTH 20.0
#define OUTLINE_BIAS -0.5
#define OUTLINE_POWER 1.0
#define PRECISION 8.0

fragment float4 shader_day48(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler sa(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float4 ogColor = texture.sample(sa, uv);
    float4 texColor = ogColor;

    if (texColor.r + texColor.g + texColor.b >= 1.0) {
        texColor = float4(1.0, 1.0, 1.0, 1.0);
    } else {
        texColor = float4(0.0, 0.0, 0.0, 1.0);
    }
    texColor.b = (uv.x + uv.y) / 2.0;

    float2 r = res.xy;
    float4 p = pow(texture.sample(sa, pixPos.xy / r), float4(2.2));
    float4 s = pow(texture.sample(sa, (pixPos.xy + 0.5) / r), float4(2.2));
    float l = clamp(pow(length(p - s), OUTLINE_POWER) * OUTLINE_STRENGTH + OUTLINE_BIAS, 0.0, 1.0);
    p = floor(pow(p, float4(1.0 / 2.2)) * (PRECISION + 0.999)) / PRECISION;
    float4 texColor2 = mix(p, OUTLINE_COLOR, l);

    return texColor * texColor2 + texColor2 * 0.5;
}

// MARK: - Day49

// https://www.shadertoy.com/view/XsX3z8

fragment float4 shader_day49(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 p = pixPos.xy / res.xy;

    float4 col = texture.sample(s, p);

    if (p.x < 0.25) {
        col = float4((col.r + col.g + col.b) / 3.0);
    } else if (p.x < 0.5) {
        col = float4(1.0) - texture.sample(s, p);
    } else if (p.x < 0.75) {
        float2 offset = float2(0.01, 0.0);
        col.r = texture.sample(s, p + offset.xy).r;
        col.g = texture.sample(s, p).g;
        col.b = texture.sample(s, p + offset.yx).b;
    } else {
        col.rgb = texture.sample(s, p).brg;
    }

    if (mod(abs(p.x + 0.5 / res.y), 0.25) < 1.0 / res.y )
        col = float(1.0);

    return  col;
}

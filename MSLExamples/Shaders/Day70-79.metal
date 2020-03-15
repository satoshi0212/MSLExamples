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

// MARK: - Day73

// https://www.shadertoy.com/view/wlGXWt Heart beat

float remap01(float t, float a, float b) {
    return (t - a) / (b - a);
}

float Circle(float2 uv, float2 position, float radius, float blur) {
    float distance = length(uv - position);
    return smoothstep(radius, radius - blur, distance);
}

float gridLines(float t, float lines) {
    return step(fract(t * lines), 0.05);
}

float3 Ring(float2 uv, float2 position) {
    float ring = Circle(uv, position, 0.08, 0.01);
    ring -= Circle(uv, position, 0.065, 0.01);
    return float3(0.10, 0.24, 0.25) * ring;
}

float spike(float x, float d, float w, float raiseBy) {
    float f1 = pow(abs(x + (d * 2.0)), raiseBy);
    return exp(-f1 / w);
}

float generateEGC(float x) {
    x -= 0.5 * 2.0;

    float a = 0.4 * 2.0;
    float d = 0.3;
    float w = 0.001;

    float f1 = a * spike(x, d, w, 2.0);
    float f2 = a * spike(x, d - 0.1, 2.0 * w, 2.0);
    float f3 = a * 0.7 * spike(x, d - 0.3, 0.002, 2.0);
    float f3a = 0.15 * spike(x, d - 0.37, 0.0001, 4.0);
    float f4 = 0.25 * spike(x, d - 0.5, 0.005, 2.0);
    float f5 = 0.1 * spike(x, d - 0.75, 0.0001, 4.0);

    float f6 = a * spike(x, d - 1.0, 0.002, 2.0);
    float f7 = 0.5 * spike(x, d - 1.1, w, 2.0);

    float f8 = 0.1 * spike(x, d - 1.3, 0.0001, 4.0);
    float f9 = 0.1 * spike(x, d - 1.45, 0.0001, 4.0);

    return f1 - f2 + f3 + f3a - f4 + f5 + f6 - f7 - f8 + f9;
}

float getDotXPosition(float time) {
    float dotX = fract(time / 5.0);
    dotX *= 2. * 2.0;
    return dotX;
}

float3 MovingDot(float2 uv, float2 dotPosition) {
    float movingDot = Circle(uv, dotPosition, 0.015, 0.01);
    float smallBlurredDot = Circle(uv, dotPosition, 0.06, 0.1);
    float bigBlurredDot = Circle(uv, dotPosition, 0.3, 0.6);

    float3 color = float3(1.0, 1.0, 1.0) * movingDot;
    color += float3(0.15, 0.68, 0.83) * smallBlurredDot;
    color += float3(0.15, 0.68, 0.83) * bigBlurredDot;
    color += Ring(uv, dotPosition);

    return color;
}

fragment float4 shader_day73(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = pixPos.xy / res.xy;
    uv *= 2.0;
    uv.y -= .5 * 2.0;
    uv.x *= res.x / res.y;

    float grid = gridLines(uv.x, 6.0) + gridLines(uv.y, 6.0);
    float3 color = float3(0.01, 0.07, 0.06) * grid;

    float dotX = getDotXPosition(time);
    float2 dotPosition = float2(dotX, generateEGC(dotX));

    color += MovingDot(uv, dotPosition);

    for (int i = 1; i < 24; i++) {
        float delayedX = dotX - (float(i) * 0.042);
        float2 trailPosition = float2(delayedX, generateEGC(delayedX));

        float trail = Circle(uv, trailPosition, 0.028, 0.1);
        float trailBlur = Circle(uv, trailPosition, 0.06, 0.5);

        float q = 1.0 - remap01(float(i), 1.0, float(20));

        color += (float3(1.0, 1.0, 1.0) * q) * trail;
        color += trailBlur * (float3(0.15, 0.68, 0.83) * q);
    }

    return float4(color, 1.0);
}

// MARK: - Day74

// https://www.shadertoy.com/view/XdcGD4 マトリックス的

#define N_CHARS 8.0   // how many characters are in the character image
#define Y_PIXELS 18.0 // reduce input image to this many mega-pixels across
#define DROP_SPEED 0.15
#define ASPECT 2.7    // aspect ratio of input webcam image

#define MIN_DROP_SPEED 0.2 // range 0-1.  is added to column speeds to avoid stopped columns.
#define STATIC_STRENGTH 0.1 // range 0-1.  how intense is the tv static
#define SCANLINE_STRENGTH 0.4 // range 0-1.  how dark are the tv scanlines
#define NUM_SCANLINES 70.0 // how many scanlines

float rand2d(float2 v) {
    return fract(sin(dot(v.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float rand74(float x) {
    return fract(sin(x) * 3928.2413);
}

fragment float4 shader_day74(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]],
                             texture2d<float, access::sample> noiseTexture [[texture(2)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float xPix = floor(uv.x * Y_PIXELS * ASPECT) / Y_PIXELS / ASPECT;
    float yOffs = mod(time * DROP_SPEED * (rand74(xPix) + MIN_DROP_SPEED), 1.0);
    float yPix = floor((uv.y + yOffs) * Y_PIXELS) / Y_PIXELS - yOffs;
    float2 uvPix = float2(xPix, yPix);

    float4 pixelColor = texture.sample(s, uvPix);

    float2 uvInPix = float2(
                            mod(uv.x * Y_PIXELS, 1.0),
                            mod((uv.y + yOffs) * Y_PIXELS, 1.0)
                            );

    float charOffset = floor(pixelColor.r * N_CHARS) / N_CHARS;
    uvInPix.x = uvInPix.x / N_CHARS + charOffset;
    float4 charColor = noiseTexture.sample(s, uvInPix);

    float result = charColor.r * pixelColor.r;

    result *= 1.0 - SCANLINE_STRENGTH * (sin(uv.y * NUM_SCANLINES * 3.14159 * 2.0) / 2.0 + 0.5);

    float4 outColor = float4(
                             max(0.0, result * 3.0 - 1.2),
                             result * 1.6,
                             max(0.0, result * 3.0 - 1.5),
                             1.0
                             );

    float stat = rand2d(uv * float2(0.0005,1.0) + time * 0.1) * STATIC_STRENGTH;
    outColor += stat;

    return outColor;
}

// MARK: - Day75

// https://www.shadertoy.com/view/Mtf3Dl Strong colored

fragment float4 shader_day75(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;
    float4 tex = texture.sample(s, uv);

    float TAU = 6.28;
    float freq1 = 10.3;
    float freq2 = 22.2;
    float freq3 = 10.7;

    return ((cos(abs(tex) * TAU * freq1) + 1.0) / 2.0)
    * ((cos(abs(tex) * TAU * freq2) + 1.0) / 2.0)
    * ((sin(abs(tex) * TAU * freq3) + 1.0) / 2.0);
}

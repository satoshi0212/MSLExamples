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

// MARK: - Day76

// https://www.shadertoy.com/view/ldfSW2 Shake

float hash1(float x) {
    return fract(sin(x * 11.1753) * 192652.37862);
}

float nse1(float x) {
    float fl = floor(x);
    return mix(hash1(fl), hash1(fl + 1.0), smoothstep(0.0, 1.0, fract(x)));
}

fragment float4 shader_day76(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler sa(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float te = time * 9.0 / 16.0;
    float s = time * 50.0;
    uv += (float2(nse1(s), nse1(s + 11.0)) * 2.0 - 1.0) * exp(-5.0 * fract(te * 4.0)) * 0.1;
    return float4(texture.sample(sa, uv).xyz, 1.0);
}

// MARK: - Day77

// https://www.shadertoy.com/view/wtG3WR Shippou

float2 rotate(float2 pos, float angle) {
    float s = sin(angle), c = cos(angle);
    return float2x2(c, -s, s, c) * pos;
}

float hash77(float2 n) {
    return fract(sin(dot(n, float2(123.0, 458.0))) * 43758.5453);
}

float cubicInOut(float time) {
    return (time < 0.5) ? (4.0 * time * time * time) : (1.0 - 4.0 * pow(1.0 - time, 3.0));
}

fragment float4 shader_day77(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              constant float& time[[buffer(1)]]) {

    float2 uv = (pixPos.xy * 2.0 - res.xy) / min(res.x, res.y);
    uv += float2(1.0, -1.0) * time * 0.05;

    float uvScale = 2.5;
    float2 uvID = floor(uv * uvScale);
    float2 uvID2 = floor(uv * uvScale + float2(0.5));
    float2 uvLocal = fract(uv * uvScale);

    float timeX = time * 1.0;
    float timef = cubicInOut(fract(timeX));
    float timei = floor(timeX);
    timeX = (timef + timei) * 0.5;

    // animation 1
    float rotDir = (0.0 == mod(uvID.y, 2.0) ? 1.0 : -1.0);
    float2 rotCenter = float2(0.5, 0.5);
    float2 uvAnim1 = uvLocal;
    uvAnim1 = rotate(uvAnim1 - rotCenter, timeX * M_PI_F * rotDir);
    uvAnim1 += rotCenter;

    // animation 2
    float2 uvAnim2 = uvLocal;
    //float uvAnim2Corner = floor(uvLocal.x);
    uvAnim2 += float2((0.5 < uvAnim2.x ? -0.5 : 0.5), (0.5 < uvAnim2.y ? -0.5 : 0.5));
    uvAnim2 = rotate(uvAnim2 - rotCenter, timeX * M_PI_F);
    uvAnim2 += rotCenter;

    // animation
    uvLocal = (fract(timeX) < 0.5) ? uvAnim1 : uvAnim2;

    // distance
    float neighborDist = 1e+4;
    for (float x = -1.0; x <= 1.0; x += 2.0)
        for (float y = -1.0; y <= 1.0; y += 2.0)
            neighborDist = min(neighborDist, distance(uvLocal, float2(x,y) * 0.5 + float2(0.5)));

    float dist = 1e+4;
    dist = distance(uvLocal, float2(0.5));
    dist = max(dist, neighborDist);

    // color
    float smoothness = 0.05;
    float thickness = 0.03;
    float center = 0.45;
    float density = smoothstep(center - thickness, center + thickness, dist);
    density = smoothstep(1.0, 1.0 - smoothness, density) * (smoothstep(0.0, 0.0 + smoothness, density));

    float colorID = (fract(time) < 0.5) ? hash77(uvID) : hash77(uvID2);
    float colorVariation = 0.3;
    float colorOffset = -0.7;
    float3 color = float3(0,1,2) * M_PI_F * 0.5 + colorID * colorVariation + colorOffset;
    color = max(sin(color), cos(color)) + 0.4;

    float colorBgWave = (sin((uv.x - uv.y) * 2.0 + time * 3.0) + sin((uv.x - uv.y * 0.5) * 1.5 + time)) * 0.5;
    float3 colorBg = float3(1.0, 0.8, 0.5) * mix(0.65, 1.0, colorBgWave);

    float3 ret = 0.0;
    ret = float3(density) * color * (colorBgWave * 0.75 + 0.25 + (1.0 - colorBgWave) * float3(0.1, 0.0, 0.0));
    ret = mix(colorBg, ret, density);
    ret = clamp(ret * (1.2 + sin(time) * 0.1), 0.0, 1.0);
    return float4(ret, 1.0);
}

// MARK: - Day78

// https://www.shadertoy.com/view/3lK3zz Seigaiha

fragment float4 shader_day78(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = (pixPos.xy - res.xy * 0.5) / min(res.x, res.y);
    uv.y *= -1.0;

    float scale = 5.0 + sin(time) * 0.25;
    uv.y -= fract(time * 0.2) / scale;

    float2 uvGrid = floor(uv * scale);
    float uvYOffset = mod(uvGrid.x, 2.0) <= 0.0 ? 0.0 : 0.5;
    float2 uvLocal = fract(uv * scale + float2(0.0, uvYOffset)) - float2(0.0, 0.5);
    float dist = 1e+5;

    // below grids (both sides only)
    for (float x = -1.0; x < 2.0; x += 2.0)
        dist = min(dist, distance(float2(x, -1.0), uvLocal));

    // self grid
    const float WAVE_DISTANCE = 1.0;
    if (WAVE_DISTANCE < dist)
        dist = min(dist, distance(float2(0.0, -0.5), uvLocal));

    // center grids (both sides only)
    if (WAVE_DISTANCE < dist)
        for (float x = -1.0; x < 2.0; x += 2.0)
            dist = min(dist, distance(float2(x, 0.0), uvLocal));

    float3 ret = 0.0;
    const float STRIPES = 8.0;
    float stripeOffset = (-cos(fract(time * 0.5) * M_PI_F) * 0.5 + 0.5) * (4.0 * M_PI_F);
    ret = (0.0 < sin(dist * STRIPES * M_PI_F - stripeOffset)) ? float3(1.0, 0.8, 0.4) : float3(1.0, 0.7, 0.2);

    float uvYGradient = pixPos.y / res.y * -1.0;
    ret *= uvYGradient * 0.4 + 1.0;

    return float4(ret, 1.0);
}

// MARK: - Day79

// https://www.shadertoy.com/view/MsSGWK Water

float water(float3 p, float time, sampler s, texture2d<float, access::sample> texture) {
    float t = time / 4.0;
    p.z += t * 2.0; p.x += t * 2.0;
    float3 c1 = texture.sample(s, p.xz / 30.0).xyz;
    p.z += t * 3.0; p.x += t * 0.5;
    float3 c2 = texture.sample(s, p.xz / 30.0).xyz;
    p.z += t * 4.0; p.x += t * 0.8;
    float3 c3 = texture.sample(s, p.xz / 30.0).xyz;
    c1 += c2 - c3;
    float z = (c1.x + c1.y + c1.z) / 3.0;
    return p.y + z / 4.0;
}

float map(float3 p, float time, sampler s, texture2d<float, access::sample> texture) {
    float d = 100.0;
    d = water(p, time, s, texture);
    return d;
}

float intersect(float3 ro, float3 rd, float time, sampler s, texture2d<float, access::sample> texture) {
    float d = 0.0;
    for (int i = 0; i <= 100; i++) {
        float h = map(ro + rd * d, time, s, texture);
        if (h < 0.1) return d;
        d += h;
    }
    return 0.0;
}

float3 norm(float3 p, float time, sampler s, texture2d<float, access::sample> texture) {
    float eps = 0.1;
    return normalize(float3(
                            map(p + float3(eps, 0, 0), time, s, texture) - map(p + float3(-eps, 0, 0), time, s, texture),
                            map(p + float3(0, eps, 0), time, s, texture) - map(p + float3(0, -eps, 0), time, s, texture),
                            map(p + float3(0, 0, eps), time, s, texture) - map(p + float3(0, 0, -eps), time, s, texture)
                            ));
}

fragment float4 shader_day79(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> abstract1Texture [[texture(6)]]) {

    constexpr sampler s(address::repeat, filter::linear);

    float2 uv = pixPos.xy / res.xy - 0.5;
    uv.y *= -1.0;
    uv.x *= res.x / res.y;
    float3 l1 = normalize(float3(1.0, 1.0, 1.0));
    float3 ro = float3(-3.0, 7.0, -5.0);
    float3 rc = float3(0.0, 0.0, 0.0);
    float3 ww = normalize(rc - ro);
    float3 uu = normalize(cross(float3(0.0, 1.0, 0.0), ww));
    float3 vv = normalize(cross(rc - ro, uu));
    float3 rd = normalize(uu * uv.x + vv * uv.y + ww);
    float d = intersect(ro, rd, time, s, abstract1Texture);
    float3 c = float3(0.0);
    if (d > 0.0) {
        float3 p = ro + rd * d;
        float3 n = norm(p, time, s, abstract1Texture);
        float spc = pow(max(0.0, dot(reflect(l1, n), rd)), 30.0);
        float3 rfa = abstract1Texture.sample(s, (p + n).xz / 6.0).xyz * (8.0 / d);
        c = rfa.xyz + spc + 0.1;
    }
    return float4(float3(c), 1.0);
}

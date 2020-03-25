#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day80

// https://www.shadertoy.com/view/WtdXzM Japanese Patterns

#define Rot(a) float2x2(cos(a), -sin(a), sin(a), cos(a))

float antialiasing(float n, float2 res) {
    return n / min(res.y, res.x);
}

float S(float d, float b, float2 res) {
    return 1.0 - smoothstep(b, antialiasing(1.0, res), d);
}

float2 mod2(float2 x, float2 y) {
    return x - y * floor(x / y);
}

float deg2rad(float num) {
    return num * M_PI_F / 180.0;
}

float ndot(float2 a, float2 b ) { return a.x * b.x - a.y * b.y; }

float dBox2d(float2 p, float2 b) {
    return max(abs(p.x) - b.x, abs(p.y) - b.y);
}

float sdBox(float2 p, float2 b) {
    float2 d = abs(p) - b;
    return length(max(d, float2(0))) + min(max(d.x, d.y), 0.0);
}

float sdPie(float2 p, float2 c, float r) {
    p.x = abs(p.x);
    float l = length(p) - r;
    float m = length(p - c * clamp(dot(p, c), 0.0, r)); // c = sin/cos of the aperture
    return max(l, m * sign(c.y * p.x - c.x * p.y));
}

float sdVesica(float2 p, float r, float d) {
    p = abs(p);
    float b = sqrt(r * r - d * d);
    return ((p.y - b) * d > p.x * b) ? length(p - float2(0.0, b)) : length(p - float2(-d, 0.0)) - r;
}

float sdRhombus(float2 p, float2 b) {
    float2 q = abs(p);
    float h = clamp((-2.0 * ndot(q, b) + ndot(b,b)) / dot(b, b), -1.0, 1.0);
    float d = length(q - 0.5 * b * float2(1.0 - h, 1.0 + h));
    return d * sign(q.x * b.y + q.y * b.x - b.x * b.y);
}

float sdEllipse(float2 p, float2 ab) {
    p = abs(p); if( p.x > p.y ) {p=p.yx;ab=ab.yx;}
    float l = ab.y*ab.y - ab.x*ab.x;
    float m = ab.x*p.x/l;      float m2 = m*m;
    float n = ab.y*p.y/l;      float n2 = n*n;
    float c = (m2+n2-1.0)/3.0; float c3 = c*c*c;
    float q = c3 + m2*n2*2.0;
    float d = c3 + m2*n2;
    float g = m + m*n2;
    float co;
    if (d < 0.0) {
        float h = acos(q / c3) / 3.0;
        float s = cos(h);
        float t = sin(h) * sqrt(3.0);
        float rx = sqrt(-c * (s + t + 2.0) + m2);
        float ry = sqrt(-c * (s - t + 2.0) + m2);
        co = (ry + sign(l) * rx + abs(g) / (rx * ry)- m) / 2.0;
    } else {
        float h = 2.0 * m * n * sqrt(d);
        float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
        float u = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
        float rx = -s - u - c * 4.0 + 2.0 * m2;
        float ry = (s - u) * sqrt(3.0);
        float rm = sqrt(rx * rx + ry * ry);
        co = (ry / sqrt(rm - rx) + 2.0 * g / rm - m) / 2.0;
    }
    float2 r = ab * float2(co, sqrt(1.0-co*co));
    return length(r-p) * sign(p.y-r.y);
}

float sdLine(float2 p, float2 a, float2 b) {
    float2 pa = p-a, ba = b-a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h );
}

float3 textile1(float2 p, float3 col, float3 rhcol, float2 res) {
    float2 size = float2(0.04,0.1);
    float rh = sdRhombus(p,size);
    col = mix(col, rhcol, S(rh, -0.01, res));

    rh = sdRhombus((p + float2(0.1, 0.06)) * Rot(deg2rad(-65.0)), size);
    col = mix(col, rhcol, S(rh, -0.01, res));

    rh = sdRhombus((p + float2(-0.1, 0.06)) * Rot(deg2rad(65.0)), size);
    col = mix(col, rhcol, S(rh, -0.01, res));
    return col;
}

float3 jpTraditionalTex1(float2 p, float3 col, float3 rhcol, float2 res) {
    float2 pref = p;

    float scale = 1.5;
    p *= scale;
    p.x = mod(p.x, 0.42) - 0.21;
    p.y = mod(p.y, 0.22) - 0.11;

    p.y = abs(p.y);
    p.y -= 0.11;
    col = textile1(p, col, rhcol, res);

    p = pref;
    p *= scale;
    p.x -= 0.21;
    p.x = mod(p.x,0.42)-0.21;
    p.y = mod(p.y,0.44)-0.22;
    p.y = abs(p.y);
    p.y -= 0.11;
    float rh = sdRhombus(p, float2(0.04, 0.1));
    col = mix(col, rhcol, S(rh,-0.01, res));

    return col;
}

float3 textile2(float2 p, float3 col, float3 bcol, float2 res) {
    float bsize = 0.15;
    p *= Rot(deg2rad(45.0));
    float b = dBox2d(p,float2(bsize));
    p = abs(p);
    p.x -= 0.067;
    p.y -= 0.067;
    float b2 = dBox2d(p, float2(bsize / 3.0));
    b = max(-b2,b);
    col = mix(col, bcol, S(b,-0.01, res));
    return col;
}

float3 jpTraditionalTex2(float2 p, float3 col, float3 bcol, float2 res) {
    float2 pref = p;

    float scale = 1.5;
    p *= scale;

    p = mod2(p, 0.45) - 0.225;
    col = textile2(p, col, bcol, res);

    p = pref;
    p *= scale;
    p -= 0.225;
    p = mod2(p, 0.45) - 0.225;
    col = textile2(p, col, bcol, res);

    return col;
}

float3 jpTraditionalTex3(float2 p, float3 col, float3 ccol, float2 res) {
    p.x = mod(p.x, 0.25) - (0.25 * 0.5);
    p.y = mod(p.y, 0.26) - (0.26 * 0.5);

    float r = 0.15;
    p = abs(p);
    p.x -= 0.06;
    p.y -= 0.065;

    float c = sdVesica(p * Rot(deg2rad(-45.0)), r - 0.02, r * 0.63);
    col = mix(col, ccol, S(c, 0.0, res));

    return col;
}

float3 textile4(float2 p, float3 col, float3 ccol, float2 res) {
    float r = 0.16;

    float c =sdPie(p,float2(0.16,0.1),r);
    c = max(-(length(p)-0.05),c);
    col = mix(col, ccol, S(c, 0.0, res));
    c =sdPie(p,float2(0.16,0.1),r-0.02);
    c = max(-(length(p)-0.075),c);
    col = mix(col, float3(1.0), S(c,0.0, res));
    c =sdPie(p,float2(0.16,0.1),r-0.037);
    c = max(-(length(p)-0.095),c);
    col = mix(col, ccol, S(c, 0.0, res));
    c = length(p-float2(0.0,0.015))-0.02;
    col = mix(col, float3(1.0), S(c, 0.0, res));
    return col;
}

float3 jpTraditionalTex4(float2 p, float3 col, float3 ccol, float2 res) {
    float2 pref = p;

    p.x = mod(p.x, 0.4) - (0.2);
    p.y = mod(p.y, 0.18) - (0.09);
    p.y += 0.085;
    col = textile4(p, col, ccol, res);
    p= pref;

    p.x += 0.2;
    p.y -= 0.085;
    p.x = mod(p.x, 0.4) - (0.2);
    p.y = mod(p.y, 0.18) - (0.09);
    p.y += 0.085;
    col = textile4(p, col, ccol, res);
    return col;
}

float3 textile5(float2 p, float3 col, float3 ccol, float2 res) {
    float2 pref = p;
    p.y = abs(p.y);
    p.y -= 0.1;
    float c = sdEllipse(p, float2(0.04, 0.06));
    float c2 = sdEllipse((p - float2(-0.02, -0.01)) * Rot(deg2rad(-10.0)), float2(0.04, 0.05));
    float c3 = sdEllipse((p - float2(0.02, -0.01)) * Rot(deg2rad(10.0)), float2(0.04, 0.05));
    c = min(c, min(c2, c3));
    c = max(-(length(p - float2(0.0, -0.05)) - 0.04), c);
    col = mix(col, ccol, S(c, 0.0, res));
    p = pref;

    p.x = abs(p.x);
    p.x -= 0.13;
    c = sdEllipse(p, float2(0.09, 0.03));
    c2 = sdEllipse((p - float2(-0.015, -0.015)) * Rot(deg2rad(15.0)), float2(0.07, 0.04));
    c3 = sdEllipse((p - float2(-0.015, 0.015)) * Rot(deg2rad(-15.0)), float2(0.07, 0.04));
    float c4 = sdEllipse(p - float2(-0.05, 0.0), float2(0.07, 0.025));
    c = min(c, min(c2, c3));
    c = max(-c4, c);
    col = mix(col, ccol, S(c, 0.0, res));
    p = pref;

    col = mix(col, ccol, S(length(p) - 0.03, 0.0, res));

    p = abs(p);

    float b = dBox2d(p - float2(0.0, 0.23), float2(0.05, 0.045));
    float b2 = dBox2d(p - float2(0.35, 0.0), float2(0.05, 0.045));
    b = min(b, b2);
    col = mix(col, ccol, S(abs(b) - 0.003, 0.0, res));

    float l = sdLine(p, float2(0.05, 0.185), float2(0.3, 0.045));
    float l2 = sdLine(p, float2(0.05, 0.205), float2(0.325, 0.048));
    float l3 = sdLine(p, float2(0.05, 0.225), float2(0.355, 0.048));
    l = min(l, min(l2, l3));
    col = mix(col, ccol, S(l - 0.002, 0.0, res));
    return col;
}

float3 jpTraditionalTex5(float2 p, float3 col, float3 ccol, float2 res) {
    float2 pref = p;

    float scale = 1.5;
    p *= scale;

    p.x = mod(p.x, 0.7) - 0.35;
    p.y = mod(p.y, 0.46) - 0.23;
    col = textile5(p, col, ccol, res);
    p = pref;

    p *= scale;
    p.x -= 0.35;
    p.y -= 0.23;
    p.x = mod(p.x, 0.7) - 0.35;
    p.y = mod(p.y, 0.46) - 0.23;
    col = textile5(p, col, ccol, res);

    return col;
}

float3 textile6(float2 p, float3 col, float3 ccol, float2 res) {
    float n = 5.0;
    float deg = 360.0 / n;
    float startRad = deg2rad(180.0);
    for(float i = 0.0; i < n; i += 1.0) {
        float rad = deg2rad(deg * i) + startRad;
        float2x2 rot = Rot(rad);
        float dist = 0.15;
        float x = sin(rad) * dist;
        float y = cos(rad) * dist;

        float v = sdVesica((p + float2(x, y)) * rot, 0.14, 0.075);

        dist = 0.07;
        x = sin(rad) * dist;
        y = cos(rad) * dist;
        float e = sdEllipse((((p + float2(x, y))) * rot), float2(0.015, 0.07));
        v = max(-e, v);
        col = mix(col, ccol, S(v, 0.0, res));
    }

    float c = length(p) - 0.03;
    col = mix(col, ccol, S(c, 0.0, res));
    return col;
}

float3 textile6RepeatBg(float2 p, float3 col, float3 ccol, float2 res) {
    float scale = 1.5;
    p *= scale;
    p.x = mod(p.x, 0.9) - 0.45;
    p.y = mod(p.y, 0.54) - 0.275;
    col = textile6(p, col, ccol, res);
    return col;
}

float3 jpTraditionalTex6(float2 p, float3 col, float3 ccol, float2 res) {
    col = textile6RepeatBg(p, col, ccol, res);
    p.x -= 0.9;
    p.y -= 0.54;
    col = textile6RepeatBg(p, col, ccol, res);
    return col;
}

float leaf(float2 p) {
    float c = length(p) - 0.1;
    p.x *= 0.8;
    p.y *= 0.6;
    float c2  = length(p - float2(0.06, 0.05)) - 0.08;
    c = max(-c2, c);
    return c;
}

float3 textile7(float2 p, float3 col, float3 ccol, float2 res) {
    float n = 5.0;
    float deg = 360.0 / n;
    float startRad = deg2rad(180.0);
    float ld = 1.0;
    for(float i = 0.0; i < n; i += 1.0) {
        float rad = deg2rad(deg * i) + startRad;
        float2x2 rot = Rot(rad);
        float dist = 0.11;
        float x = sin(rad) * dist;
        float y = cos(rad) * dist;
        float l = leaf((p + float2(x, y)) * rot);
        ld = min(ld, l);
    }

    ld = max(-(length(p) - 0.05), ld);
    col = mix(col, ccol, S(ld, 0.0, res));
    float c = length(p) - 0.02;
    ld = min(c, ld);
    col = mix(col, ccol, S(ld, 0.0, res));
    return col;
}

float3 jpTraditionalTex7(float2 p, float3 col, float3 ccol, float2 res) {
    float2 pref = p;
    float scale = 1.5;
    p *= scale;

    p.x = mod(p.x, 0.8) - 0.4;
    p.y = mod(p.y, 0.5) - 0.25;
    col = textile7(p, col, ccol, res);
    p = pref;

    p *= scale;
    p.x -= 0.4;
    p.y -= 0.25;
    p.x = mod(p.x, 0.8) - 0.4;
    p.y = mod(p.y, 0.5) - 0.25;
    col = textile7(p, col, ccol, res);
    return col;
}

float3 textile8(float2 p, float3 col, float3 ccol, float2 res) {
    float b = sdBox(p * Rot(deg2rad(45.0)), float2(0.05)) - 0.05;
    b = max(-(length(p) - 0.03), b);
    col = mix(col, ccol, S(b, 0.0, res));
    return col;
}

float3 jpTraditionalTex8(float2 p, float3 col, float3 ccol, float2 res) {
    float2 pref = p;
    p = mod2(p, 0.3) - 0.15;
    col = textile8(p, col, ccol, res);
    p = pref;

    p -= 0.15;
    p = mod2(p, 0.3) - 0.15;
    col = textile8(p, col, ccol, res);
    return col;
}

float3 textile9(float2 p, float3 col, float3 ccol, float2 res) {
    float b = dBox2d(p, float2(0.1));
    p.x = abs(p.x);
    p.x -= 0.04;
    float b2 = dBox2d(p, float2(0.02, 0.11));
    b = max(-b2, b);
    col = mix(col, ccol, S(b, 0.0, res));
    return col;
}

float3 textile9RepeatBg(float2 p, float3 col, float3 ccol, float2 res) {
    float2 pref = p;
    p.x = mod(p.x,0.45) - 0.225;
    p.y = mod(p.y,0.45) - 0.225;
    col = textile9(p, col, ccol, res);
    p = pref;

    p.y -= 0.225;
    p.x = mod(p.x, 0.45) - 0.225;
    p.y = mod(p.y, 0.45) - 0.225;
    col = textile9(p * Rot(deg2rad(90.0)), col, ccol, res);
    return col;
}

float3 jpTraditionalTex9(float2 p, float3 col, float3 ccol, float2 res) {
    col = textile9RepeatBg(p, col, ccol, res);
    p -= 0.225;
    col = textile9RepeatBg(p, col, ccol, res);
    return col;
}

fragment float4 shader_day80(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = (pixPos.xy - 0.5 * res.xy) / res.y;
    float3 col = float3(0.8, 0.9, 1.0);
    float3 baseCol = float3(0.0, 0.3, 0.5);

    uv.y *= -1.0;
    uv.y -= time * 0.1;

    float ratio = 1.0;
    float ntextile = 9.0;
    float scene = mod(time, ratio * ntextile);
    if (scene<ratio) {
        col = jpTraditionalTex1(uv, col, baseCol, res);
    } else if (scene >= ratio && scene < ratio * 2.0) {
        col = jpTraditionalTex2(uv, col, baseCol, res);
    } else if (scene >= ratio * 2.0 && scene < ratio * 3.0) {
        col = jpTraditionalTex3(uv, col, baseCol, res);
    } else if (scene >= ratio * 3.0 && scene < ratio * 4.0) {
        col = jpTraditionalTex4(uv, col, baseCol, res);
    } else if (scene >= ratio * 4.0 && scene < ratio * 5.0) {
        col = jpTraditionalTex5(uv, col, baseCol, res);
    } else if (scene >= ratio * 5.0 && scene < ratio * 6.0) {
        col = jpTraditionalTex6(uv, col, baseCol, res);
    } else if (scene >= ratio * 6.0 && scene < ratio * 7.0) {
        col = jpTraditionalTex7(uv, col, baseCol, res);
    } else if (scene >= ratio * 7.0 && scene < ratio * 8.0) {
        col = jpTraditionalTex8(uv, col, baseCol, res);
    } else if (scene >= ratio * 8.0 && scene < ratio * 9.0) {
        col = jpTraditionalTex9(uv, col, baseCol, res);
    }

    return float4(col, 1.0);
}

// MARK: - Day81

// https://www.shadertoy.com/view/ltffzl Rain
// https://github.com/Silence-GitHub/BBMetalImage/blob/master/BBMetalImage/BBMetalImage/BBMetalBilateralBlurFilter.metal

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

float3 bilateralBlur(texture2d<float, access::sample> texture, float2 uv, float2 res) {

    const int GAUSSIAN_SAMPLES = 9;
    const float2 inCoordinate = uv;

    int multiplier = 0;
    float2 blurStep;
    float2 singleStepOffset(float(6.0) / res.x, float(6.0) / res.y);
    float2 blurCoordinates[GAUSSIAN_SAMPLES];

    for (int i = 0; i < GAUSSIAN_SAMPLES; i++) {
        multiplier = (i - ((GAUSSIAN_SAMPLES - 1) / 2));
        blurStep = float(multiplier) * singleStepOffset;
        blurCoordinates[i] = inCoordinate + blurStep;
    }

    float4 centralColor;
    float gaussianWeightTotal;
    float4 sum;
    float4 sampleColor;
    float distanceFromCentralColor;
    float gaussianWeight;

    constexpr sampler quadSampler;
    const float distanceNormalizationFactor = float(0.01);

    centralColor = texture.sample(quadSampler, blurCoordinates[4]);
    gaussianWeightTotal = 0.18;
    sum = centralColor * 0.18;

    sampleColor = texture.sample(quadSampler, blurCoordinates[0]);
    distanceFromCentralColor = min(distance(centralColor, sampleColor) * distanceNormalizationFactor, 1.0);
    gaussianWeight = 0.05 * (1.0 - distanceFromCentralColor);
    gaussianWeightTotal += gaussianWeight;
    sum += sampleColor * gaussianWeight;

    sampleColor = texture.sample(quadSampler, blurCoordinates[1]);
    distanceFromCentralColor = min(distance(centralColor, sampleColor) * distanceNormalizationFactor, 1.0);
    gaussianWeight = 0.09 * (1.0 - distanceFromCentralColor);
    gaussianWeightTotal += gaussianWeight;
    sum += sampleColor * gaussianWeight;

    sampleColor = texture.sample(quadSampler, blurCoordinates[2]);
    distanceFromCentralColor = min(distance(centralColor, sampleColor) * distanceNormalizationFactor, 1.0);
    gaussianWeight = 0.12 * (1.0 - distanceFromCentralColor);
    gaussianWeightTotal += gaussianWeight;
    sum += sampleColor * gaussianWeight;

    sampleColor = texture.sample(quadSampler, blurCoordinates[3]);
    distanceFromCentralColor = min(distance(centralColor, sampleColor) * distanceNormalizationFactor, 1.0);
    gaussianWeight = 0.15 * (1.0 - distanceFromCentralColor);
    gaussianWeightTotal += gaussianWeight;
    sum += sampleColor * gaussianWeight;

    sampleColor = texture.sample(quadSampler, blurCoordinates[5]);
    distanceFromCentralColor = min(distance(centralColor, sampleColor) * distanceNormalizationFactor, 1.0);
    gaussianWeight = 0.15 * (1.0 - distanceFromCentralColor);
    gaussianWeightTotal += gaussianWeight;
    sum += sampleColor * gaussianWeight;

    sampleColor = texture.sample(quadSampler, blurCoordinates[6]);
    distanceFromCentralColor = min(distance(centralColor, sampleColor) * distanceNormalizationFactor, 1.0);
    gaussianWeight = 0.12 * (1.0 - distanceFromCentralColor);
    gaussianWeightTotal += gaussianWeight;
    sum += sampleColor * gaussianWeight;

    sampleColor = texture.sample(quadSampler, blurCoordinates[7]);
    distanceFromCentralColor = min(distance(centralColor, sampleColor) * distanceNormalizationFactor, 1.0);
    gaussianWeight = 0.09 * (1.0 - distanceFromCentralColor);
    gaussianWeightTotal += gaussianWeight;
    sum += sampleColor * gaussianWeight;

    sampleColor = texture.sample(quadSampler, blurCoordinates[8]);
    distanceFromCentralColor = min(distance(centralColor, sampleColor) * distanceNormalizationFactor, 1.0);
    gaussianWeight = 0.05 * (1.0 - distanceFromCentralColor);
    gaussianWeightTotal += gaussianWeight;
    sum += sampleColor * gaussianWeight;

    float4 ret = sum / gaussianWeightTotal;
    return float3(ret.rgb);
}

fragment float4 shader_day81(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 uv = (pixPos.xy - 0.5 * res.xy) / res.y;
    uv.y *= -1.0;
    float2 UV = pixPos.xy / res.xy;

    float T = time;
    float t = T * 0.2;

    float rainAmount = sin(T * 0.05) * 0.3 + 0.7;

    float staticDrops = S(-0.5, 1.0, rainAmount) * 2.0;
    float layer1 = S(0.25, 0.75, rainAmount);
    float layer2 = S(0.0, 0.5, rainAmount);

    float2 c = Drops(uv, t, staticDrops, layer1, layer2);
    float2 e = float2(0.001, 0.0);
    float cx = Drops(uv + e, t, staticDrops, layer1, layer2).x;
    float cy = Drops(uv + e.yx, t, staticDrops, layer1, layer2).x;
    float2 n = float2(cx - c.x, cy - c.x);

    float3 color = bilateralBlur(texture, UV +n, res);

    t = (T + 3.0) * 0.5;
    float colFade = sin(t * 0.2) * 0.5 + 0.5;
//    color.rgb *= mix(float3(1.0), float3(0.8, 0.9, 1.3), colFade);
    color.rgb *= mix(float3(1.0), float3(0.6, 0.7, 1.1), colFade);

    return float4(color.rgb, 1.0);
}

// MARK: - Day82

// https://www.shadertoy.com/view/ldB3Dh Depth like

fragment float4 shader_day82(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 pixel = (pixPos.xy - res.xy * 0.5) / res.xy;

    float3 col;
    for (int i = 1; i < 50; i++) {
        float depth = 40.0 + float(i);
        float2 uv = pixel * depth / 210.0;
        col = texture.sample(s, fract(uv + 0.5)).rgb;
        if ((1.0 - (col.y * col.y)) < float(i + 1) / 50.0) {
            break;
        }
    }

    col = min(col * col * 1.5, 1.0);
    return float4(col, 1.0);
}

// MARK: - Day83

// https://www.shadertoy.com/view/Md3GWf Noise strip

float2 stepnoise(float2 p, float size) {
    p += 10.0;
    float x = floor(p.x/size)*size;
    float y = floor(p.y/size)*size;

    x = fract(x * 0.1) + 1.0 + x * 0.0002;
    y = fract(y * 0.1) + 1.0 + y * 0.0003;

    float a = fract(1.0 / (0.000001 * x * y + 0.00001));
    a = fract(1.0 / (0.000001234 * a + 0.00001));

    float b = fract(1.0 / (0.000002 * (x * y + x) + 0.00001));
    b = fract(1.0 / (0.0000235 * b + 0.00001));

    return float2(a, b);
}

float tent(float f) {
    return 1.0 - abs(fract(f) - 0.5) * 2.0;
}

#define SEED1 (1.705)
#define SEED2 (1.379)
#define DMUL 8.12235325

float poly(float a, float b, float c, float ta, float tb, float tc) {
    return (a*ta + b*tb + c*tc) / (ta+tb+tc);
}

float mask(float2 p) {
    float2 r = stepnoise(p, 5.5) - 0.5;
    p[0] += r[0] * DMUL;
    p[1] += r[1] * DMUL;
    float f = fract(p[0] * SEED1 + p[1] / (SEED1 + 0.15555)) * 1.03;
    return poly(pow(f, 150.0), f * f, f, 1.0, 0.0, 1.3);
}

float s(float x, float y, float2 uv, sampler sa, texture2d<float, access::sample> texture, float2 res) {
    float4 clr = texture.sample(sa, float2(x, y) / res.xy + uv);
    float f = clr[0] * 0.3 + clr[1] * 0.6 + clr[1] * 0.1;
    return f;
}

float3x3 mynormalize(float3x3 mat) {
    float sum = mat[0][0] + mat[0][1] + mat[0][2]
    + mat[1][0] + mat[1][1] + mat[1][2]
    + mat[2][0] + mat[2][1] + mat[2][2];
    mat[0] /= sum;
    mat[1] /= sum;
    mat[2] /= sum;
    return mat;
}

fragment float4 shader_day83(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler sa(address::clamp_to_edge, filter::linear);

    float2 uv = pixPos.xy;

    float4 clr = texture.sample(sa, pixPos.xy / res.xy);
    float f = clr[0] * 0.3 + clr[1] * 0.6 + clr[1] * 0.1;
    float2 uv3 = pixPos.xy / res.xy;
    float d = 0.5;
    float3x3 mat = float3x3(
                            float3(d, d,   d),
                            float3(d, 2.0, d),
                            float3(d, d,   d)
                            );

    float f1 = s(0.0, 0.0, uv3, sa, texture, res);

    mat = mynormalize(mat) * 1.0;
    f = s(-1.0, -1.0, uv3, sa, texture, res) * mat[0][0] + s(-1.0, 0.0, uv3, sa, texture, res) * mat[0][1] + s(-1.0, 1.0, uv3, sa, texture, res) * mat[0][2]
    + s(0.0, -1.0, uv3, sa, texture, res) * mat[1][0] + s(0.0, 0.0, uv3, sa, texture, res) * mat[1][1] + s(0.0, 1.0, uv3, sa, texture, res) * mat[1][2]
    + s(1.0, -1.0, uv3, sa, texture, res) * mat[2][0] + s(1.0, 0.0, uv3, sa, texture, res) * mat[2][1] + s(1.0, 1.0, uv3, sa, texture, res) * mat[2][2];

    f = (f - s(0.0, 0.0, uv3, sa, texture, res));
    f *= 40.0;
    f = f1 - f;

    float c = mask(uv);
    c = float(f >= c);

    return float4(c, c, c, 1.0);
}

// MARK: - day84

// https://www.shadertoy.com/view/MdsXRB Light bulb

fragment float4 shader_day84(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 R = res.xy;
    float2 u = pixPos.xy / R;
    float4 o = texture.sample(s, u);
    u *= R / R.y;
    u.x -= 0.5 * floor(mod(32.0 * u.y + 0.5, 2.0)) / 32.0;
    float2 u0 = floor(u * 32.0 + 0.5) / 32.0;
    float d = length(u - u0) * 32.0;
    o = smoothstep(o, float4(0), float4(d));
    return o;
}

// MARK: - Day85

// https://www.shadertoy.com/view/ldBfzw Fireworks

#define N(h) fract(sin(float4(6.0, 9.0, 1.0, 0.0) * h) * 900.0)

fragment float4 shader_day85(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float4 ret = float4(0.0);
    float2 u = pixPos.xy / res.y;
    float e = 0.0;
    float d = 0.0;
    float4 p = 0.0;

    for (float i = 0.0; i < 5.0; i++) {
        d = floor(e = i * 9.1 + time);
        p = N(d) + 0.3;
        e -= d;
        for (d = 0.0; d < 30.0; d++) {
            ret += p * (1.0 - e) / 1000.0 / length(u - (p - e * (N(d + i) - 0.5)).xy);
        }
    }

    return ret;
}


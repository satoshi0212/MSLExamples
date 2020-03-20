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

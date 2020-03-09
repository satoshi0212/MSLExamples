#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day60

// https://www.shadertoy.com/view/Xd2GR3 Tiled hexagon

float4 hexagon(float2 p) {
    float2 q = float2( p.x * 2.0 * 0.5773503, p.y + p.x * 0.5773503);
    float2 pi = floor(q);
    float2 pf = fract(q);
    float v = mod(pi.x + pi.y, 3.0);
    float ca = step(1.0, v);
    float cb = step(2.0, v);
    float2 ma = step(pf.xy, pf.yx);
    float e = dot(ma, 1.0 - pf.yx + ca * (pf.x + pf.y - 1.0) + cb * (pf.yx - 2.0 * pf.xy));
    p = float2(q.x + floor(0.5 + p.y / 1.5), 4.0 * p.y / 3.0) * 0.5 + 0.5;
    float f = length((fract(p) - 0.5) * float2(1.0, 0.85));
    return float4(pi + ca - cb * ma, e, f);
}

float hash60(float2 p) {
    float n = dot(p, float2(127.1, 311.7));
    return fract(sin(n) * 43758.5453);
}

float noise60(sampler s, float3 x, texture2d<float, access::sample> rgbNoiseTexture) {
    float3 p = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float2 uv = (p.xy + float2(37.0, 17.0) * p.z) + f.xy;
    float2 rg = rgbNoiseTexture.sample(s, (uv + 0.5) / 256.0).yx;
    return mix(rg.x, rg.y, f.z);
}

fragment float4 shader_day60(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> rgbNoiseTexture [[texture(4)]]) {
    constexpr sampler s(address::repeat, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    float3 tot = float3(0.0);
    float2 pos = (-res.xy + 2.0 * pixPos.xy) / res.y;

    pos *= 1.0 + 0.3 * length(pos);

    float4 h = hexagon(8.0 * pos + 0.5 * time);
    float n = noise60(s, float3(0.3 * h.xy + time * 0.1, time), rgbNoiseTexture);
    float3 col = 0.15 + 0.15 * hash60(h.xy + 1.2) * float3(1.0);
    col *= smoothstep(0.10, 0.11, h.z);
    col *= smoothstep(0.10, 0.11, h.w);
    col *= 1.0 + 0.15 * sin(40.0 * h.z);
    col *= 0.75 + 0.5 * h.z * n;

    h = hexagon(6.0 * pos + 0.6 * time);
    n = noise60(s, float3(0.3 * h.xy + time * 0.1, time), rgbNoiseTexture);
    float3 colb = 0.9 + 0.8 * sin(hash60(h.xy) * 1.5 + 2.0 + float3(0.0, 1.0, 1.0));
    colb *= smoothstep(0.10, 0.11, h.z);
    colb *= 1.0 + 0.15 * sin(40.0 * h.z);
    colb *= 0.75 + 0.5 * h.z * n;

    h = hexagon(6.0 * (pos + 0.1 * float2(-1.3, 1.0)) + 0.6 * time);
    col *= 1.0 - 0.8 * smoothstep(0.45, 0.451, noise60(s, float3(0.3 * h.xy + time * 0.1, time), rgbNoiseTexture));
    col = mix(col, colb, smoothstep(0.45, 0.451, n));
    col *= pow(16.0 * uv.x * (1.0 - uv.x) * uv.y * (1.0 - uv.y), 0.1);
    tot += col;

    return float4(tot, 1.0);
}

// MARK: - Day61

// https://www.shadertoy.com/view/4lK3zy 独特な色合いのポスタライズ

float3 getGrayscale(float3 origColor) {
    return float3(origColor.r * 0.21 +
                  origColor.g * 0.72 +
                  origColor.b * 0.07);
}

float3 getPosterColor(float3 trueColor) {
    const float posterLevelRange = 1.0 / 3.0;
    float3 grayscale = getGrayscale(trueColor);
    float3 modColor = mix(trueColor, grayscale, -0.5);
    modColor = clamp(modColor * 2.0 - 0.4, 0.0, 1.0);
    return float3((floor(modColor.r / posterLevelRange + 0.5)) * posterLevelRange,
                  (floor(modColor.g / posterLevelRange + 0.5)) * posterLevelRange,
                  (floor(modColor.b / posterLevelRange + 0.5)) * posterLevelRange);
}

float easeIn (float perc) {
    return perc * perc;
}

fragment float4 shader_day61(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]],
                             texture2d<float, access::sample> noiseTexture [[texture(2)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = pixPos.xy / res.xy;

    const int roundDist = 24;
    const float roundAmount = 0.3;
    const int averageDist = 2;

    float2 texelSize = float2(1.0) / res.xy;
    float3 color = float3(0.0, 0.0, 0.0);
    for (int x = -averageDist; x <= averageDist; x++) {
        for (int y = -averageDist; y <= averageDist; y++) {
            float4 texelColor = texture.sample(s, uv + (float2(texelSize.x * float(x), texelSize.y * float(y))));
            color += texelColor.rgb * texelColor.a;
        }
    }
    color /= (pow(float(2 * averageDist + 1), 2.0));

    float3 overlay = getGrayscale(noiseTexture.sample(s, uv).rgb);
    float3 posterColor = getPosterColor(color);

    float brighten = 0.0;
    float darken = 0.0;
    float3 testUp;
    float3 testDown;
    for (int offset = roundDist; offset > 0; offset--) {
        testUp = texture.sample(s, uv + (float2(float(offset) * -0.25, float(offset) * 1.0) / res.xy)).rgb;
        testUp = getPosterColor(testUp);
        testDown = texture.sample(s, uv + (float2(float(offset) * 0.25, float(offset) * -1.0) / res.xy)).rgb;
        testDown = getPosterColor(testDown);
        if (testUp.r != posterColor.r && testUp.g != posterColor.g && testUp.b != posterColor.b) {
            brighten = easeIn(1.0 - float(offset) / float(roundDist));
        }
        if (testDown.r != posterColor.r && testDown.g != posterColor.g && testDown.b != posterColor.b) {
            darken = easeIn(1.0 - float(offset) / float(roundDist));
        }
    }

    return float4(posterColor
                  + float3(brighten * roundAmount)
                  - float3(darken * roundAmount)
                  + (overlay / 14.0 - 0.07)
                  , 1.0);
}

// MARK: - Day62

// https://www.shadertoy.com/view/4tdSWr Clouds

float2 hash62(float2 p) {
    p = float2(dot(p, float2(127.1,311.7)), dot(p, float2(269.5,183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise62(float2 p) {
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;
    float2 i = floor(p + (p.x + p.y) * K1);
    float2 a = p - i + (i.x + i.y) * K2;
    float2 o = (a.x > a.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float2 b = a - o + K2;
    float2 c = a - 1.0 + 2.0 * K2;
    float3 h = max(0.5 - float3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    float3 n = h * h * h * h * float3(dot(a, hash62(i)), dot(b, hash62(i + o)), dot(c, hash62(i + 1.0)));
    return dot(n, float3(70.0));
}

float fbm62(float2 n) {
    const float2x2 m = float2x2(1.6,  1.2, -1.2, 1.6);
    float total = 0.0;
    float amplitude = 0.1;
    for (int i = 0; i < 7; i++) {
        total += noise62(n) * amplitude;
        n = m * n;
        amplitude *= 0.4;
    }
    return total;
}

fragment float4 shader_day62(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    const float cloudscale = 1.1;
    const float speed = 0.03;
    const float2x2 m = float2x2(1.6,  1.2, -1.2, 1.6);

    float2 p = pixPos.xy / res.xy;
    float2 uv = p * float2(res.x / res.y, 1.0);

    float timeX = time * speed;
    float q = fbm62(uv * cloudscale * 0.5);

    float r = 0.0;
    uv *= cloudscale;
    uv -= q - timeX;
    float weight = 0.8;
    for (int i = 0; i < 8; i++){
        r += abs(weight * noise62(uv));
        uv = m * uv + timeX;
        weight *= 0.7;
    }

    float f = 0.0;
    uv = p * float2(res.x / res.y, 1.0);
    uv *= cloudscale;
    uv -= q - timeX;
    weight = 0.7;
    for (int i = 0; i < 8; i++) {
        f += weight * noise62(uv);
        uv = m * uv + timeX;
        weight *= 0.6;
    }

    f *= r + f;

    float c = 0.0;
    timeX = time * speed * 2.0;
    uv = p * float2(res.x / res.y, 1.0);
    uv *= cloudscale * 2.0;
    uv -= q - timeX;
    weight = 0.4;
    for (int i = 0; i < 7; i++) {
        c += weight * noise62(uv);
        uv = m * uv + timeX;
        weight *= 0.6;
    }

    float c1 = 0.0;
    timeX = time * speed * 3.0;
    uv = p * float2(res.x / res.y, 1.0);
    uv *= cloudscale * 3.0;
    uv -= q - timeX;
    weight = 0.4;
    for (int i = 0; i < 7; i++) {
        c1 += abs(weight * noise62(uv));
        uv = m * uv + timeX;
        weight *= 0.6;
    }

    c += c1;

    float3 skycolour = mix(float3(0.4, 0.7, 1.0), float3(0.2, 0.4, 0.6), p.y);
    float3 cloudcolour = float3(1.1, 1.1, 0.9) * clamp((0.5 + 0.3 * c), 0.0, 1.0);
    f = 0.2 + 8.0 * f * r;
    float3 result = mix(skycolour, clamp(0.5 * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));
    return float4(result, 1.0);
}

// MARK: - Day63

// https://www.shadertoy.com/view/tt2XzG Gear

float smax(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0);
    return max(a, b) + h * h * 0.25 / k;
}

float sdSphere(float3 p, float r) {
    return length(p) - r;
}

float sdVerticalSemiCapsule(float3 p, float h, float r) {
    p.y = max(p.y - h, 0.0);
    return length(p) - r;
}

float sdCross(float2 p, float2 b, float r) {
    p = abs(p);
    p = (p.y > p.x) ? p.yx : p.xy;

    float2 q = p - b;
    float k = max(q.y, q.x);
    float2 w = (k > 0.0) ? q : float2(b.y - p.x, -k);

    return sign(k) * length(max(w, 0.0)) + r;
}

float dot2(float2 v) { return dot(v, v); }

float sdTrapezoid(float2 p, float r1, float r2, float he) {
    float2 k1 = float2(r2, he);
    float2 k2 = float2(r2 - r1, 2.0 * he);
    p.x = abs(p.x);
    float2 ca = float2(max(0.0, p.x - ((p.y < 0.0) ? r1 : r2)), abs(p.y) - he);
    float2 cb = p - k1 + k2 * clamp(dot(k1 - p, k2) / dot2(k2), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot2(ca), dot2(cb)));
}

float2 iSphere(float3 ro, float3 rd, float rad) {
    float b = dot( ro, rd );
    float c = dot( ro, ro ) - rad * rad;
    float h = b * b - c;
    if (h < 0.0) return float2(-1.0);
    h = sqrt(h);
    return float2(-b-h, -b+h);
}

float dents(float2 q, float tr, float y) {
    const float an = 6.283185/12.0;
    float fa = (atan2(q.y,q.x) + an * 0.5) / an;
    float sym = an * floor(fa);
    float2 r = float2x2(cos(sym), -sin(sym), sin(sym), cos(sym)) * q;
    float d = length(max(abs(r - float2(0.17, 0.0)) - tr * float2(0.042, 0.041 * y),0.0));
    return d - 0.005 * tr;
}

float4 gear(float3 q, float off, float time) {
    {
        float an = 2.0 * time * sign(q.y) + off * 6.283185 / 24.0;
        float co = cos(an), si = sin(an);
        q.xz = float2x2(co, -si, si, co) * q.xz;
    }

    q.y = abs(q.y);

    float an2 = 2.0 * min(1.0-2.0*abs(fract(0.5+time/10.0)-0.5),1.0/2.0);
    float3 tr = min(10.0 * an2 - float3(4.0, 6.0, 8.0), 1.0);

    // ring
    float d = abs(length(q.xz) - 0.155 * tr.y) - 0.018;

    // add dents
    float r = length(q);
    d = min(d, dents(q.xz,tr.z, r));

    // slice it
    float de = -0.0015 * clamp(600.0 * abs(dot(q.xz,q.xz) - 0.155 * 0.155), 0.0, 1.0);
    d = smax(d, abs(r - 0.5) - 0.03 + de, 0.005 * tr.z);

    // add cross
    float d3 = sdCross(q.xz, float2(0.15, 0.022) * tr.y, 0.02 * tr.y);
    float2 w = float2(d3, abs(q.y - 0.485) - 0.005 * tr.y);
    d3 = min(max(w.x, w.y),0.0) + length(max(w, 0.0)) - 0.003 * tr.y;
    d = min(d, d3);

    // add pivot
    d = min(d, sdVerticalSemiCapsule(q, 0.5 * tr.x, 0.01));

    // base
    d = min(d, sdSphere(q - float3(0.0, 0.12, 0.0), 0.025));

    return float4(d, q.xzy);
}

float2 rot(float2 v) {
    return float2(v.x - v.y, v.y + v.x) * 0.707107;
}

float4 map63(float3 p, float time) {
    // center sphere
    float4 d = float4(sdSphere(p, 0.12), p);

    // gears. There are 18, but we only evaluate 4
    float3 qx = float3(rot(p.zy),p.x); if(abs(qx.x)>abs(qx.y)) qx=qx.zxy;
    float3 qy = float3(rot(p.xz),p.y); if(abs(qy.x)>abs(qy.y)) qy=qy.zxy;
    float3 qz = float3(rot(p.yx),p.z); if(abs(qz.x)>abs(qz.y)) qz=qz.zxy;
    float3 qa = abs(p); qa = (qa.x>qa.y && qa.x>qa.z) ? p.zxy :
    (qa.z>qa.y             ) ? p.yzx :
    p.xyz;
    float4 t;
    t = gear( qa,0.0,time ); if( t.x<d.x ) d=t;
    t = gear( qx,1.0,time ); if( t.x<d.x ) d=t;
    t = gear( qz,1.0,time ); if( t.x<d.x ) d=t;
    t = gear( qy,1.0,time ); if( t.x<d.x ) d=t;

    return d;
}

float3 calcNormal(float3 pos, float time) {
    float2 e = float2(1.0, -1.0) * 0.5773;
    const float eps = 0.00025;
    return normalize(e.xyy * map63( pos + e.xyy*eps, time ).x +
                     e.yyx * map63( pos + e.yyx*eps, time ).x +
                     e.yxy * map63( pos + e.yxy*eps, time ).x +
                     e.xxx * map63( pos + e.xxx*eps, time ).x );
}

float calcAO(float3 pos, float3 nor, float time) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.01 + 0.12 * float(i) / 4.0;
        float d = map63(pos + h * nor, time).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

float calcSoftshadow(float3 ro, float3 rd, float k, float time) {
    float res = 1.0;

    // bounding sphere
    float2 b = iSphere( ro, rd, 0.535 );
    if (b.y > 0.0) {
        // raymarch
        float tmax = b.y;
        float t = max(b.x,0.001);
        for (int i = 0; i < 64; i++) {
            float h = map63(ro + rd * t, time).x;
            res = min(res, k * h / t);
            t += clamp(h, 0.012, 0.2);
            if (res < 0.001 || t > tmax) break;
        }
    }

    return clamp(res, 0.0, 1.0);
}

float4 intersect(float3 ro, float3 rd, float time) {
    float4 res = float4(-1.0);

    // bounding sphere
    float2 tminmax = iSphere(ro, rd, 0.535);
    if (tminmax.y > 0.0) {
        // raymarch
        float t = max(tminmax.x, 0.001);
        for (int i = 0; i < 128 && t < tminmax.y; i++) {
            float4 h = map63(ro + t * rd, time);
            if (h.x < 0.001) { res = float4(t, h.yzw); break; }
            t += h.x;
        }
    }

    return res;
}

float3x3 setCamera(float3 ro, float3 ta, float cr) {
    float3 cw = normalize(ta - ro);
    float3 cp = float3(sin(cr), cos(cr), 0.0);
    float3 cu = normalize(cross(cw,cp));
    float3 cv =          (cross(cu,cw));
    return float3x3(cu, cv, cw);
}

fragment float4 shader_day63(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> rustyMetalTexture [[texture(5)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float3 tot = float3(0.0);
    float2 p = (2.0 * pixPos.xy - res.xy) / res.y;
    p.y *= -1.0;

    // camera
    float an = 6.2831 * time / 40.0;
    float3 ta = float3(0.0, 0.0, 0.0);
    float3 ro = ta + float3(1.3 * cos(an), 0.5, 1.2 * sin(an));

    ro += 0.005 * sin(92.0 * time / 40.0 + float3(0.0, 1.0, 3.0));
    ta += 0.009 * sin(68.0 * time / 40.0 + float3(2.0, 4.0, 6.0));

    // camera-to-world transformation
    float3x3 ca = setCamera(ro, ta, 0.0);

    // ray direction
    float fl = 2.0;
    float3 rd = ca * normalize(float3(p, fl));

    // background
    float3 col = float3(1.0 + rd.y) * 0.03;

    // raymarch geometry
    float4 tuvw = intersect(ro, rd, time);
    if (tuvw.x > 0.0) {
        // shading/lighting
        float3 pos = ro + tuvw.x * rd;
        float3 nor = calcNormal(pos, time);

        float3 te = 1.0 * rustyMetalTexture.sample(s, tuvw.yz * 2.0).xyz + 1.0 * rustyMetalTexture.sample(s, tuvw.yw * 1.0).xyz;

        float3 mate = 0.22 * te;
        float len = length(pos);

        mate *= 1.0 + float3(2.0, 0.5, 0.0) * (1.0 - smoothstep(0.121, 0.122,len));

        float focc = 0.1 + 0.9 * clamp(0.5 + 0.5 * dot(nor, pos / len), 0.0, 1.0);
        focc *= 0.1 + 0.9 * clamp(len * 2.0, 0.0, 1.0);
        float ks = clamp(te.x * 1.5, 0.0, 1.0);
        float3  f0 = mate;
        float kd = (1.0 - ks) * 0.125;

        float occ = calcAO( pos, nor, time ) * focc;

        col = float3(0.0);

        // top
        {
            float3 lig = normalize(float3(0.8, 0.2, 0.6));
            float dif = clamp(dot(nor, lig), 0.0, 1.0);
            float3 hal = normalize(lig - rd);
            float sha = 1.0; if (dif > 0.001) sha = calcSoftshadow(pos + 0.001 * nor, lig, 20.0, time);
            float3 spe = pow(clamp(dot(nor, hal), 0.0, 1.0), 16.0) * (f0 + (1.0 - f0) * pow(clamp(1.0 + dot(hal, rd), 0.0, 1.0), 5.0));
            col += kd * mate * 2.0 * float3(1.00, 0.70, 0.50) * dif * sha;
            col += ks *        2.0 * float3(1.00, 0.80, 0.70) * dif * sha * spe * 3.14;
        }

        // side
        {
            float3  ref = reflect(rd,nor);
            float fre = clamp(1.0+dot(nor,rd),0.0,1.0);
            float sha = occ;
            col += kd * mate * 25.0 * float3(0.19, 0.22, 0.24) * (0.6 + 0.4 * nor.y) * sha;
            col += ks * 25.0 * float3(0.19, 0.22, 0.24) * sha * smoothstep(-1.0 + 1.5 * focc, 1.0 - 0.4 * focc, ref.y) * (f0 + (1.0 - f0) * pow(fre, 5.0));
        }

        // bottom
        {
            float dif = clamp(0.4 - 0.6 * nor.y, 0.0, 1.0);
            col += kd * mate * 5.0 * float3(0.25, 0.20, 0.15) * dif * occ;
        }
    }

    // vignetting
    col *= 1.0 - 0.1 * dot(p, p);

    // gamma
    tot += pow(col, float3(0.45));

    // s-curve
    tot = min(tot, 1.0);
    tot = tot * tot * (3.0 - 2.0 * tot);

    // cheap dithering
    tot += sin(pixPos.x * 114.0) * sin(pixPos.y * 211.1) / 512.0;

    return float4(tot, 1.0);
}

// MARK: - Day64

// https://www.shadertoy.com/view/4df3D8 点描画

float brush(float col, float2 p, float4 b, float an, float time) {
    p += an * cos(time + 100.0 * b.yz);
    float2 dd = p - b.yz;
    col = mix(col, b.x, exp(-b.w * b.w * dot(dd, dd)));
    if (abs(b.z - 0.5) < 0.251) {
        dd.x = p.x - 1.0 + b.y;
        col = mix(col, b.x, exp(-b.w * b.w * dot(dd, dd)));
    }
    return col;
}

fragment float4 shader_day64(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = pixPos.xy / res.y;
    uv.x -= 0.5 * (res.x / res.y - 1.0);
    uv.y *= -1.0;
    uv.y += 1.0;

    float an = smoothstep(0.0, 1.0, cos(time));
    float col = 0.0;

    col = brush( col, uv, float4(1.000,0.371,0.379,11.770), an, time);
    col = brush( col, uv, float4(0.992,0.545,0.551,8.359), an, time);
    col = brush( col, uv, float4(0.749,0.623,0.990,36.571), an, time);
    col = brush( col, uv, float4(1.000,0.510,0.395,11.315), an, time);
    col = brush( col, uv, float4(1.000,0.723,0.564,15.170), an, time);
    col = brush( col, uv, float4(0.953,0.729,0.750,14.629), an, time);
    col = brush( col, uv, float4(0.706,0.982,0.033,16.254), an, time);
    col = brush( col, uv, float4(1.000,0.855,0.652,26.256), an, time);
    col = brush( col, uv, float4(1.000,0.664,0.623,81.920), an, time);
    col = brush( col, uv, float4(0.000,0.881,0.750,8.031), an, time);
    col = brush( col, uv, float4(0.686,0.682,0.900,27.676), an, time);
    col = brush( col, uv, float4(1.000,0.189,0.684,18.618), an, time);
    col = brush( col, uv, float4(0.000,0.904,0.750,8.031), an, time);
    col = brush( col, uv, float4(1.000,0.422,0.195,44.522), an, time);
    col = brush( col, uv, float4(1.000,0.779,0.750,16.787), an, time);
    col = brush( col, uv, float4(1.000,0.645,0.330,14.222), an, time);
    col = brush( col, uv, float4(1.000,0.197,0.648,22.505), an, time);
    col = brush( col, uv, float4(0.702,0.512,0.393,35.310), an, time);
    col = brush( col, uv, float4(1.000,0.744,0.621,14.949), an, time);
    col = brush( col, uv, float4(0.671,0.885,0.092,24.675), an, time);
    col = brush( col, uv, float4(0.000,0.344,0.750,8.031), an, time);
    col = brush( col, uv, float4(1.000,0.760,0.465,40.960), an, time);
    col = brush( col, uv, float4(0.008,0.908,0.311,8.031), an, time);
    col = brush( col, uv, float4(0.016,0.959,0.750,10.039), an, time);
    col = brush( col, uv, float4(0.004,0.930,0.750,12.800), an, time);
    col = brush( col, uv, float4(1.000,0.555,0.250,19.883), an, time);
    col = brush( col, uv, float4(1.000,0.770,1.018,15.876), an, time);
    col = brush( col, uv, float4(0.000,0.828,0.756,36.571), an, time);
    col = brush( col, uv, float4(0.580,0.566,0.424,89.043), an, time);
    col = brush( col, uv, float4(0.988,0.162,0.691,40.157), an, time);
    col = brush( col, uv, float4(0.000,0.314,0.750,8.031), an, time);
    col = brush( col, uv, float4(0.000,0.947,0.125,32.000), an, time);
    col = brush( col, uv, float4(0.914,0.844,0.725,52.513), an, time);
    col = brush( col, uv, float4(1.000,0.313,0.762,42.667), an, time);
    col = brush( col, uv, float4(0.996,0.676,0.689,85.333), an, time);
    col = brush( col, uv, float4(0.980,0.346,0.559,24.675), an, time);
    col = brush( col, uv, float4(1.000,0.553,0.250,18.789), an, time);
    col = brush( col, uv, float4(0.004,0.258,0.248,8.031), an, time);
    col = brush( col, uv, float4(1.000,0.420,0.742,30.567), an, time);
    col = brush( col, uv, float4(0.906,0.543,0.250,22.756), an, time);
    col = brush( col, uv, float4(0.863,0.674,0.322,20.078), an, time);
    col = brush( col, uv, float4(0.753,0.357,0.686,78.769), an, time);
    col = brush( col, uv, float4(0.906,0.795,0.705,37.236), an, time);
    col = brush( col, uv, float4(0.933,0.520,0.365,38.642), an, time);
    col = brush( col, uv, float4(0.996,0.318,0.488,14.734), an, time);
    col = brush( col, uv, float4(0.337,0.486,0.281,81.920), an, time);
    col = brush( col, uv, float4(0.965,0.691,0.516,16.650), an, time);
    col = brush( col, uv, float4(0.808,0.582,0.973,52.513), an, time);
    col = brush( col, uv, float4(0.012,0.240,0.928,8.063), an, time);
    col = brush( col, uv, float4(1.000,0.496,0.217,31.508), an, time);
    col = brush( col, uv, float4(0.000,0.658,0.953,34.133), an, time);
    col = brush( col, uv, float4(0.871,0.582,0.172,62.061), an, time);
    col = brush( col, uv, float4(0.855,0.346,0.342,17.504), an, time);
    col = brush( col, uv, float4(0.878,0.787,0.648,28.845), an, time);
    col = brush( col, uv, float4(0.000,0.984,0.111,35.310), an, time);
    col = brush( col, uv, float4(0.855,0.514,0.965,66.065), an, time);
    col = brush( col, uv, float4(0.561,0.613,0.350,81.920), an, time);
    col = brush( col, uv, float4(0.992,0.818,0.902,21.558), an, time);
    col = brush( col, uv, float4(0.914,0.746,0.615,40.157), an, time);
    col = brush( col, uv, float4(0.557,0.580,0.125,60.235), an, time);
    col = brush( col, uv, float4(0.475,0.547,0.414,70.621), an, time);
    col = brush( col, uv, float4(0.843,0.680,0.793,20.277), an, time);
    col = brush( col, uv, float4(1.000,0.230,0.758,56.889), an, time);
    col = brush( col, uv, float4(1.000,0.299,0.691,68.267), an, time);
    col = brush( col, uv, float4(0.737,0.518,0.100,68.267), an, time);
    col = brush( col, uv, float4(0.996,0.227,0.514,41.796), an, time);
    col = brush( col, uv, float4(0.929,0.850,0.770,62.061), an, time);
    col = brush( col, uv, float4(0.682,0.834,0.111,30.118), an, time);
    col = brush( col, uv, float4(0.996,0.854,0.793,58.514), an, time);
    col = brush( col, uv, float4(0.490,0.736,0.889,19.321), an, time);
    col = brush( col, uv, float4(0.980,0.465,0.725,16.126), an, time);
    col = brush( col, uv, float4(0.992,0.484,1.010,23.273), an, time);
    col = brush( col, uv, float4(0.008,0.949,0.727,23.540), an, time);
    col = brush( col, uv, float4(0.012,0.086,0.086,8.031), an, time);
    col = brush( col, uv, float4(1.000,0.121,0.750,44.522), an, time);
    col = brush( col, uv, float4(0.427,0.617,0.891,27.676), an, time);
    col = brush( col, uv, float4(0.804,0.693,0.633,78.769), an, time);
    col = brush( col, uv, float4(0.012,0.711,0.084,13.745), an, time);
    col = brush( col, uv, float4(0.082,0.584,0.338,107.789), an, time);
    col = brush( col, uv, float4(0.929,0.613,0.268,19.692), an, time);
    col = brush( col, uv, float4(0.200,0.549,0.420,128.000), an, time);
    col = brush( col, uv, float4(1.000,0.402,0.717,26.947), an, time);
    col = brush( col, uv, float4(0.000,0.551,0.168,45.511), an, time);
    col = brush( col, uv, float4(0.992,0.627,0.621,56.889), an, time);
    col = brush( col, uv, float4(0.902,0.361,0.748,40.960), an, time);
    col = brush( col, uv, float4(0.984,0.344,0.754,38.642), an, time);
    col = brush( col, uv, float4(0.902,0.203,0.818,51.200), an, time);
    col = brush( col, uv, float4(1.000,0.230,0.803,52.513), an, time);
    col = brush( col, uv, float4(0.922,0.738,0.691,47.628), an, time);
    col = brush( col, uv, float4(0.000,0.385,0.797,43.574), an, time);
    col = brush( col, uv, float4(0.000,0.725,0.305,62.061), an, time);
    col = brush( col, uv, float4(0.000,0.150,0.750,45.511), an, time);
    col = brush( col, uv, float4(1.000,0.742,0.408,47.628), an, time);
    col = brush( col, uv, float4(0.000,0.645,0.643,60.235), an, time);
    col = brush( col, uv, float4(1.000,0.645,0.438,35.310), an, time);
    col = brush( col, uv, float4(0.510,0.564,0.789,18.450), an, time);
    col = brush( col, uv, float4(0.863,0.211,0.781,30.567), an, time);
    col = brush( col, uv, float4(0.106,0.508,0.328,89.043), an, time);
    col = brush( col, uv, float4(0.012,0.410,0.875,14.629), an, time);
    col = brush( col, uv, float4(1.000,0.871,0.877,48.762), an, time);
    col = brush( col, uv, float4(1.000,0.258,0.779,37.926), an, time);
    col = brush( col, uv, float4(0.000,0.436,0.807,28.845), an, time);
    col = brush( col, uv, float4(0.918,0.861,0.836,49.951), an, time);
    col = brush( col, uv, float4(1.000,0.291,0.770,40.960), an, time);
    col = brush( col, uv, float4(0.000,0.750,0.283,27.676), an, time);
    col = brush( col, uv, float4(0.965,0.596,0.572,28.055), an, time);
    col = brush( col, uv, float4(0.902,0.803,0.953,24.976), an, time);
    col = brush( col, uv, float4(0.957,0.498,0.600,16.126), an, time);
    col = brush( col, uv, float4(0.914,0.322,0.432,15.634), an, time);
    col = brush( col, uv, float4(0.008,0.025,0.621,17.809), an, time);
    col = brush( col, uv, float4(0.000,0.916,0.713,56.889), an, time);
    col = brush( col, uv, float4(0.914,0.547,0.971,47.628), an, time);
    col = brush( col, uv, float4(0.000,0.207,0.432,37.926), an, time);
    col = brush( col, uv, float4(0.875,0.176,0.793,46.545), an, time);
    col = brush( col, uv, float4(0.000,0.646,0.668,41.796), an, time);
    col = brush( col, uv, float4(1.000,0.721,0.691,51.200), an, time);
    col = brush( col, uv, float4(0.451,0.559,0.754,49.951), an, time);
    col = brush( col, uv, float4(0.969,0.846,0.750,58.514), an, time);
    col = brush( col, uv, float4(0.000,0.900,0.146,36.571), an, time);
    col = brush( col, uv, float4(1.000,0.613,0.635,85.333), an, time);
    col = brush( col, uv, float4(0.596,0.807,0.150,58.514), an, time);
    col = brush( col, uv, float4(0.898,0.330,0.760,40.157), an, time);
    col = brush( col, uv, float4(0.694,0.594,0.012,51.200), an, time);
    col = brush( col, uv, float4(0.698,0.592,0.055,53.895), an, time);
    col = brush( col, uv, float4(0.902,0.268,0.773,39.385), an, time);
    col = brush( col, uv, float4(0.925,0.838,0.660,58.514), an, time);
    col = brush( col, uv, float4(0.843,0.670,0.242,28.444), an, time);
    col = brush( col, uv, float4(0.243,0.465,0.285,85.333), an, time);
    col = brush( col, uv, float4(0.816,0.588,0.674,44.522), an, time);
    col = brush( col, uv, float4(0.008,0.283,0.115,8.031), an, time);
    col = brush( col, uv, float4(0.247,0.414,0.691,60.235), an, time);
    col = brush( col, uv, float4(1.000,0.104,0.781,60.235), an, time);
    col = brush( col, uv, float4(0.000,0.619,0.660,60.235), an, time);
    col = brush( col, uv, float4(0.584,0.650,0.994,46.545), an, time);
    col = brush( col, uv, float4(0.000,0.219,0.393,36.571), an, time);
    col = brush( col, uv, float4(1.000,0.307,0.645,97.524), an, time);
    col = brush( col, uv, float4(0.953,0.639,0.771,38.642), an, time);
    col = brush( col, uv, float4(0.000,0.238,0.357,34.712), an, time);
    col = brush( col, uv, float4(0.922,0.713,0.352,53.895), an, time);
    col = brush( col, uv, float4(0.965,0.387,0.748,43.574), an, time);
    col = brush( col, uv, float4(0.000,0.898,0.633,41.796), an, time);
    col = brush( col, uv, float4(0.941,0.352,0.488,14.734), an, time);
    col = brush( col, uv, float4(0.933,0.439,0.725,30.567), an, time);
    col = brush( col, uv, float4(0.310,0.541,0.906,47.628), an, time);
    col = brush( col, uv, float4(0.941,0.502,0.689,24.094), an, time);
    col = brush( col, uv, float4(0.094,0.527,0.330,85.333), an, time);
    col = brush( col, uv, float4(0.000,0.090,0.688,55.351), an, time);
    col = brush( col, uv, float4(0.000,0.652,0.713,75.852), an, time);
    col = brush( col, uv, float4(0.949,0.320,0.623,107.789), an, time);
    col = brush( col, uv, float4(0.890,0.775,0.750,22.505), an, time);
    col = brush( col, uv, float4(0.012,0.918,0.490,14.322), an, time);
    col = brush( col, uv, float4(1.000,0.871,0.967,58.514), an, time);
    col = brush( col, uv, float4(0.000,0.324,0.676,64.000), an, time);
    col = brush( col, uv, float4(0.008,0.141,0.248,8.031), an, time);
    col = brush( col, uv, float4(0.000,0.633,0.707,75.852), an, time);
    col = brush( col, uv, float4(0.910,0.385,0.207,44.522), an, time);
    col = brush( col, uv, float4(0.012,0.703,0.182,31.508), an, time);
    col = brush( col, uv, float4(0.000,0.617,0.703,73.143), an, time);
    col = brush( col, uv, float4(0.890,0.352,0.225,45.511), an, time);
    col = brush( col, uv, float4(0.933,0.826,0.604,44.522), an, time);
    col = brush( col, uv, float4(0.914,0.777,0.574,25.924), an, time);
    col = brush( col, uv, float4(0.631,0.781,0.182,68.267), an, time);
    col = brush( col, uv, float4(1.000,0.873,0.916,48.762), an, time);
    col = brush( col, uv, float4(0.694,0.520,0.113,81.920), an, time);
    col = brush( col, uv, float4(0.000,0.900,0.926,58.514), an, time);
    col = brush( col, uv, float4(0.184,0.598,0.344,146.286), an, time);
    col = brush( col, uv, float4(0.863,0.678,0.250,35.310), an, time);
    col = brush( col, uv, float4(0.090,0.566,0.332,78.769), an, time);
    col = brush( col, uv, float4(0.420,0.445,0.301,56.889), an, time);
    col = brush( col, uv, float4(0.973,0.617,0.516,18.124), an, time);
    col = brush( col, uv, float4(0.000,0.191,0.500,39.385), an, time);
    col = brush( col, uv, float4(0.000,0.240,0.326,31.508), an, time);
    col = brush( col, uv, float4(0.000,0.264,0.322,55.351), an, time);
    col = brush( col, uv, float4(0.000,0.604,0.699,70.621), an, time);
    col = brush( col, uv, float4(0.000,0.113,0.604,43.574), an, time);
    col = brush( col, uv, float4(0.894,0.760,0.697,49.951), an, time);
    col = brush( col, uv, float4(0.914,0.725,0.383,55.351), an, time);
    col = brush( col, uv, float4(0.000,0.199,0.467,48.762), an, time);
    col = brush( col, uv, float4(0.000,0.904,0.660,52.513), an, time);
    col = brush( col, uv, float4(0.922,0.611,0.191,45.511), an, time);
    col = brush( col, uv, float4(0.059,0.789,0.869,30.118), an, time);
    col = brush( col, uv, float4(0.976,0.641,0.213,40.960), an, time);
    col = brush( col, uv, float4(0.918,0.402,0.742,47.628), an, time);
    col = brush( col, uv, float4(0.945,0.717,0.582,40.157), an, time);
    col = brush( col, uv, float4(0.000,0.299,0.672,58.514), an, time);
    col = brush( col, uv, float4(0.000,0.719,0.666,48.762), an, time);
    col = brush( col, uv, float4(0.882,0.697,0.271,58.514), an, time);
    col = brush( col, uv, float4(0.929,0.752,0.436,64.000), an, time);
    col = brush( col, uv, float4(1.000,0.867,0.813,56.889), an, time);
    col = brush( col, uv, float4(0.643,0.588,0.090,64.000), an, time);
    col = brush( col, uv, float4(0.012,0.063,0.922,10.952), an, time);
    col = brush( col, uv, float4(0.878,0.186,0.750,31.508), an, time);
    col = brush( col, uv, float4(0.953,0.648,0.613,120.471), an, time);
    col = brush( col, uv, float4(0.973,0.180,0.576,45.511), an, time);
    col = brush( col, uv, float4(0.741,0.943,0.076,52.513), an, time);
    col = brush( col, uv, float4(0.059,0.545,0.332,89.043), an, time);
    col = brush( col, uv, float4(0.094,0.295,0.734,85.333), an, time);
    col = brush( col, uv, float4(0.008,0.676,0.721,85.333), an, time);
    col = brush( col, uv, float4(0.550,0.350,0.650,85.000), an, time);

    return float4(col, col, col, 1.0);
}

// MARK: - Day65

// https://www.shadertoy.com/view/ltj3Wc 毛筆

float opU( float d1, float d2 ){ return min(d1,d2); }
float opS( float d2, float d1 ){ return max(-d1,d2); }
float opI( float d1, float d2) { return max(d1,d2); }

float magicBox(float3 p) {
    const int MAGIC_BOX_ITERS = 13;
    const float MAGIC_BOX_MAGIC = 0.55;
    p = 1.0 - abs(1.0 - mod(p, 2.0));
    float lastLength = length(p);
    float tot = 0.0;
    for (int i = 0; i < MAGIC_BOX_ITERS; i++) {
        p = abs(p) / (lastLength * lastLength) - MAGIC_BOX_MAGIC;
        float newLength = length(p);
        tot += abs(newLength - lastLength);
        lastLength = newLength;
    }
    return tot;
}

float magicBox(float2 uv) {
    const float3x3 M = float3x3(0.28862355854826727, 0.6997227302779844, 0.6535170557707412,
                                0.06997493955670424, 0.6653237235314099, -0.7432683571499161,
                                -0.9548821651308448, 0.26025457467376617, 0.14306504491456504);
    float3 p = 0.5 * M * float3(uv, 0.0);
    return magicBox(p);
}

float2x2 rot2D(float r) {
    float c = cos(r);
    float s = sin(r);
    return float2x2(c, s, -s, c);
}

float nsin(float a) { return 0.5 + 0.5 * sin(a); }
float ncos(float a) { return 0.5 + 0.5 * cos(a); }

float rand65(float2 co) {
    return fract(sin(dot(co.xy ,float2(12.9898,78.233))) * 43758.5453);
}

float rand65(float n) {
    return fract(cos(n * 89.42) * 343.42);
}

float dtoa(float d, float amount) {
    return clamp(1.0 / (clamp(d, 1.0 / amount, 1.0) * amount), 0.0, 1.0);
}

float sdAxisAlignedRect(float2 uv, float2 tl, float2 br) {
    float2 d = max(tl - uv, uv - br);
    return length(max(float2(0.0), d)) + min(0.0, max(d.x, d.y));
}

float sdCircle(float2 uv, float2 origin, float radius) {
    return length(uv - origin) - radius;
}

float smoothstep4(float e1, float e2, float e3, float e4, float val) {
    return min(smoothstep(e1, e2, val), 1.0 - smoothstep(e3, e4, val));
}

float2 hash65(float2 p) {
    p = float2(dot(p, float2(127.1, 311.7)),
               dot(p, float2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise65(float2 p) {
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;
    float2 i = floor(p + (p.x + p.y) * K1);
    float2 a = p - i + (i.x + i.y) * K2;
    float2 o = (a.x > a.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float2 b = a - o + K2;
    float2 c = a - 1.0 + 2.0 * K2;
    float3 h = max(0.5 - float3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    float3 n = h * h * h * h * float3(dot(a, hash65(i + 0.0)), dot(b, hash65(i + o)), dot(c, hash65(i + 1.0)));
    return dot(n, float3(70.0));
}

float noise01(float2 p) {
    return clamp((noise65(p) + 0.5) * 0.5, 0.0, 1.0);
}

float3 dtocolor(float3 inpColor, float dist) {
    float3 ret;
    if (dist > 0.0)
        ret = mix(float3(0,0, 0.5), float3(0.5, 0.5, 1.0), sin(dist * M_PI_F * 2 * 50.0));
    else
        ret = mix(float3(1.0, 0.5, 0.5), float3(0.5, 0.0, 0.0), sin(dist * M_PI_F * 2 * 50.0));
    ret = mix(ret, float3(0), clamp(abs(dist), 0.0, 1.0));
    return ret;
}

float smoothf(float x) {
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

float3 colorBrushStroke(float2 res, float2 uvLine, float2 uvPaper, float2 lineSize, float sdGeometry, float3 inpColor, float4 brushColor) {
    float posInLineY = (uvLine.y / lineSize.y);

    if (posInLineY > 0.0) {
        float mouseX = 0.2;//iMouse.x == 0. ? 0.2 : (iMouse.x / iResolution.x);
        posInLineY = pow(posInLineY, (pow(mouseX, 2.0) * 15.0) + 1.5);
    }

    float strokeBoundary = dtoa(sdGeometry, 300.0);
    float strokeTexture = 0.0
    + noise01(uvLine * float2(min(res.y, res.x) * 0.2, 1.0))
    + noise01(uvLine * float2(79.0, 1.0))
    + noise01(uvLine * float2(14.0, 1.0))
    ;
    strokeTexture *= 0.333 * strokeBoundary;
    strokeTexture = max(0.008, strokeTexture);
    float strokeAlpha = pow(strokeTexture, max(0.,posInLineY) + 0.09);
    const float strokeAlphaBoost = 1.09;
    if(posInLineY > 0.)
        strokeAlpha = strokeAlphaBoost * max(0., strokeAlpha - pow(posInLineY,0.5));
    else
        strokeAlpha *= strokeAlphaBoost;

    strokeAlpha = smoothf(strokeAlpha);

    float paperBleedAmt = 60.0 + (rand65(uvPaper.y) * 30.0) + (rand65(uvPaper.x) * 30.0);

    float alpha = strokeAlpha * brushColor.a * dtoa(sdGeometry, paperBleedAmt);
    alpha = clamp(alpha, 0.0, 1.0);
    return mix(inpColor, brushColor.rgb, alpha);
}

float3 colorBrushStrokeLine(float2 res, float2 uv, float3 inpColor, float4 brushColor, float2 p1_, float2 p2_, float lineWidth) {
    float lineAngle = M_PI_F - atan2(p1_.x - p2_.x, p1_.y - p2_.y);
    float2x2 rotMat = rot2D(lineAngle);
    float lineLength = distance(p2_, p1_);
    float2 tl = (p1_ * rotMat);
    float2 br = tl + float2(0.0, lineLength);
    float2 uvLine = uv * rotMat;

    lineWidth *= mix(1.0, 0.9, smoothstep(tl.y, br.y, uvLine.y));

    uvLine.x += (noise01(uvLine * 1.0) - 0.5) * 0.02;
    uvLine.x += cos(uvLine.y * 3.0) * 0.009;
    uvLine.x += (noise01(uvLine * 5.0) - 0.5) * 0.005;

    float d = sdAxisAlignedRect(uvLine, tl, br) - lineWidth / 2.0;
    uvLine = tl - uvLine;

    float2 lineSize = float2(lineWidth, lineLength);
    float3 ret = colorBrushStroke(res, float2(uvLine.x, -uvLine.y), uv, lineSize, d, inpColor, brushColor);
    return ret;
}

float3 humanizeBrushStrokeDonut(float2 uvLine, float radius_, bool clockwise, float lineLength) {
    float2 humanizedUVLine = uvLine;

    float twistAmt = 0.24;
    float linePosY = humanizedUVLine.y / lineLength;
    humanizedUVLine.x += linePosY * twistAmt;

    float humanizedRadius = radius_;
    humanizedRadius += (noise01(uvLine * 1.0) - 0.5) * 0.04;
    humanizedRadius += sin(uvLine.y * 3.0) * 0.019;
    humanizedUVLine.x += sin(uvLine.x * 30.0) * 0.02;
    humanizedUVLine.x += (noise01(uvLine * 5.0) - 0.5) * 0.005;

    return float3(humanizedUVLine, humanizedRadius);
}

float3 colorBrushStrokeDonut(float2 res, float2 uv, float3 inpColor, float4 brushColor, float2 o, float radius_, float angleStart, float sweepAmt, float lineWidth, bool clockwise) {
    float2 uvLine = uv - o;
    float angle = atan2(uvLine.x, uvLine.y) + M_PI_F;
    angle = mod(angle - angleStart + M_PI_F, M_PI_F * 2);
    if (!clockwise)
        angle = M_PI_F * 2 - angle;
    float lineLength = radius_ * M_PI_F * 2;
    uvLine = float2(radius_ - length(uvLine), angle / (M_PI_F * 2 * lineLength));

    float lineWidth1 = lineWidth * mix(1.0, 0.9, smoothstep(0.0, lineLength, uvLine.y));

    float3 hu = humanizeBrushStrokeDonut(uvLine, radius_, clockwise, lineLength);
    float2 humanizedUVLine = hu.xy;
    float humanizedRadius = hu.z;

    float d = opS(sdCircle(uv, o, humanizedRadius),
                  sdCircle(uv, o, humanizedRadius));
    d -= lineWidth1 * 0.5;

    float3 ret = colorBrushStroke(res, humanizedUVLine, uv, float2(lineWidth1, lineLength), d, inpColor, brushColor);

    float3 ret2 = float3(1.0);
    if (angle > M_PI_F) {
        uvLine.y -= lineLength;
        hu = humanizeBrushStrokeDonut(uvLine, radius_, clockwise, lineLength);
        humanizedUVLine = hu.xy;
        humanizedRadius = hu.z;
        float2 strokeStartPos = o + float2(sin(angleStart), cos(angleStart)) * humanizedRadius;
        d = distance(uv, strokeStartPos);
        d -= lineWidth * 0.5 * 1.0;
        ret2 = colorBrushStroke(res, humanizedUVLine, uv, float2(lineWidth, lineLength), d, inpColor, brushColor);
    }
    return min(ret, ret2);
}

float2 getuv_centerX(float2 res, float2 pixPos, float2 newTL, float2 newSize) {
    float2 ret = float2(pixPos.x / res.x, (res.y - pixPos.y) / res.y);
    ret *= newSize;
    float aspect = res.x / res.y;
    ret.x *= aspect;
    float newWidth = newSize.x * aspect;
    return ret + float2(newTL.x - (newWidth - newSize.x) / 2.0, newTL.y);
}

fragment float4 shader_day65(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = getuv_centerX(res, pixPos.xy, float2(-1.0, -1.0), float2(2.0, 2.0));
    uv.y *= -1.0;

    float2 uvSignature = (pixPos.xy / res.y * 2.0) - 1.0;
    uvSignature.y *= -1.0;

    float3 col = float3(1.0, 1.0, 0.875);
    float dist;

    float yo = sin(-uv.x * M_PI_F * 0.5) * 0.2;
    col = colorBrushStrokeLine(res, uv, col, float4(float3(0.8, 0.1, 0.0), 0.9),
                               float2(-1.4, -0.4 + yo),
                               float2(2.6, -0.4 + yo), 0.3);

    col = colorBrushStrokeLine(res, uv, col, float4(float3(0.8, 0.1, 0.0), 0.4),
                               float2(1.3, 0.0 + yo),
                               float2(-2.9, 0.0 + yo), 0.03);

    col = colorBrushStrokeLine(res, uv, col, float4(float3(0.8, 0.1, 0.0), 0.52),
                               float2(1.3, 0.3 + yo + (cos(uv.x * 12.0) * 0.025)),
                               float2(-2.9, 0.3 + yo), 0.1);


    col = colorBrushStrokeDonut(res, uv, col, float4(0.0, 0.0, 0.0, 0.9),
                                float2(0.0, 0.0),// origin
                                0.6,// radius
                                0.2,// angle of brush start
                                0.5,// sweep amt 0-1
                                0.3,// width
                                false);// clockwise

    // paint blotches
    float blotchAmt = smoothstep(20.0, 50.0, magicBox((uv + 12.0) * 2.0));
    blotchAmt = pow(blotchAmt, 3.0);
    blotchAmt = 0.7 * smoothstep(0.2, 0.4, blotchAmt);
    col *= 1.0 - blotchAmt;

    // signature
    dist = sdAxisAlignedRect(uvSignature, float2(-0.68), float2(-0.55));
    float amt = 90.0 + (rand65(uvSignature.y) * 100.0) + (rand65(uvSignature.x / 4.0) * 90.0);
    float vary = sin(uvSignature.x * uvSignature.y * 50.0) * 0.0047;
    dist = opS(dist - 0.028 + vary, dist - 0.019 - vary);
    col = mix(col, float3(0.8, 0.1, 0.0), dtoa(dist, amt) * 0.3);
    col = mix(col, float3(0.8, 0.1, 0.0), dtoa(dist, amt * 4.0) * 0.95);

    // grain
    col.rgb += (rand65(uv) - 0.5) * 0.08;
    col.rgb = clamp(col.rgb, 0.0, 1.0);

    // vignette
    float2 uvScreen = (pixPos.xy / res.xy * 2.0) - 1.0;
    float vignetteAmt = 1.0 - dot(uvScreen * 0.5, uvScreen * 0.62);
    col *= vignetteAmt;

    return float4(col, 1.0);
}

// MARK: - Day66

// https://www.shadertoy.com/view/4dl3zn デザイン 丸

fragment float4 shader_day66(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = -1.0 + 2.0 * pixPos.xy / res.xy;
    uv.x *= res.x / res.y;

    float3 color = float3(0.8 + 0.2 * uv.y);

    for (int i = 0; i < 40; i++) {
        float pha =     sin(float(i) * 546.13 + 1.0) * 0.5 + 0.5;
        float siz = pow(sin(float(i) * 651.74 + 5.0) * 0.5 + 0.5, 4.0);
        float pox =     sin(float(i) * 321.55 + 4.1) * res.x / res.y;

        float rad = 0.1 + 0.5*siz;
        float2 pos = float2(pox, -1.0 - rad + (2.0 + 2.0 * rad) * mod(pha + 0.1 * time * (0.2 + 0.8 * siz), 1.0));
        float dis = length(uv - pos);
        float3 col = mix(float3(0.94, 0.3, 0.0), float3(0.1, 0.4, 0.8), 0.5 + 0.5 * sin(float(i) * 1.2 + 1.9));

        float f = length(uv - pos) / rad;
        f = sqrt(clamp(1.0 -f * f, 0.0, 1.0));
        color -= col.zyx * (1.0 - smoothstep(rad * 0.95, rad, dis)) * f;
    }

    color *= sqrt(1.5 - 0.5 * length(uv));
    return float4(color, 1.0);
}

// MARK: - Day67

// https://www.shadertoy.com/view/MdSGDm Motion blur

float3 diskWithMotionBlur(float3 pcol, float2 puv, float3 dpr, float2 dv, float3 dcol) {
    float2 xc = puv - dpr.xy;
    float a = dot(dv, dv);
    float b = dot(dv, xc);
    float c = dot(xc, xc) - dpr.z * dpr.z;
    float h = b * b - a * c;
    if (h > 0.0) {
        h = sqrt( h );
        float ta = max(0.0, (-b - h) / a);
        float tb = min(1.0, (-b + h) / a);
        if (ta < tb)
            pcol = mix(pcol, dcol, clamp(2.0 * (tb - ta), 0.0, 1.0));
    }
    return pcol;
}

float3 hash3( float n ) { return fract(sin(float3(n, n + 1.0, n + 2.0)) * 43758.5453123); }
float4 hash4( float n ) { return fract(sin(float4(n, n + 1.0, n + 2.0, n + 3.0)) * 43758.5453123); }

float2 getPosition( float time, float4 id ) { return float2(       0.9 * sin((8.0 * (0.75 + 0.5 * id.z)) * time + 20.0 * id.x),        0.75 * cos(8.0 * (0.75 + 0.5 * id.w) * time + 20.0 * id.y)); }
float2 getVelocity( float time, float4 id ) { return float2( 8.0 * 0.9 * cos((8.0 * (0.75 + 0.5 * id.z)) * time + 20.0 * id.x), -8.0 * 0.75 * sin(8.0 * (0.75 + 0.5 * id.w) * time + 20.0 * id.y)); }

fragment float4 shader_day67(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 p = (2.0 * pixPos.xy - res.xy) / res.y;
    float3 col = float3(0.03) + 0.015 * p.y;

    for (int i = 0; i < 16; i++) {
        float4 off = hash4(float(i) * 13.13);
        float3 sph = float3(getPosition(time, off), 0.02 + 0.1 * off.x);
        float2 dv = getVelocity(time, off) /24.0;
        float3 sphcol = 0.55 + 0.45 * sin(3.0 * off.z + float3(4.0, 0.0, 2.0));
        col = diskWithMotionBlur(col, p, sph, dv, sphcol);
    }

    col = pow(col, float3(0.4545));
    col += (1.0 / 255.0) * hash3(p.x + 13.0 * p.y);
    return float4(col, 1.0);
}

// MARK: - Day68

// https://www.shadertoy.com/view/XsfGWB Loading

fragment float4 shader_day68(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float3 col = float3(0.2, 0.2, 0.2);

    float2 p = (-1.0 + 2.0 * pixPos.xy / res.xy) * float2(res.x / res.y, 1.0);
    float2 q = p + 0.1;

    float t = mod(time, 8.0);
    if (t < 4.0) {
        q *= 1.0 + 0.3 * q.x;
    }
    float r = length(q);
    float a = atan2(q.y, q.x) + 1.0 * time;
    col = mix(col, float3(0.1), (1.0 - smoothstep(0.0, 0.24, abs(r - 0.5))) * smoothstep(-1.0, 1.0, sin(17.0 * a)));

    if (t < 4.0) {
        p *= 1.0 + 0.3 * p.x;
    }
    r = length(p);
    a = atan2(p.y,p.x) + 1.0 * time;
    col = mix(col, float3(1.0, 1.0, 1.0), (1.0 - smoothstep(0.10, 0.14, abs(r - 0.5))) * smoothstep(0.4, 1.0, sin(17.0 * a)));

    float gl = 0.6 + 0.4 * sin(0.5 * 6.2831 * time);
    col += gl * float3(1.0, 0.5, 0.2) * (1.0 - smoothstep(0.10, 0.20, abs(r - 0.5))) * smoothstep(-1.0, 0.5, sin(17.0 * a));

    return float4(col, 1.0);
}

// MARK: - Day69

// https://www.shadertoy.com/view/MsSGD1 手書きスケッチ風

#define EDGE_WIDTH 0.15
#define RAYMARCH_ITERATIONS 40
#define SHADOW_ITERATIONS 50
#define SHADOW_STEP 1.0
#define SHADOW_SMOOTHNESS 256.0
#define SHADOW_DARKNESS 0.75

// Distance functions by www.iquilezles.org
float fSubtraction(float a, float b) {return max(-a,b);}
float fIntersection(float d1, float d2) {return max(d1,d2);}
float fUnion(float d1, float d2) {
    d1 = min(d1, d2);
    return d1;
}
float pSphere(float3 p, float s) {return length(p)-s;}
float pRoundBox(float3 p, float3 b, float r) {return length(max(abs(p)-b,0.0))-r;}
float pTorus(float3 p, float2 t) {float2 q = float2(length(p.xz)-t.x,p.y); return length(q)-t.y;}
float pTorus2(float3 p, float2 t) {float2 q = float2(length(p.xy)-t.x,p.z); return length(q)-t.y;}
float pCapsule(float3 p, float3 a, float3 b, float r) {float3 pa = p - a, ba = b - a;
float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 ); return length( pa - ba*h ) - r;}

float distf(float3 p) {
    float d = 100000.0;

    d = fUnion(d, pRoundBox(float3(0,0,10) + p, float3(21,21,1), 1.0));
    d = fUnion(d, pSphere(float3(10,10,0) + p, 8.0));
    d = fUnion(d, pSphere(float3(16,0,4) + p, 4.0));
    d = fUnion(d, pCapsule(p, float3(10,10,12), float3(15,15,-6.5), 1.5));
    d = fUnion(d, pCapsule(p, float3(10,10,12), float3(5,15,-6.5), 1.5));
    d = fUnion(d, pCapsule(p, float3(10,10,12), float3(10,5,-6.5), 1.5));
    d = fUnion(d, pTorus(float3(15,-15,0) + p, float2(6,2)));
    d = fUnion(d, pTorus2(float3(10,-15,0) + p, float2(6,2)));
    d = fUnion(d, pRoundBox(float3(-10,10,-2) + p, float3(1,1,9), 1.0));
    d = fUnion(d, pRoundBox(float3(-10,10,-4) + p, float3(0.5,6,0.5), 1.0));
    d = fUnion(d, pRoundBox(float3(-10,10,2) + p, float3(6,0.5,0.5), 1.0));

    return d;
}

float3 normal(float3 p) {
    const float eps = 0.01;
    float3 n = float3((distf(float3(p.x-eps,p.y,p.z)) - distf(float3(p.x+eps,p.y,p.z))),
                      (distf(float3(p.x,p.y-eps,p.z)) - distf(float3(p.x,p.y+eps,p.z))),
                      (distf(float3(p.x,p.y,p.z-eps)) - distf(float3(p.x,p.y,p.z+eps)))
                      );
    return normalize(n);
}

float4 raymarch(float3 from, float3 increment) {
    const float maxDist = 200.0;
    const float minDist = 0.001;
    const int maxIter = RAYMARCH_ITERATIONS;

    float dist = 0.0;

    float lastDistEval = 1e10;
    float edge = 0.0;

    for(int i = 0; i < maxIter; i++) {
        float3 pos = (from + increment * dist);
        float distEval = distf(pos);

        if (lastDistEval < EDGE_WIDTH && distEval > lastDistEval + 0.001) {
            edge = 1.0;
        }

        if (distEval < minDist) {
            break;
        }

        dist += distEval;
        if (distEval < lastDistEval) lastDistEval = distEval;
    }

    float mat = 1.0;
    if (dist >= maxDist) mat = 0.0;

    return float4(dist, mat, edge, 0);
}

float shadow(float3 from, float3 increment) {
    const float minDist = 1.0;

    float res = 1.0;
    float t = 1.0;
    for(int i = 0; i < SHADOW_ITERATIONS; i++) {
        float h = distf(from + increment * t);
        if(h < minDist)
            return 0.0;

        res = min(res, SHADOW_SMOOTHNESS * h / t);
        t += SHADOW_STEP;
    }
    return res;
}

float rand(float x) {
    return fract(sin(x) * 43758.5453);
}

float triangle(float x) {
    return abs(1.0 - mod(abs(x), 2.0)) * 2.0 - 1.0;
}

float4 getPixel(float2 p, float3 from, float3 increment, float3 light, float time) {
    float4 c = raymarch(from, increment);
    float3 hitPos = from + increment * c.x;
    float3 normalDir = normal(hitPos);

    float diffuse = 1.0 + min(0.0, dot(normalDir, -light));
    float inshadow = 0.0;

    diffuse = max(diffuse, inshadow);

    if (c.y == 0.0) diffuse = min(pow(length(p), 4.0) * 0.125,1.0);

    float xs = (rand(time * 6.6) * 0.1 + 0.9);
    float ys = (rand(time * 6.6) * 0.1 + 0.9);
    float hatching = max((clamp((sin(p.x * xs * (170.0 + rand(time) * 30.0) +
                                     p.y * ys * (110.0 + rand(time * 1.91) * 30.0)) * 0.5 + 0.5) -
                                (1.0 - diffuse), 0.0, 1.0)),
                         (clamp((sin(p.x * xs * (-110.0 + rand(time * 4.74) * 30.0) +
                                     p.y * ys * (170.0 + rand(time * 3.91) * 30.0)) * 0.5 + 0.5) -
                                (1.0 - diffuse) - 0.4, 0.0, 1.0)));

    float4 mCol = mix(float4(1.0, 0.9, 0.8, 1.0), float4(1.0, 0.9, 0.8, 1.0) * 0.5, hatching);

    return mix(mCol,float4(1.0, 0.9, 0.8, 1.0) * 0.5,c.z);
}

fragment float4 shader_day69(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time2 [[buffer(1)]]) {

    float time = floor(time2 * 16.0) / 16.0;
    float2 q = pixPos.xy / res.xy;
    float2 p = -1.0 + 2.0 * q;
    p.x *= - res.x / res.y;
    p += float2(triangle(p.y * rand(time) * 4.0) * rand(time * 1.9) * 0.015,
                triangle(p.x * rand(time * 3.4) * 4.0) * rand(time * 2.1) * 0.015);
    p += float2(rand(p.x * 3.1 + p.y * 8.7) * 0.01,
                rand(p.x * 1.1 + p.y * 6.7) * 0.01);

    p.y *= -1.0;

    float2 m = float2(time * 0.06 + 1.67, 0.78);
    m = -1.0 + 2.0 * m;
    m *= float2(4.0,-0.75);
    m.y += 0.75;

    // camera position
    float dist = 50.0;
    float3 ta = float3(0.0, 0.0, 0.0);
    float3 ro = float3(cos(m.x) * cos(m.y) * dist, sin(m.x) * cos(m.y) * dist, sin(m.y) * dist);
    float3 light = float3(cos(m.x - 2.27) * 50.0, sin(m.x - 2.27) * 50.0, -20.0);

    // camera direction
    float3 cw = normalize(ta - ro);
    float3 cp = float3(0.0, 0.0, 1.0);
    float3 cu = normalize(cross(cw, cp));
    float3 cv = normalize(cross(cu, cw));
    float3 rd = normalize(p.x * cu + p.y * cv + 2.5 * cw);

    // calculate color
    float4 col = getPixel(p, ro, rd, normalize(light), time);
    return col;
}

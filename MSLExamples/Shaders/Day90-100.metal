#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day90

// https://www.shadertoy.com/view/3dSXD1 paper city

#define MARCH_STEPS 26
#define SHADOW_STEPS 1

float2x2 rot90(float a) {
    float ca = cos(a);
    float sa = sin(a);
    return float2x2(ca, sa, -sa, ca);
}

float box90(float3 p, float3 s) {
    float3 ap = abs(p) - s;
    return max(ap.x, max(ap.y, ap.z));
}

float tri(float3 p, float3 s) {
    p.y = -p.y;
    p.xz = abs(p.xz);
    return max(max(-p.y - s.y, dot(p.xy, float2(0.7)) - s.x), p.z - s.z);
}

float cone(float3 p, float a, float b) {
    return max(length(p.xz) - p.y * a, p.y - b);
}

float3 rep(float3 p, float3 s) {
    return (fract(p / s + 0.5) - 0.5) * s;
}

float2 rep(float2 p, float2 s) {
    return (fract(p / s + 0.5) - 0.5) * s;
}

float rep(float p, float s) {
    return (fract(p / s + 0.5) - 0.5) * s;
}

float house(float3 p, float s) {
    float t = tri(p + float3(0, 3, 0) * s, float3(1, 1, 3.5) * s);
    t = min(t, box90(p, float3(2, 2, 3) * s));
    return t;
}

float minitower(float3 p) {
    p.y += 5.0;
    float3 p2 = p;
    if (abs(p2.x) < abs(p2.z)) p2.xz = p2.zx;
    float t = min(house(p + float3(0, 3, 0), 0.5), house(p2, 1.0));
    t = min(t, house(p - float3(0, 5, 0), 1.5));
    return t;
}

float tower(float3 p) {
    p.y += 15.0;
    float3 p2 = p;
    if (abs(p2.x) < abs(p2.z)) p2.xz = p2.zx;
    float t = min(house(p+float3(0,3,0),0.5), house(p2, 1.0));
    t = min(t, house(p-float3(0,5,0),1.5));
    p2.x -= sign(p2.x)*5.0;
    p2.x = abs(p2.x);
    p2.z = abs(p2.z);
    t = min(t, house(p2.zyx-float3(2,8,2),0.3));
    t = min(t, house(p2-float3(0,12,0),1.5));
    return t;
}

float wall(float3 p) {
    p.x -= cos(p.z*0.1)*2.0;
    p.x -= sin(p.z*0.03)*3.0;
    float3 rp=p;
    rp.z = rep(rp.z, 5.0);
    float w = box90(rp+float3(0,1,0), float3(2,1,50));
    rp.x = abs(rp.x)-2.0;
    float m = box90(rp-float3(0,2,0), float3(0.25,5,1.6));
    return min(w, m);
}

float field(float3 p) {
    float3 p2 = p;
    if(abs(p2.x)<abs(p2.z)) p2.xz=p2.zx;

    float tmp = box90(p2, float3(5,5,5));
    float f = max(abs(tmp-4.0), -p.y-2.0);
    f=min(f, box90(p, float3(7,0.5,7)));

    float3 p3 = p;
    p3.xz=rep(p3.xz, float2(2.5));

    float a = box90(p3, float3(0.2,2,0.2));
    a = min(a, cone(p3+float3(0,4,0), 0.3,3.0));
    f=min(f, max(a,tmp-3.8));

    return f;
}

float village(float3 p) {
    float3 p2=p;
    p2.xz = abs(p2.xz);
    float w = wall(p);
    p2.xz -= 23.0;
    float t=tower(p2);
    float3 p3 = p;
    p3.z = p3.z-4.5*sign(p.x);
    p3.x = abs(p3.x)-25.0;
    float f=field(p3);

    float res = t;
    res = min(res, w);
    res = min(res, f);

    p.z = p.z+10.0*sign(p.x);
    p.x = -abs(p.x);
    res = min(res, minitower(p+float3(29,1,0)));

    return res;
}

float map(float3 p) {
    float t1=sin(length(p.xz)*0.009);
    float s=12.0;
    for(int i=0; i<6; ++i) {
        p.xz = abs(p.xz)-s;
        p.xz = p.xz * rot90(0.55 + t1 + float(i) * 0.34);
        s /= 0.85;
    }
    p.x+=3.0;

    return min(village(p), -p.y);
}

float getao(float3 p, float3 n, float dist) {
    return clamp(map(p+n*dist)/dist, 0.0, 1.0);
}

float noise90(float2 p) {
    float2 ip=floor(p);
    p=smoothstep(0.0,1.0,fract(p));
    float2 st=float2(67,137);
    float2 v=dot(ip,st)+float2(0,st.y);
    float2 val=mix(fract(sin(v)*9875.565), fract(sin(v+st.x)*9875.565), p.x);
    return mix(val.x,val.y,p.y);
}

float fractal(float2 p) {
    float d=0.5;
    float v=0.0;
    for(int i=0; i<5; ++i) {
        v += noise90(p / d) * d;
        d *= 0.5;
    }
    return v;
}

float3 sky(float3 r, float3 l, float time) {
    float v=pow(max(dot(r,l),0.0),3.0);

    float2 sphereuv = float2(abs(atan2(r.z, r.x)) + time * 0.03, atan2(r.y, length(r.xz)));

    float skyn = fractal(sphereuv * float2(5, 10));
    float skyn2 = fractal(sphereuv * float2(5, 10) * 0.3 - float2(time * 0.06,0));
    skyn2=smoothstep(0.3,0.7,skyn2);

    float3 blue = mix(float3(0.5,0.5,0.8), float3(0.0), skyn2*skyn);

    return mix(blue*0.2, float3(1,0.7,0.4)*(skyn2*0.8+0.2), v);
}

float3 sky2(float3 r, float3 l) {
    float v = pow(max(dot(r, l), 0.0), 3.0);

    float3 blue = float3(0.5, 0.5, 0.8);

    return mix(blue*0.2, float3(1,0.7,0.4), v);
}

fragment float4 shader_day90(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    float2 uv = float2(pixPos.x / res.x, pixPos.y / res.y);
    uv -= 0.5;
    uv /= float2(res.y / res.x, 1.0);
    uv.y *= -1.0;

    float t2 = time + 10.0;
    float3 s = float3(0, 0, -100);
    s.yz = s.yz * rot90(sin(t2 * 0.3) * 0.2 + 0.5);
    s.xz = s.xz * rot90(t2 * 0.2);
    float3 t = float3(0, 30, 60);
    t.yz = t.yz * rot90(sin(t2) * 0.3 - 0.2);
    t.xz = t.xz * rot90(t2 * 0.32);

    float3 cz = normalize(t - s);
    float3 cx = normalize(cross(cz, float3(0, 1, 0)));
    float3 cy = normalize(cross(cz, cx));
    float3 r = normalize(uv.x * cx + uv.y * cy + cz * 0.7);

    float3 p = s;
    float dd = 0.0;
    for (int i = 0; i < MARCH_STEPS; ++i) {
        float d = map(p);
        if (abs(d) < 0.001) break;
        if (dd > 500.0) { dd = 500.0; break; }
        p += d * r * 0.8;
        dd += d;
    }

    float fog = 1.0 - clamp(dd / 500.0, 0.0, 1.0);

    float3 col = float3(0);
    float2 off = float2(0.01, 0);
    float3 n = normalize(map(p) - float3(map(p - off.xyy), map(p - off.yxy), map(p - off.yyx)));

    float ao = (getao(p, n, 12.0) * 0.5 + 0.5) * (getao(p, n, 2.0) * 0.3 + 0.7) * (getao(p, n, 0.5) * 0.8 + 0.2);

    float3 l = normalize(float3(-1, -2, -2.5));
    float f = pow(1.0 - abs(dot(n, r)), 3.0);

    float shad = 1.0;
    float3 sp = p + n * 0.5 - r * 0.2;
    for (int i = 0; i < SHADOW_STEPS; ++i) {
        float d = map(sp);
        if (d < 0.2) { shad = 0.0; break; }
        sp += d * l * 3.0;
    }

    col += max(0.0, dot(n, l)) * fog * float3(1, 0.7, 0.4) * 1.5 * mix(0.0, ao * 0.5 + 0.5, shad);
    col += (-n.y*0.5+0.5) * ao * fog * float3(0.5,0.5,0.8) * 0.5;
    col += sky2(reflect(r,n), l)*f*10.0*fog * (0.5+0.5*shad);

    col += sky(r, l, time) * pow(dd * 0.01, 1.4);

    col = 1.0 - exp(-col * 2.5);
    col = pow(col, float3(2.3));
    col = pow(col, float3(0.4545));

    return float4(col, 1.0);
}

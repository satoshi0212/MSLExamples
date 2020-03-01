#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Day30

float sun(float2 uv, float battery, float time) {
    float val = smoothstep(0.3, 0.29, length(uv));
    float bloom = smoothstep(0.7, 0.0, length(uv));
    float cut = 3.0 * sin((uv.y + time * 0.2 * (battery + 0.02)) * 100.0)
      + clamp(uv.y * 14.0 + 1.0, -6.0, 6.0);
    cut = clamp(cut, 0.0, 1.0);
    return clamp(val * cut, 0.0, 1.0) + bloom * 0.6;
}

float grid(float2 uv, float battery, float time) {
    float2 size = float2(uv.y, uv.y * uv.y * 0.2) * 0.01;
    uv += float2(0.0, time * 4.0 * (battery + 0.05));
    uv = abs(fract(uv) - 0.5);
    float2 lines = smoothstep(size, float2(0.0), uv);
    lines += smoothstep(size * 5.0, float2(0.0), uv) * 0.4 * battery;
    return clamp(lines.x + lines.y, 0.0, 3.0);
}

fragment float4 shader_day30(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = float2((2.0 * pixPos.xy - res) / min(res.x, res.y));
    uv.y *= -1.0;

    float battery = 1.0;
    float fog = smoothstep(0.1, -0.02, abs(uv.y + 0.2));
    float3 col = float3(0.0, 0.1, 0.2);
    if (uv.y < -0.2) {
        uv.y = 3.0 / (abs(uv.y + 0.2) + 0.05);
        uv.x *= uv.y * 1.0;
        float gridVal = grid(uv, battery, time);
        col = mix(col, float3(1.0, 0.5, 1.0), gridVal);
    } else {
        uv.y -= battery * 1.1 - 0.51;
        float2 sunUV = uv + float2(0.75, 0.2);
        col = float3(1.0, 0.2, 1.0);
        float sunVal = sun(sunUV, battery, time);
        col = mix(col, float3(1.0, 0.4, 0.1), sunUV.y * 2.0 + 0.2);
        col = mix(float3(0.0, 0.0, 0.0), col, sunVal);
        col += mix(col, mix(float3(1.0, 0.12, 0.8), float3(0.0, 0.0, 0.2),
                            clamp(uv.y * 3.5 + 3.0, 0.0, 1.0)), step(0.0, 1.0));
        col = mix(col, float3(0.0, 0.0, 0.2), 1.0 - smoothstep(0.075 - 0.0001, 0.075, 1.0));
        col += float3(1.0, 1.0, 1.0)*(1.0 - smoothstep(0.0,0.01,abs(1.0 - 0.075)));
    }

    col += fog * fog * fog;
    col = mix(float3(col.r, col.r, col.r) * 0.5, col, battery * 0.7);

    return float4(col, 1.0);
}

// MARK: - Day31

float3 rgb2hsv(float3 c) {
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float noise(float2 n) {
    const float2 d = float2(0.0, 1.0);
    float2 b = floor(n), f = smoothstep(float2(0.0), float2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

float fbm(float2 n) {
    float total = 0.0, amplitude = 1.0;
    for (int i = 0; i < 5; i++) {
        total += noise(n) * amplitude;
        n += n * 1.7;
        amplitude *= 0.47;
    }
    return total;
}

fragment float4 shader_day31(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    const float3 c1 = float3(0.5, 0.0, 0.1);
    const float3 c2 = float3(0.9, 0.1, 0.0);
    const float3 c3 = float3(0.2, 0.1, 0.7);
    const float3 c4 = float3(1.0, 0.9, 0.1);
    const float3 c5 = float3(0.1);
    const float3 c6 = float3(0.9);

    float2 speed = float2(0.0, 0);
    float shift = 1.327 + sin(time * 1.0) / 5.4;
    float alpha = 1.0;

    float dist = 3.5 - sin(time * 0.4) / 1.89;

    float2 p = pixPos.xy * dist / res.xx;
    p.y *= -1.0;
    p.x -= time / 1.1;

    float q = fbm(p - time * 0.01 + 1.0 * sin(time) / 10.0);
    float qb = fbm(p - time * 0.002 + 0.1 * cos(time) / 5.0);
    float q2 = fbm(p - time * 0.44 - 5.0 * cos(time) / 7.0) - 6.0;
    float q3 = fbm(p - time * 0.9 - 10.0 * cos(time) / 30.0) - 4.0;
    float q4 = fbm(p - time * 2.0 - 20.0 * sin(time) / 20.0) + 2.0;
    q = (q + qb - .4 * q2 -2.0 * q3  + .6 * q4) / 3.8;
    float2 r = float2(fbm(p + q / 2.0 + time * speed.x - p.x - p.y), fbm(p + q - time * speed.y));
    float3 c = mix(c1, c2, fbm(p + r)) + mix(c3, c4, r.x) - mix(c5, c6, r.y);
    float3 color = float3(c * cos(shift * pixPos.y / res.y));
    color += .05;
    color.r *= .8;
    float3 hsv = rgb2hsv(color);
    hsv.y *= hsv.z * 1.1;
    hsv.z *= hsv.y * 1.13;
    hsv.y = (2.2 - hsv.z * 0.9) * 1.20;
    color = hsv2rgb(hsv);
    return float4(color.x, color.y, color.z, alpha);
}

// MARK: - Day32

#define SMOOTH(r, R) (1.0 - smoothstep(R - 1.0, R + 1.0, r))
#define MOV(a, b, c, d, t) (float2(a * cos(t) + b * cos(0.1 * (t)), c * sin(t) + d * cos(0.1 * (t))))
#define blue1 float3(0.74, 0.95, 1.00)
#define blue3 float3(0.35, 0.76, 0.83)
#define red   float3(1.00, 0.38, 0.227)

float movingLine(float2 uv, float2 center, float radius, float time) {
    float theta0 = 90.0 * time;
    float2 d = uv - center;
    float r = sqrt(dot(d, d));
    if (r < radius) {
        float2 p = radius * float2(cos(theta0 * M_PI_F / 180.0), - sin(theta0 * M_PI_F / 180.0));
        float l = length(d - p * clamp(dot(d,p) / dot(p,p), 0.0, 1.0));
        d = normalize(d);
        float theta = mod(180.0 * atan2(d.y, d.x) / M_PI_F + theta0, 360.0);
        float gradient = clamp(1.0 - theta / 90.0, 0.0, 1.0);
        return SMOOTH(l, 1.0) + 0.5 * gradient;
    } else {
        return 0.0;
    }
}

float circle(float2 uv, float2 center, float radius, float width) {
    float r = length(uv - center);
    return SMOOTH(r - width / 2.0, radius) - SMOOTH(r + width / 2.0, radius);
}

float cross(float2 uv, float2 center, float radius) {
    float2 d = uv - center;
    int x = int(d.x);
    int y = int(d.y);
    float r = sqrt( dot( d, d ) );
    if ((r < radius) && ((x == y) || (x == -y))) {
        return 1.0;
    } else {
        return 0.0;
    }
}

float bip2(float2 uv, float2 center, float time) {
    float r = length(uv - center);
    float R = 8.0 + mod(87.0 * time, 80.0);
    return (0.5 - 0.5 * cos(30.0 * time)) * SMOOTH(r, 5.0)
        + SMOOTH(6.0, r) - SMOOTH(8.0, r)
        + smoothstep(max(8.0, R - 20.0), R, r) - SMOOTH(R, r);
}

fragment float4 shader_day32(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {
    float2 uv = pixPos.xy;
    float2 c = res.xy / 2.0;

    float3 finalColor = float3(0.3 * cross(uv, c, 240.0));
    finalColor += (circle(uv, c, 100.0, 1.0) + circle(uv, c, 165.0, 1.0)) * blue1;
    finalColor += (circle(uv, c, 240.0, 2.0));
    finalColor += movingLine(uv, c, 320.0, time) * blue3;
    finalColor += circle(uv, c, 10.0, 1.0) * blue3;

    if (length(uv - c) < 240.0) {
        float2 p = 130.0 * MOV(1.3, 1.0, 1.0, 1.4, 3.0 + 0.1 * time);
        finalColor += bip2(uv, c + p, time) * red;
    }

    return float4(finalColor, 1.0);
}

// MARK: - Day33

float2 mod(float2 x, float2 y) {
    return x - y * floor(x / y);
}

fragment float4 shader_day33(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 p = (pixPos.xy * 2.0 - res) / min(res.x, res.y);
    float2 q = mod(p, 0.2) - 0.1;
    float s = sin(time);
    float c = cos(time);
    q *= float2x2(c, s, -s, c);
    float v = 0.1 / abs(q.y) * abs(q.x);
    float r = v * abs(sin(time * 6.0) + 1.5);
    float g = v * abs(sin(time * 4.5) + 1.5);
    float b = v * abs(sin(time * 3.0) + 1.5);
    return float4(r, g, b, 1.0);
}

// MARK: - Day34

float sdTorus(float3 p, float2 t) {
    float2 q = float2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float distanceField(float3 p) {
    return -sdTorus(p.yxz, float2(5.0, 1.0));
}

float3 castRay(float3 pos, float3 dir) {
    for (int i = 0; i < 80; i++) {
        float dist = distanceField(pos);
        pos += dist * dir;
    }
    return pos;
}

float random(float2 st) {
    return fract(sin(dot(st.xy, float2(12.9898,78.233))) * 43758.5453123);
}

float pattern(float2 st, float2 v, float t) {
    float2 p = floor(st + v);
    return step(t, random(25.0 + p * 0.000004) + random(p.x) * 0.75);
}

fragment float4 shader_day34(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 screenPos = ((pixPos.xy + float2(-0.25, -0.25)) / res.xy) * 2.0 - 1.0;
    screenPos *= -1.0;

    float3 cameraPos = float3(0.0, 4.2, -3.8);
    float3 cameraDir = float3(0., 0.22, 1.3);
    float3 planeU = float3(1.0, 0.0, 0.0) * 0.8;
    float3 planeV = float3(0.0, res.y / res.x, 0.0);
    float3 rayDir = normalize(cameraDir + screenPos.x * planeU + screenPos.y * planeV);
    float3 rayPos = castRay(cameraPos, rayDir);
    float majorAngle = atan2(rayPos.z, rayPos.y);
    float minorAngle = atan2(rayPos.x, length(rayPos.yz) - 5.0);
    float2 st = float2(majorAngle / M_PI_F / 2.0, minorAngle / M_PI_F);
    float2 grid = float2(1000.0, 50.0);
    st *= grid;
    float2 ipos = floor(st);
    float2 fpos = fract(st);
    float2 vel = float2(time * 0.09 * max(grid.x, grid.y));
    vel *= float2(1.0, 0.0) * (0.4 + 2.0 * pow(random(1.0 + ipos.y), 2.0));

    float3 color = float3(0.0);
    float replaceMouse = 0.75 + 0.45 * sin(0.6 * time + 0.015 * st.x);
    color.r = pattern(st, vel, replaceMouse);
    color.g = pattern(st, vel, replaceMouse);
    color.b = pattern(st, vel, replaceMouse);

    color *= step(0.2, fpos.y);
    return float4(color, 1.0);
}

// MARK: - Day35

float distLine(float2 p, float2 a, float2 b) {
    float2 pa = p - a;
    float2 ba = b - a;
    float t = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * t);
}

float line(float2 p, float2 a, float2 b) {
    float d = distLine(p, a, b);
    float m = smoothstep(0.03, 0.01, d);
    float d2 =  length(a - b);
    m *= smoothstep(1.2, 0.8, d2) * 0.5 + smoothstep(0.05, 0.03, abs(d2 - 0.75));
    return m;
}

float distTriangle(float2 p, float2 p0, float2 p1, float2 p2 ) {
    float2 e0 = p1 - p0;
    float2 e1 = p2 - p1;
    float2 e2 = p0 - p2;

    float2 v0 = p - p0;
    float2 v1 = p - p1;
    float2 v2 = p - p2;

    float2 pq0 = v0 - e0 * clamp( dot(v0, e0) / dot(e0, e0), 0.0, 1.0 );
    float2 pq1 = v1 - e1 * clamp( dot(v1, e1) / dot(e1, e1), 0.0, 1.0 );
    float2 pq2 = v2 - e2 * clamp( dot(v2, e2) / dot(e2, e2), 0.0, 1.0 );

    float s = sign(e0.x * e2.y - e0.y * e2.x);
    float2 d = min(min(float2( dot( pq0, pq0 ), s * (v0.x * e0.y - v0.y * e0.x)),
                       float2( dot( pq1, pq1 ), s * (v1.x * e1.y - v1.y * e1.x))),
                   float2( dot( pq2, pq2 ), s * (v2.x * e2.y - v2.y * e2.x)));

    return -sqrt(d.x) * sign(d.y);
}

float N21x(float2 p) {
    p = fract(p * float2(233.34, 851.73));
    p += dot(p, p + 23.45);
    return fract(p.x * p.y);
}

float2 N22x(float2 p) {
    float n = N21x(p);
    return float2(n, N21x(p + n));
}

float2 getPos(float2 id, float2 offset, float time) {
    float2 n = N22x(id + offset) * time;
    return offset + sin(n) * 0.4;
}

float layer(float2 uv, float time) {
    float2 gv = fract(uv) - 0.5;
    float2 id = floor(uv);

    float2 p[9];
    int i = 0;
    for (float y = -1.0; y <= 1.0; y++) {
        for (float x = -1.0; x <= 1.0; x++) {
            p[i++] = getPos(id, float2(x, y), time);
        }
    }

    float t = time * 10.0;
    float m = 0.0;
    for (int i = 0; i < 9; i++) {
        m += line(gv, p[4], p[i]);
        float2 j = (p[i] - gv) * 20.0;
        float sparkle = 1.0 / dot(j, j);
        m += sparkle * (sin(t + fract(p[i].x) * 10.0) * 0.5 + 0.5);
    }
    m += line(gv, p[1], p[3]);
    m += line(gv, p[1], p[5]);
    m += line(gv, p[7], p[3]);
    m += line(gv, p[7], p[5]);

    return m;
}

fragment float4 shader_day35(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = (pixPos.xy - 0.5 * res.xy) / res.y;
    uv.y *= -1.0;

    float m = 0.0;
    float t = time * 0.1;

    float gradient = uv.y;

    float s = sin(t);
    float c = cos(t);
    float2x2 rot = float2x2(c, -s, s, c);
    uv *= rot;

    for (float i = 0.0; i < 1.0; i += 1.0 / 4.0) {
        float z = fract(i + t);
        float size = mix(10.0, 0.5, z);
        float fade = smoothstep(0.0, 0.5, z) * smoothstep(1.0, 0.8, z);
        m += layer(uv * size + i * 20.0, time) * fade;
    }

    float3 base = sin(t * 5.0 * float3(0.345, 0.456, 0.567)) * 0.4 + 0.6;
    float3 col = m * base;
    col -= gradient * base;

    return float4(col, 1.0);
}

// MARK: - Day36

float3x3 fromEuler(float3 ang) {
    float2 a1 = float2(sin(ang.x),cos(ang.x));
    float2 a2 = float2(sin(ang.y),cos(ang.y));
    float2 a3 = float2(sin(ang.z),cos(ang.z));
    float3x3 m;
    m[0] = float3(a1.y * a3.y + a1.x * a2.x * a3.x, a1.y * a2.x * a3.x + a3.y * a1.x, -a2.y * a3.x);
    m[1] = float3(-a2.y * a1.x, a1.y * a2.y, a2.x);
    m[2] = float3(a3.y * a1.x * a2.x + a1.y * a3.x, a1.x * a3.x - a1.y * a3.y * a2.x, a2.y * a3.y);
    return m;
}

float hash(float2 p) {
    float h = dot(p, float2(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

float noise2(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return -1.0 + 2.0 * mix(mix(hash(i + float2(0.0,0.0)),
                                hash(i + float2(1.0,0.0)), u.x),
                            mix(hash(i + float2(0.0,1.0)),
                                hash(i + float2(1.0,1.0)), u.x), u.y);
}

// lighting
float diffuse(float3 n, float3 l, float p) {
    return pow(dot(n, l) * 0.4 + 0.6, p);
}

float specular(float3 n, float3 l, float3 e, float s) {
    float nrm = (s + 8.0) / (M_PI_F * 8.0);
    return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
}

// sky
float3 getSkyColor(float3 e) {
    e.y = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;
    return float3(pow(1.0 - e.y, 2.0), 1.0 - e.y, 0.6 + (1.0 - e.y) * 0.4) * 1.1;
}

// sea
float sea_octave(float2 uv, float choppy) {
    uv += noise2(uv);
    float2 wv = 1.0 - abs(sin(uv));
    float2 swv = abs(cos(uv));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

float map(float3 p, float time) {
    float freq = 0.16;
    float amp = 0.6;
    float choppy = 4.0;
    float2 uv = p.xz;
    uv.x *= 0.75;

    float d, h = 0.0;
    for (int i = 0; i < 3; i++) {
        d = sea_octave((uv + (1.0 + time * 0.8)) * freq, choppy);
        d += sea_octave((uv - (1.0 + time * 0.8)) * freq, choppy);
        h += d * amp;
        uv *= float2x2(1.6, 1.2, -1.2, 1.6);
        freq *= 1.9;
        amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
}

float map_detailed(float3 p, float time) {
    float freq = 0.16;
    float amp = 0.6;
    float choppy = 4.0;
    float2 uv = p.xz;
    uv.x *= 0.75;

    float d, h = 0.0;
    for (int i = 0; i < 5; i++) {
        d = sea_octave((uv + (1.0 + time * 0.8)) * freq, choppy);
        d += sea_octave((uv - (1.0 + time * 0.8)) * freq, choppy);
        h += d * amp;
        uv *= float2x2(1.6, 1.2, -1.2, 1.6);
        freq *= 1.9;
        amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
}

float3 getSeaColor(float3 p, float3 n, float3 l, float3 eye, float3 dist) {
    float fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
    fresnel = pow(fresnel, 3.0) * 0.5;
    float3 reflected = getSkyColor(reflect(eye,n));
    float3 refracted = float3(0.0, 0.09, 0.18) + diffuse(n, l, 80.0) * float3(0.8, 0.9, 0.6) * 0.6 * 0.12;
    float3 color = mix(refracted,reflected,fresnel);
    float atten = max(1.0 - dot(dist,dist) * 0.001, 0.0);
    color += float3(0.8, 0.9, 0.6) * 0.6 * (p.y - 0.6) * 0.18 * atten;
    color += float3(specular(n, l, eye, 60.0));
    return color;
}

// tracing
float3 getNormal(float3 p, float eps, float time) {
    float3 n;
    n.y = map_detailed(p, time);
    n.x = map_detailed(float3(p.x + eps, p.y, p.z), time) - n.y;
    n.z = map_detailed(float3(p.x, p.y, p.z + eps), time) - n.y;
    n.y = eps;
    return normalize(n);
}

float3 heightMapTracing(float3 ori, float3 dir, float3 p, float time) {
    float tm = 0.0;
    float tx = 1000.0;
    float hx = map(ori + dir * tx, time);
    if (hx > 0.0) return tx;
    float hm = map(ori + dir * tm, time);
    float tmid = 0.0;
    for (int i = 0; i < 6; i++) {
        tmid = mix(tm, tx, hm / (hm - hx));
        p = ori + dir * tmid;
        float hmid = map(p, time);
        if (hmid < 0.0) {
            tx = tmid;
            hx = hmid;
        } else {
            tm = tmid;
            hm = hmid;
        }
    }
    return p;
}

float3 getPixel(float2 uv, float time, float2 res) {
    // ray
    float3 ang = float3(sin(time * 3.0) * 0.1, sin(time) * 0.2 + 0.3, time);
    float3 ori = float3(0.0, 3.5, time * 5.0);
    float3 dir = normalize(float3(uv.xy, -2.0)); dir.z += length(uv) * 0.14;
    dir = normalize(dir) * fromEuler(ang);

    // tracing
    float3 p = 0.0;
    p = heightMapTracing(ori, dir, p, time);
    float3 dist = p - ori;
    float3 n = getNormal(p, dot(dist,dist) * (0.1 / res.x), time);
    float3 light = normalize(float3(0.0, 1.0, 0.8));

    // color
    return mix(
               getSkyColor(dir),
               getSeaColor(p, n, light, dir, dist),
               pow(smoothstep(0.0, -0.02, dir.y), 0.2));
}

fragment float4 shader_day36(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = (pixPos.xy * 2.0 - res.xy) / min(res.x, res.y);
    uv.y *= -1.0;

    float3 color = getPixel(uv, time * 0.3, res);
    return float4(pow(color, float3(0.65)), 1.0);
}

// MARK: - Day37

fragment float4 shader_day37(float4 pixPos [[position]],
                             constant float2& res[[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 p = (pixPos.xy * 2.0 - res.xy) / min(res.x, res.y);
    p.y *= -1.0;

    p += float2(sin(time), -cos(time)) * 0.5;
    float l = 0.0;
    for (float i = 0.0; i < 8.0; i++) {
        float j = i + 1.0;
        float2 q = p + float2(cos(time * j * 0.25), sin(time / j)) * 0.5;
        float u = abs(sin((atan2(q.y, q.x) - length(q) + time) * 12.0) * 0.5) + 0.75;
        l += 0.01 / abs(u - length(q));
    }
    return float4(float3(l), 1.0);
}

// MARK: - Day39

fragment float4 shader_day39(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float3x3 sobelX = float3x3(-1.0, -2.0, -1.0,
                               0.0,  0.0, 0.0,
                               1.0,  2.0,  1.0);
    float3x3 sobelY = float3x3(-1.0,  0.0,  1.0,
                               -2.0,  0.0, 2.0,
                               -1.0,  0.0,  1.0);

    float sumX = 0.0;
    float sumY = 0.0;

    for (int i =  -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            float x = (pixPos.x + float(i)) / res.x;
            float y = (pixPos.y + float(j)) / res.y;
            sumX += length(texture.sample(s, float2(x, y)).xyz) * float(sobelX[1+i][1+j]);
            sumY += length(texture.sample(s, float2(x, y)).xyz) * float(sobelY[1+i][1+j]);
        }
    }

    float3 col = abs(sumX) + abs(sumY) > 1.0 ? 1.0 : 0.0;
    return float4(col, 1.0);
}

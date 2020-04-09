#include <metal_stdlib>
#include "Common.h"

using namespace metal;

// MARK: - Mesh

// https://www.shadertoy.com/view/XdcGzM Mesh

fragment float4 shader_dayME(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float4 o = 0.0;
    float2 uv = pixPos.xy / 8.0;
    float2 p = floor(uv + 0.5);

#define T(x,y) texture.sample(s, 8.0 * float2(x, y) / res.xy).g
#define M(c,T) o += pow(0.5 + 0.5 * cos(6.28 * (uv - p).c + 4.0 * (2.0 * T - 1.0)), 6.0)

    M(y, T(uv.x, p.y));
    M(x, T(p.x, uv.y));

    //float x = clamp(sin(time), 0.5, 1.0);
    //return float4(x * o.x, x * o.y, o.z, 1.0);
    return o;
}

// MARK: - Edges

fragment float4 shader_dayAA(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 uv = pixPos.xy / res.xy;
    float4 color = texture.sample(s, uv);
    float gray = length(color.rgb);
    //return float4(float3(step(0.06, length(float2(dfdx(gray), dfdy(gray))))), 1.0);
    return float4(float3(pow(length(float2(dfdx(gray), dfdy(gray))), 0.5)), 1.0);
}

// MARK: - WIP done

// https://www.shadertoy.com/view/lscBRf rainbow rain

#define FALLING_SPEED  0.25
#define STRIPES_FACTOR 5.0

float sphere(float2 coord, float2 pos, float r) {
    float2 d = pos - coord;
    return smoothstep(60.0, 0.0, dot(d, d) - r * r);
}

fragment float4 shader_dayRA(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]]) {

    float2 uv = pixPos.xy / res.xy;
    uv.y *= -1.0;

    float2 clamped_uv = (round(pixPos.xy / STRIPES_FACTOR) * STRIPES_FACTOR) / res.xy;
    float value = fract(sin(clamped_uv.x) * 43758.5453123);

    float3 col = float3(1.0 - mod(uv.y * 0.5 + (time * (FALLING_SPEED + value / 5.0)) + value, 0.5));
    //float3 col = float3(mod(uv.y * 0.5 + (time * (FALLING_SPEED + value / 5.0)) + value, 0.5));
    col *= clamp(cos(time * 2.0 + uv.xyx + float3(0.0, 2.0, 4.0)), 0.0, 1.0);
    col += float3(sphere(pixPos.xy,
                         float2(clamped_uv.x, (1.0 - 2.0 * mod((time * (FALLING_SPEED + value / 5.0)) + value, 0.5))) * res.xy,
                         0.9)) / 2.0;
//    col += float3(sphere(pixPos.xy,
//                         float2(clamped_uv.x, (2.0 * mod((time * (FALLING_SPEED + value / 5.0)) + value, 0.5))) * res.xy,
//                         0.9)) / 2.0;
//    col *= float3(exp(-pow(abs(uv.y - 0.5), 6.0) / pow(2.0 * 0.05, 2.0)));
    col *= float3(exp(-pow(abs(uv.y + 0.5), 6.0) / pow(2.0 * 0.05, 2.0)));
    return float4(col, 1.0);
}

// MARK: - Color wash machine

// http://glslsandbox.com/e#50791.0 Color wash machine

fragment float4 shader_day77_(float4 pixPos [[position]],
                              constant float2& res [[buffer(0)]],
                              constant float& time[[buffer(1)]]) {

    float3 p = float3(pixPos.xy / res.y, sin(time));
    for (int i = 0; i < 40; i++) {
        p.xzy = float3(1.3, 0.999, 0.7) * (abs((abs(p) / dot(p, p) - float3(1.0, 1.0, cos(time) * 0.5))));
    }
    return float4(p, 1.0);
}

// https://www.shadertoy.com/view/WdsyRM Wave in a box
// https://www.shadertoy.com/view/4dfGzs Voxel Edges

// MARK: - xxx

// https://www.shadertoy.com/view/XlsXDN Money

//void mainImage( out float4 fragColor, in float2 fragCoord )
//{
//    float2 xy = fragCoord.xy / iResolution.yy;
//
//    float amplitud = 0.03;
//    float frecuencia = 10.0;
//    float gris = 1.0;
//    float divisor = 4.8 / iResolution.y;
//    float grosorInicial = divisor * 0.2;
//
//    const int kNumPatrones = 6;
//
//    // x: seno del angulo, y: coseno del angulo, z: factor de suavizado
//    float3 datosPatron[kNumPatrones];
//    datosPatron[0] = float3(-0.7071, 0.7071, 3.0); // -45
//    datosPatron[1] = float3(0.0, 1.0, 0.6); // 0
//    datosPatron[2] = float3(0.0, 1.0, 0.5); // 0
//    datosPatron[3] = float3(1.0, 0.0, 0.4); // 90
//    datosPatron[4] = float3(1.0, 0.0, 0.3); // 90
//    datosPatron[5] = float3(0.0, 1.0, 0.2); // 0
//
//    float4 color = texture(iChannel0, float2(fragCoord.x / iResolution.x, xy.y));
//    fragColor = color;
//
//    for(int i = 0; i < kNumPatrones; i++)
//    {
//        float coseno = datosPatron[i].x;
//        float seno = datosPatron[i].y;
//
//        // RotaciÃ³n del patrÃ³n
//        float2 punto = float2(
//                          xy.x * coseno - xy.y * seno,
//                          xy.x * seno + xy.y * coseno
//                          );
//
//        float grosor = grosorInicial * float(i + 1);
//        float dist = mod(punto.y + grosor * 0.5 - sin(punto.x * frecuencia) * amplitud, divisor);
//        float brillo = 0.3 * color.r + 0.4 * color.g + 0.3 * color.b;
//
//        if(dist < grosor && brillo < 0.75 - 0.12 * float(i))
//        {
//            // Suavizado
//            float k = datosPatron[i].z;
//            float x = (grosor - dist) / grosor;
//            float fx = abs((x - 0.5) / k) - (0.5 - k) / k;
//            gris = min(fx, gris);
//        }
//    }
//
//
//    float mx = iMouse.x;
//    if(iMouse.z < 1.0) mx = iResolution.x * 0.5;
//    if(fragCoord.x < mx) fragColor = float4(gris, gris, gris, 1.0);
//}

// MARK: - Day88

// https://www.shadertoy.com/view/ttdXD8 Glitch

float rand88(float n) { return fract(sin(n) * 43758.5453123); }

float noise(float p) {
    float fl = floor(p);
    float fc = fract(p);
    return mix(rand88(fl), rand88(fl + 1.0), fc);
}

float blockyNoise(float2 uv, float threshold, float scale, float seed, sampler s, texture2d<float, access::sample> noiseTexture, float time) {
    float scroll = floor(time + sin(11.0 * time) + sin(time)) * 0.77;
    float2 noiseUV = (uv.yy / scale + scroll) / 64.0;
    float noise2 = noiseTexture.sample(s, noiseUV).r;
    float id = floor(noise2 * 20.0);
    id = noise(id + seed) - 0.5;
    if (abs(id) > threshold)
        id = 0.0;
    return id;
}

fragment float4 shader_day88_(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]],
                             texture2d<float, access::sample> noiseTexture [[texture(2)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float rgbIntesnsity = 0.1 + 0.1 * sin(time * 3.7);
    float displaceIntesnsity = 0.2 +  0.3 * pow(sin(time * 1.2), 5.0);
    float interlaceIntesnsity = 0.01;
    float dropoutIntensity = 0.1;

    float2 uv = pixPos.xy / res.xy;

    float displace = blockyNoise(uv + float2(uv.y, 0.0), displaceIntesnsity, 25.0, 66.6, s, noiseTexture, time);
    displace *= blockyNoise(uv.yx + float2(0.0, uv.x), displaceIntesnsity, 111.0, 13.7, s, noiseTexture, time);

    uv.x += displace;

    float2 offs = 0.1 * float2(blockyNoise(uv.xy + float2(uv.y, 0.0), rgbIntesnsity, 65.0, 341.0, s, noiseTexture, time), 0.0);

    float colr = texture.sample(s, uv - offs).r;
    float colg = texture.sample(s, uv).g;
    float colb = texture.sample(s ,uv + offs).b;

    float line = fract(pixPos.y / 3.0);
    float3 mask = float3(3.0, 0.0, 0.0);
    if (line > 0.333)
        mask = float3(0.0, 3.0, 0.0);
    if (line > 0.666)
        mask = float3(0.0, 0.0, 3.0);

    float maskNoise = blockyNoise(uv, interlaceIntesnsity, 90.0, time, s, noiseTexture, time) * max(displace, offs.x);

    maskNoise = 1.0 - maskNoise;
    if ( maskNoise == 1.0)
        mask = float3(1.0);

    float dropout = blockyNoise(uv, dropoutIntensity, 11.0, time, s, noiseTexture, time) * blockyNoise(uv.yx, dropoutIntensity, 90.0, time, s, noiseTexture, time);
    mask *= (1.0 - 5.0 * dropout);

    return float4(mask * float3(colr, colg, colb), 1.0);
}

// MARK: - Day88...

// https://www.shadertoy.com/view/WsSXRh

float2x2 rot2d(float angle) { return float2x2(cos(angle), -sin(angle), sin(angle), cos(angle)); }
float r(float a, float b) { return fract(sin(dot(float2(a, b), float2(12.9898, 78.233))) * 43758.5453); }

float h(float a) {
    return fract(sin(dot(a, float2(12.9898, 78.233))) * 43758.5453);
}

float noise88(float3 x) {
    float3 p = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    return mix(mix(mix( h(n+0.0), h(n+1.0),f.x),
                   mix( h(n+57.0), h(n+58.0),f.x),f.y),
               mix(mix( h(n+113.0), h(n+114.0),f.x),
                   mix( h(n+170.0), h(n+171.0),f.x),f.y),f.z);
}

float3 dnoise2f(float2 p) {
    float i = floor(p.x), j = floor(p.y);
    float u = p.x-i, v = p.y-j;
    float du = 30.*u*u*(u*(u-2.)+1.);
    float dv = 30.*v*v*(v*(v-2.)+1.);
    u=u*u*u*(u*(u*6.-15.)+10.);
    v=v*v*v*(v*(v*6.-15.)+10.);
    float a = r(i,     j    );
    float b = r(i+1.0, j    );
    float c = r(i,     j+1.0);
    float d = r(i+1.0, j+1.0);
    float k0 = a;
    float k1 = b-a;
    float k2 = c-a;
    float k3 = a-b-c+d;
    return float3(k0 + k1*u + k2*v + k3*u*v,
                  du*(k1 + k3*v),
                  dv*(k2 + k3*u));
}

float fbm(float2 uv, float time) {
    float2 p = uv;
    float f, dx, dz, w = 0.5;
    f = dx = dz = 0.0;
    for (int i = 0; i < 28; ++i) {
        float3 n = dnoise2f(uv);
        dx += n.y;
        dz += n.z;
        f += w * n.x / (1.0 + dx * dx + dz * dz);
        w *= 0.86;
        uv *= float2(1.16);
        uv *= rot2d(1.25 * noise88(float3(p * 0.1, 0.12 * time)) +
                    0.75 * noise88(float3(p * 0.1, 0.20 * time)));
    }
    return f;
}

float fbmLow(float2 uv) {
    float f, dx, dz, w = 0.5;
    f = dx = dz = 0.0;
    for (int i = 0; i < 4; ++i) {
        float3 n = dnoise2f(uv);
        dx += n.y;
        dz += n.z;
        f += w * n.x / (1.0 + dx * dx + dz * dz);
        w *= 0.75;
        uv *= float2(1.5);
    }
    return f;
}

fragment float4 shader_day88__(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float ripple = 0.5;

    float2 uv = 1.0 - 2.0 * (pixPos.xy / res.xy);
    uv.y /= res.x / res.y;
    float t = time * 0.6;

    float2 rv = uv / (length(uv * 2.5) * (uv * 30.0));
    uv *= rot2d(0.3 * t);
    float val = 0.5 * fbm(uv * 2.0 * fbmLow(length(uv) + rv - t), t);
    uv *= rot2d(-0.6 * t);

    return texture.sample(s, (pixPos.xy / res.xy) * 1.0 - fbm(uv * val * 20.0, t) * ripple);
}

// MARK: - WIP done

// https://www.shadertoy.com/view/MdsGzM Edge glow

float2 getUV(float2 pixPos, float2 res) {
    return float2(pixPos.x / res.x, pixPos.y / res.y);
}

bool isEdgeFragment(float2 pixPos, float2 res, sampler s, texture2d<float, access::sample> texture) {
    float kernelX[9];
    kernelX[0] = -1.;
    kernelX[1] = -1.;
    kernelX[2] = -1.;
    kernelX[3] = -1.;
    kernelX[4] =  8.;
    kernelX[5] = -1.;
    kernelX[6] = -1.;
    kernelX[7] = -1.;
    kernelX[8] = -1.;

    float4 result = float4(0.);
    float2 uv = getUV(pixPos, res);

    for (float y = 0.; y < 3.0; ++y) {
        for (float x = 0.; x < 3.0; ++x) {
            result += texture.sample(s, float2(uv.x + (float(int(x - 3.0 / 2.0)) / res.x),
                                               uv.y + (float(int(y - 3.0 / 2.0)) / res.y)))
            * kernelX[int(x + (y * 3.0))];
        }
    }

    return ((length(result) > 0.2) ? true : false);
}

fragment float4 shader_dayAW(float4 pixPos [[position]],
                             constant float2& res [[buffer(0)]],
                             constant float& time[[buffer(1)]],
                             texture2d<float, access::sample> texture [[texture(1)]]) {

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float4 actualColor = texture.sample(s, getUV(pixPos.xy, res));
    if (!isEdgeFragment(pixPos.xy, res,s, texture)) {
        return actualColor;
    } else {
        return float4(0.0, 1.0, 0.0, 1.0) * sin(time * 5.0) + actualColor * cos(time * 5.0);
    }
}

// MARK: - WIP not done
//
//// https://www.shadertoy.com/view/4dfGzs maze
//
//float noise(sampler s, texture2d<float, access::sample> texture, float3 x) {
//    float3 p = floor(x);
//    float3 f = fract(x);
//    f = f * f * (3.0 - 2.0 * f);
//    float2 uv = (p.xy + float2(37.0, 17.0) * p.z) + f.xy;
//    float2 rg = texture.sample(s, (uv + 0.5) / 256.0, 0.0).yx;
//    return mix(rg.x, rg.y, f.z);
//}
//
//float mapTerrain(sampler s, texture2d<float, access::sample> texture, float time, float3 p) {
//    p *= 0.1;
//    p.xz *= 0.6;
//
//    float timeX = 0.5 + 0.15 * time;
//    float ft = fract( timeX );
//    float it = floor( timeX );
//    ft = smoothstep( 0.7, 1.0, ft );
//    timeX = it + ft;
//    float spe = 1.4;
//
//    float f;
//    f  = 0.5000 * noise(s, texture, p * 1.00 + float3(0.0, 1.0, 0.0) * spe * timeX);
//    f += 0.2500 * noise(s, texture, p * 2.02 + float3(0.0, 2.0, 0.0) * spe * timeX);
//    f += 0.1250 * noise(s, texture, p * 4.01);
//    return 25.0 * f - 10.0;
//}
//
//float map(float3 gro, sampler s, texture2d<float, access::sample> texture, float time, float3 c) {
//    float3 p = c + 0.5;
//    float f = mapTerrain(s, texture, time, p) + 0.25 * p.y;
//    f = mix(f, 1.0, step(length(gro - p), 5.0));
//    return step(f, 0.5);
//}
//
//struct CastRayResult {
//    float value;
//    float3 oVos;
//    float3 oDir;
//};
//
//CastRayResult castRay(float3 ro, float3 rd, float3 oVos, float3 oDir, float3 gro, sampler s, texture2d<float, access::sample> texture, float time) {
//    float3 pos = floor(ro);
//    float3 ri = 1.0 / rd;
//    float3 rs = sign(rd);
//    float3 dis = (pos-ro + 0.5 + rs * 0.5) * ri;
//
//    float res = -1.0;
//    float3 mm = float3(0.0);
//    for (int i = 0; i < 128; i++) {
//        if (map(gro, s, texture, time, pos) > 0.5) {
//            res = 1.0;
//            break;
//        }
//        mm = step(dis.xyz, dis.yzx) * step(dis.xyz, dis.zxy);
//        dis += mm * rs * ri;
//        pos += mm * rs;
//    }
//
//    //float3 nor = -mm * rs;
//    float3 vos = pos;
//
//    // intersect the cube
//    float3 mini = (pos - ro + 0.5 - 0.5 * float3(rs)) * ri;
//    float t = max(mini.x, max(mini.y, mini.z));
//
//    oDir = mm;
//    oVos = vos;
//
//    CastRayResult ret;
//    ret.value = t * res;
//    ret.oVos = oVos;
//    ret.oDir = oDir;
//    return ret;
//}
//
//float3 path(float t, float ya) {
//    float2 p = 100.0 * sin( 0.02 * t * float2(1.0, 1.2) + float2(0.1, 0.9));
//    p += 50.0 * sin(0.04 * t * float2(1.3, 1.0) + float2(1.0, 4.5));
//    return float3(p.x, 18.0 + ya * 4.0 * sin(0.05 * t), p.y);
//}
//
//float3x3 setCamera96(float3 ro, float3 ta, float cr) {
//    float3 cw = normalize(ta - ro);
//    float3 cp = float3(sin(cr), cos(cr), 0.0);
//    float3 cu = normalize(cross(cw, cp));
//    float3 cv = normalize(cross(cu, cw));
//    return float3x3(cu, cv, -cw);
//}
//
//float maxcomp(float4 v ) {
//    return max(max(v.x, v.y), max(v.z, v.w));
//}
//
//float isEdge(float2 uv, float4 va, float4 vb, float4 vc, float4 vd ) {
//    float2 st = 1.0 - uv;
//
//    float4 wb = smoothstep(0.85, 0.99, float4(uv.x,
//                                              st.x,
//                                              uv.y,
//                                              st.y)) * ( 1.0 - va + va * vc);
//    float4 wc = smoothstep(0.85, 0.99, float4(uv.x * uv.y,
//                                              st.x * uv.y,
//                                              st.x * st.y,
//                                              uv.x * st.y)) * ( 1.0 - vb + vd * vb);
//    return maxcomp(max(wb, wc));
//}
//
//float calcOcc(float2 uv, float4 va, float4 vb, float4 vc, float4 vd) {
//    float2 st = 1.0 - uv;
//    float4 wa = float4(uv.x, st.x, uv.y, st.y) * vc;
//    float4 wb = float4(uv.x * uv.y,
//                       st.x * uv.y,
//                       st.x * st.y,
//                       uv.x * st.y) * vd * (1.0 - vc.xzyw) * (1.0 - vc.zywx);
//    return wa.x + wa.y + wa.z + wa.w + wb.x + wb.y + wb.z + wb.w;
//}
//
//float3 render(float3 ro, float3 rd, float3 gro, sampler s, texture2d<float, access::sample> texture, float time) {
//    float3 col = float3(0.0);
//
//    // raymarch
//    float3 vos = float3(0.0);
//    float3 dir = float3(0.0);
//    CastRayResult ret = castRay(ro, rd, vos, dir, gro, s, texture, time);
//    float t = ret.value;
//    vos = ret.oVos;
//    dir = ret.oDir;
//    if (t > 0.0) {
//        float3 nor = -dir * sign(rd);
//        float3 pos = ro + rd * t;
//        float3 uvw = pos - vos;
//
//        float3 v1  = vos + nor + dir.yzx;
//        float3 v2  = vos + nor - dir.yzx;
//        float3 v3  = vos + nor + dir.zxy;
//        float3 v4  = vos + nor - dir.zxy;
//        float3 v5  = vos + nor + dir.yzx + dir.zxy;
//        float3 v6  = vos + nor - dir.yzx + dir.zxy;
//        float3 v7  = vos + nor - dir.yzx - dir.zxy;
//        float3 v8  = vos + nor + dir.yzx - dir.zxy;
//        float3 v9  = vos + dir.yzx;
//        float3 v10 = vos - dir.yzx;
//        float3 v11 = vos + dir.zxy;
//        float3 v12 = vos - dir.zxy;
//        float3 v13 = vos + dir.yzx + dir.zxy;
//        float3 v14 = vos - dir.yzx + dir.zxy ;
//        float3 v15 = vos - dir.yzx - dir.zxy;
//        float3 v16 = vos + dir.yzx - dir.zxy;
//
//        float4 vc = float4(map(gro, s, texture, time, v1),  map(gro, s, texture, time, v2),  map(gro, s, texture, time, v3),  map(gro, s, texture, time, v4) );
//        float4 vd = float4(map(gro, s, texture, time, v5),  map(gro, s, texture, time, v6),  map(gro, s, texture, time, v7),  map(gro, s, texture, time, v8) );
//        float4 va = float4(map(gro, s, texture, time, v9),  map(gro, s, texture, time, v10), map(gro, s, texture, time, v11), map(gro, s, texture, time, v12));
//        float4 vb = float4(map(gro, s, texture, time, v13), map(gro, s, texture, time, v14), map(gro, s, texture, time, v15), map(gro, s, texture, time, v16));
//
//        float2 uv = float2(dot(dir.yzx, uvw), dot(dir.zxy, uvw));
//
//        // wireframe
//        float www = 1.0 - isEdge( uv, va, vb, vc, vd );
//
//        float3 wir = smoothstep( 0.4, 0.5, abs(uvw-0.5) );
//        float vvv = (1.0-wir.x*wir.y)*(1.0-wir.x*wir.z)*(1.0-wir.y*wir.z);
//
//        col = 2.0 * texture.sample(s, 0.01 * pos.xz).zyx;
//        col += 0.8 * float3(0.1, 0.3, 0.4);
//        col *= 0.5;
//        col *= 1.0 - 0.75*(1.0-vvv)*www;
//
//        float3 lig = normalize(float3(-0.4, 0.3, 0.7)); // mark
//
//        // lighting
//        float dif = clamp(dot(nor, lig), 0.0, 1.0);
//        float bac = clamp(dot(nor, normalize(lig * float3(-1.0, 0.0, -1.0))), 0.0, 1.0 );
//        float sky = 0.5 + 0.5 * nor.y;
//        float amb = clamp(0.75 + pos.y / 25.0, 0.0, 1.0);
//        float occ = 1.0;
//
//        // ambient occlusion
//        occ = calcOcc( uv, va, vb, vc, vd );
//        occ = 1.0 - occ/8.0;
//        occ = occ*occ;
//        occ = occ*occ;
//        occ *= amb;
//
//        // lighting
//        float3 lin = float3(0.0);
//        lin += 2.5*dif*float3(1.00,0.90,0.70)*(0.5+0.5*occ);
//        lin += 0.5*bac*float3(0.15,0.10,0.10)*occ;
//        lin += 2.0*sky*float3(0.40,0.30,0.15)*occ;
//
//        // line glow
//        float lineglow = 0.0;
//        lineglow += smoothstep( 0.4, 1.0,     uv.x )*(1.0-va.x*(1.0-vc.x));
//        lineglow += smoothstep( 0.4, 1.0, 1.0-uv.x )*(1.0-va.y*(1.0-vc.y));
//        lineglow += smoothstep( 0.4, 1.0,     uv.y )*(1.0-va.z*(1.0-vc.z));
//        lineglow += smoothstep( 0.4, 1.0, 1.0-uv.y )*(1.0-va.w*(1.0-vc.w));
//        lineglow += smoothstep( 0.4, 1.0,      uv.y*      uv.x )*(1.0-vb.x*(1.0-vd.x));
//        lineglow += smoothstep( 0.4, 1.0,      uv.y* (1.0-uv.x))*(1.0-vb.y*(1.0-vd.y));
//        lineglow += smoothstep( 0.4, 1.0, (1.0-uv.y)*(1.0-uv.x))*(1.0-vb.z*(1.0-vd.z));
//        lineglow += smoothstep( 0.4, 1.0, (1.0-uv.y)*     uv.x )*(1.0-vb.w*(1.0-vd.w));
//
//        float3 linCol = 2.0*float3(5.0,0.6,0.0);
//        linCol *= (0.5+0.5*occ)*0.5;
//        lin += 3.0*lineglow*linCol;
//
//        col = col * lin;
//        col += 8.0 * linCol * float3(1.0, 2.0, 3.0) * (1.0 - www);
//        col += 0.1 * lineglow * linCol;
//        col *= min(0.1, exp(-0.07 * t));
//
//        // blend to black & white
//        float3 col2 = float3(1.3) * (0.5 + 0.5 * nor.y) * occ * www * (0.9 + 0.1 * vvv) * exp(-0.04 * t);
//        float mi = sin(-1.57 + 0.5 * time);
//        mi = smoothstep(0.70, 0.75, mi);
//        col = mix(col, col2, mi);
//    }
//
//    // gamma
//    col = pow(col, float3(0.45));
//
//    return col;
//}
//
//fragment float4 shader_dayAJ(float4 pixPos [[position]],
//                             constant float2& res [[buffer(0)]],
//                             constant float& time[[buffer(1)]],
//                             texture2d<float, access::sample> texture [[texture(4)]]) {
//
//    constexpr sampler s(address::repeat, filter::linear);
//
//    float2 q = pixPos.xy / res.xy;
//    float2 p = -1.0 + 2.0 * q;
//    p.x *= res.x/ res.y;
//
//    q.y *= -1.0;
//    p.y *= -1.0;
//
//    float timeX = 2.0 * time;
//
//    float cr = 0.0;//0.2 * cos(0.1 * timeX);
//    float3 ro = path(timeX + 0.0, 1.0);
//    float3 ta = path(timeX + 5.0, 1.0) - float3(0.0, 6.0, 0.0);
//
//    //float3 gro = ro;
//    float3 gro = float3(1.0);
//
//    float3x3 cam = setCamera96(ro, ta, cr);
//
//    float r2 = p.x * p.x * 0.32 + p.y * p.y;
//    p *= (7.0 - sqrt(37.5 - 11.5 * r2)) / (r2 + 1.0);
//    float3 rd = normalize(cam * float3(p.xy, -2.5));
//
//    float3 col = render(ro, rd, gro, s, texture, timeX);
//
//    //col *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.1);
//
//    return float4(col, 1.0);
//}

// MARK: - WIP

// https://www.shadertoy.com/view/WtSGDz Sakura

float2x2 rotate2d(float _angle) {
    return float2x2(cos(_angle), -sin(_angle),
                    sin(_angle), cos(_angle));
}

float circle(float2 uv, float radius, float blur) {
    float d = length(uv);
    return smoothstep(radius, radius - blur, d);
}

fragment float4 shader_day100a(float4 pixPos [[position]],
                               constant float2& res [[buffer(0)]],
                               constant float& time[[buffer(1)]]) {

    float2 uv = pixPos.xy / res;
    uv -= float2(0.5, 1.1);
    //    uv.y *= -1.0;

    float s = 0.01;
    float3 coll = float3(0.0);
    float t = time / 4.0;
    float vl = 0.0;
    float r = 0.05;
    for (float f = 0.0; f < 1.0; f += s) {
        float2 st = uv;

        st.x += fract((sin(f * 1245.0)) * 114.0) - 0.5;
        st.y += fract(t * sin(f + 0.1) + f * 2.0) * 1.2;
        //st.y -= fract(t * sin(f + 0.1) + f * 2.0) * 1.2;

        st *= mix(f, 0.9, 2.0);

        st.x *= res.x / res.y;
        st = rotate2d(time + sin(f * 175.0) * 1854.0) * st;
        st.y *= 1.82;
        st.y -= abs(st.x / 3.0 + sin(time + fract(f)) * 0.01);
        //st.y += abs(st.x / 3.0 + sin(time + fract(f)) * 0.01);
        vl = max(circle(st, r, 0.027), vl);

        coll = vl * float3(1.0, 0.5, 0.7);
    }

    return max(float4(coll, 1.0), float4(0.0));
}


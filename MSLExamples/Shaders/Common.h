//
//  Common.h
//  ShaderArtSamples
//
//  Created by Youichi Takatsu on 2019/08/25.
//  Copyright © 2019 TakatsuYouichi. All rights reserved.
//

#ifndef Common_h
#define Common_h

// Noise
float N11(float s);
float N21(float2 p);
float3 hsv2rgb(float h, float s, float v);

// Gereral Mod
float mod(float a, float b);

// Grid
float grid(float2 p);

// Rotation
metal::float2x2 rot(float radian);

float rand(float2 n);
float3 mod(float3 x, float3 y);
float deg2rad(float num);

#endif /* Common_h */

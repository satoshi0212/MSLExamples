//
//  GPUDevice.swift
//  ShaderArtSamples
//
//  Created by Youichi Takatsu on 2019/08/04.
//  Copyright © 2019 TakatsuYouichi. All rights reserved.
//

import Foundation
import Metal
import simd

typealias Acceleration = SIMD3<Float>

class GPUDevice {
    static let shared = GPUDevice()
    
    let device = MTLCreateSystemDefaultDevice()!
    var library : MTLLibrary!
    lazy var vertexFunction : MTLFunction = library.makeFunction(name: "vertexShader")!
    lazy var fragmentFunctions : [MTLFunction] = library.functionNames.compactMap{ library.makeFunction(name: $0) }.filter{ $0.functionType == .fragment }
    
    var resolutionBuffer : MTLBuffer! = nil
    var timeBuffer : MTLBuffer! = nil
    var volumeBuffer : MTLBuffer! = nil
    var accelerationBuffer : MTLBuffer! = nil
    var touchedPositionBuffer : MTLBuffer! = nil

    private init() {
        library = device.makeDefaultLibrary()
        
        setUpBeffers()
    }
    
    func setUpBeffers() {
        resolutionBuffer = device.makeBuffer(length: 2 * MemoryLayout<Float>.size, options: [])
        timeBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: [])
        volumeBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: [])
        accelerationBuffer = device.makeBuffer(length: MemoryLayout<Acceleration>.size, options: [])
        touchedPositionBuffer = device.makeBuffer(length: 2 * MemoryLayout<Float>.size, options: [])
    }
    
    func updateResolution(width: Float, height: Float) {
        memcpy(resolutionBuffer.contents(), [width, height], MemoryLayout<Float>.size * 2)
    }
    
    func updateTime(_ time: Float) {
        updateBuffer(time, timeBuffer)
    }

    func updateVolume(_ volume: Float) {
        updateBuffer(volume, volumeBuffer)
    }
    
    func updateAcceleration(_ acceleration: Acceleration) {
        updateBuffer(acceleration, accelerationBuffer)
    }
    
    func updateTouchedPosition(x : Float, y: Float) {
        memcpy(touchedPositionBuffer.contents(), [x, y], MemoryLayout<Float>.size * 2)
    }
    
    func render() {
        
    }
    
    private func updateBuffer<T>(_ data:T, _ buffer: MTLBuffer) {
        let pointer = buffer.contents()
        let value = pointer.bindMemory(to: T.self, capacity: 1)
        value[0] = data
    }
}

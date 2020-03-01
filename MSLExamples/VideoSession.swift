//
//  VideoSession
//  ShaderArtSamples
//
//  Created by Youichi Takatsu on 2019/08/04.
//  Copyright © 2019 TakatsuYouichi. All rights reserved.
//

import Foundation
import AVFoundation
import Metal

protocol SessionDelegate {
    func session(_ session: VideoSession, didReceiveVideoTexture texture: MTLTexture)
    func session(_ session: VideoSession, didReceiveAudioVolume value: Float)
}

class VideoSession : NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {

    private var session = AVCaptureSession()
    private var videoDevice: AVCaptureDevice!
    private var audioDevice: AVCaptureDevice!
    
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let audioDataOutput = AVCaptureAudioDataOutput()
    
    private var videoQueue = DispatchQueue(label: "VideoSession")
    private var audioQueue = DispatchQueue(label: "AudioSession")

    private var viewSize: CGSize?

    var textureCache: CVMetalTextureCache?

    public var delegate: SessionDelegate?
    
    public var videoOrientation: AVCaptureVideoOrientation {
        didSet {
            guard let connection = videoDataOutput.connection(with: .video) else { return }
            connection.videoOrientation = videoOrientation
        }
    }
    
    public static let shared : VideoSession = VideoSession()
    
    private override init() {
        videoOrientation = .landscapeRight
        super.init()
    }
    
    private func initializeCache() {
        let result = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, GPUDevice.shared.device, nil, &textureCache)
        if result != kCVReturnSuccess {
            fatalError()
        }
    }
    
    // MARK: - Session Start/End

    public func startSession(viewSize: CGSize) {

        self.viewSize = viewSize

        self.session.beginConfiguration()
        
        // Input
        self.videoDevice = AVCaptureDevice.default(for: .video)
        let videoInput = try! AVCaptureDeviceInput(device: self.videoDevice)
        self.session.addInput(videoInput)

        self.audioDevice = AVCaptureDevice.default(for: .audio)
        let audioInput = try! AVCaptureDeviceInput(device: self.audioDevice)
        self.session.addInput(audioInput)

        // Output
        self.videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String : Int(kCVPixelFormatType_32BGRA)]
        self.videoDataOutput.alwaysDiscardsLateVideoFrames = true
        self.videoDataOutput.setSampleBufferDelegate(self, queue: self.videoQueue)
        self.session.addOutput(self.videoDataOutput)
        self.videoDataOutput.connection(with: .video)?.videoOrientation = .landscapeRight

        self.audioDataOutput.setSampleBufferDelegate(self, queue: self.audioQueue)
        self.session.addOutput(self.audioDataOutput)

        self.session.commitConfiguration()
        
        // Initialize Texture Cache
        self.initializeCache()
        
        self.session.startRunning()
    }
    
    public func endSession() {
        self.session.stopRunning()
        for input in self.session.inputs {
            self.session.removeInput(input)
        }
        for output in self.session.outputs {
            self.session.removeOutput(output)
        }
    }
    
    // MARK: - AVCaptureDataOutputDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if output == videoDataOutput {
            guard let texture = createTexture(from: sampleBuffer, textureCache: textureCache) else { return }
            self.delegate?.session(self, didReceiveVideoTexture: texture)
        }
        else if output == audioDataOutput {
            guard let channel = connection.audioChannels.first else { return }
            let volume = Float(exp(channel.averagePowerLevel/20.0))
            self.delegate?.session(self, didReceiveAudioVolume: volume)
        }
    }
    
    private func createTexture(from sampleBuffer: CMSampleBuffer?, textureCache: CVMetalTextureCache?) -> MTLTexture? {
        guard
            let sampleBuffer = sampleBuffer,
            let textureCache = textureCache,
            let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
            let viewSize = viewSize
            else {
                return nil
        }
        
        let width = CVPixelBufferGetWidth(imageBuffer)
        //let height = CVPixelBufferGetHeight(imageBuffer) // Memo: 歪みを抑制するためviewSizeにより算出
        let height = Int(CGFloat(width) * viewSize.height / viewSize.width)

        var imageTexture: CVMetalTexture?
        let result = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, imageBuffer, nil, .bgra8Unorm, width, height, 0, &imageTexture)
        
        guard let unwrappedImageTexture = imageTexture, let texture = CVMetalTextureGetTexture(unwrappedImageTexture), result == kCVReturnSuccess else {
            return nil
        }
        
        return texture
    }
}

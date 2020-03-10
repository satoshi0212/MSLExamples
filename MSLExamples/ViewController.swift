import AVFoundation
import MetalKit
import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var metalView: MTKView!

    private let gpu = GPUDevice.shared
    private let scaleFactor = UIScreen.main.scale
    private let startDate = Date()
    private let semaphore = DispatchSemaphore(value: 1)

    private var commandQueue : MTLCommandQueue! = nil
    private var pipelineState : MTLRenderPipelineState! = nil
    private var volumeLevel : Float = 0.0
    private var touched = CGPoint(x: 0.0, y: 0.0)
    private var cameraTexture : MTLTexture? = nil
    private var grayNoiseSmallTexture : MTLTexture! = nil
    private var rgbaNoiseSmallTexture : MTLTexture! = nil
    private var rgbaNoiseTexture : MTLTexture! = nil
    private var rustyMetalTexture : MTLTexture! = nil

    override func viewDidLoad() {
        super.viewDidLoad()

        metalView.device = gpu.device
        metalView.delegate = self
        metalView.depthStencilPixelFormat = .invalid
        metalView.framebufferOnly = false

        commandQueue = gpu.device.makeCommandQueue()

        grayNoiseSmallTexture = loadTexture(image: UIImage(named: "grayNoiseSmall")!, rect: CGRect(x: 0, y: 0, width: 64, height: 64))
        rgbaNoiseSmallTexture = loadTexture(image: UIImage(named: "rgbaNoiseSmall")!, rect: CGRect(x: 0, y: 0, width: 64, height: 64))
        rgbaNoiseTexture = loadTexture(image: UIImage(named: "rgbaNoise")!, rect: CGRect(x: 0, y: 0, width: 256, height: 256))
        rustyMetalTexture = loadTexture(image: UIImage(named: "rustyMetal")!, rect: CGRect(x: 0, y: 0, width: 512, height: 512))

        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.vertexFunction = GPUDevice.shared.vertexFunction
        pipelineStateDescriptor.fragmentFunction = gpu.library.makeFunction(name: "shader_day70") // TODO: 文字列指定しているシェーダ名を一覧化
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        pipelineState = try! gpu.device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)

        VideoSession.shared.videoOrientation = .landscapeRight
        VideoSession.shared.delegate = self
    }

    private func loadTexture(image: UIImage, rect: CGRect) -> MTLTexture {
        let textureLoader = MTKTextureLoader(device: gpu.device)
        let imageRef = image.cgImage!.cropping(to: rect)!
        let imageData = UIImage(cgImage: imageRef).pngData()!
        return try! textureLoader.newTexture(data: imageData, options: nil)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)

        VideoSession.shared.startSession(viewSize: view.bounds.size)
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)

        VideoSession.shared.endSession()
    }

    override var prefersHomeIndicatorAutoHidden: Bool {
        return true
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return; }
        touched = touch.location(in: metalView)
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        touched = touch.location(in: metalView)
    }
}

extension ViewController : SessionDelegate {
    func session(_ session: VideoSession, didReceiveVideoTexture texture: MTLTexture) {
        cameraTexture = texture
    }

    func session(_ session: VideoSession, didReceiveAudioVolume volume: Float) {
        volumeLevel = volume
    }
}

extension ViewController : MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        gpu.updateResolution(width: Float(size.width), height: Float(size.height))
    }

    func draw(in view: MTKView) {
        _ = semaphore.wait(timeout: .distantFuture)
        guard
            let renderPassDesicriptor = metalView.currentRenderPassDescriptor,
            let currentDrawable = metalView.currentDrawable,
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let cameraTexture = cameraTexture,
            let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesicriptor) else {
                semaphore.signal()
                return
        }
        gpu.updateTime(Float(Date().timeIntervalSince(startDate)))
        gpu.updateVolume(volumeLevel)
        gpu.updateTouchedPosition(x: Float(scaleFactor * touched.x), y: Float(scaleFactor * touched.y))

        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setFragmentBuffer(gpu.resolutionBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentBuffer(gpu.timeBuffer, offset: 0, index: 1)
        renderEncoder.setFragmentBuffer(gpu.volumeBuffer, offset: 0, index: 2)
        renderEncoder.setFragmentBuffer(gpu.touchedPositionBuffer, offset: 0, index: 3)

        renderEncoder.setFragmentTexture(cameraTexture, index: 1)
        renderEncoder.setFragmentTexture(grayNoiseSmallTexture, index: 2)
        renderEncoder.setFragmentTexture(rgbaNoiseSmallTexture, index: 3)
        renderEncoder.setFragmentTexture(rgbaNoiseTexture, index: 4)
        renderEncoder.setFragmentTexture(rustyMetalTexture, index: 5)

        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.endEncoding()

        commandBuffer.addScheduledHandler { [weak self] (_) in
            guard let self = self else { return }
            self.semaphore.signal()
        }

        commandBuffer.present(currentDrawable)
        commandBuffer.commit()
    }
}

// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, managing camera capture for real-time inference.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The VideoCapture component manages the camera and video processing pipeline for real-time
//  object detection. It handles setting up the AVCaptureSession, managing camera devices,
//  configuring camera properties like focus and exposure, and processing video frames for
//  model inference. The class delivers capture frames to the predictor component for real-time
//  analysis and returns results through delegate callbacks. It also supports camera controls
//  such as switching between front and back cameras, zooming, and capturing still photos.

import AVFoundation
import CoreVideo
import UIKit
import Vision

/// Protocol for receiving video capture frame processing results.
@MainActor
protocol VideoCaptureDelegate: AnyObject {
  func onPredict(result: YOLOResult)
  func onInferenceTime(speed: Double, fps: Double)
}

func bestCaptureDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
  if UserDefaults.standard.bool(forKey: "use_telephoto"),
    let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: position)
  {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInDualCamera, for: .video, position: position)
  {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInWideAngleCamera, for: .video, position: position)
  {
    return device
  } else {
    return nil
  }
}

class VideoCapture: NSObject, @unchecked Sendable {
  var predictor: Predictor!
  var previewLayer: AVCaptureVideoPreviewLayer?
  weak var delegate: VideoCaptureDelegate?
  var captureDevice: AVCaptureDevice?
  let captureSession = AVCaptureSession()
  var videoInput: AVCaptureDeviceInput? = nil
  let videoOutput = AVCaptureVideoDataOutput()
  var photoOutput = AVCapturePhotoOutput()
  let cameraQueue = DispatchQueue(label: "camera-queue")
  var lastCapturedPhoto: UIImage? = nil
  var inferenceOK = true
  var longSide: CGFloat = 3
  var shortSide: CGFloat = 4
  var frameSizeCaptured = false

  private var currentBuffer: CVPixelBuffer?

  // Normalized crop region [0,1] applied before inference and encoding.
  var cropLeft: CGFloat = 0.0
  var cropTop: CGFloat = 0.0
  var cropRight: CGFloat = 1.0
  var cropBottom: CGFloat = 1.0
  var cropEnabled = false

  func setCropRegion(left: CGFloat, top: CGFloat, right: CGFloat, bottom: CGFloat) {
    cropLeft = left; cropTop = top; cropRight = right; cropBottom = bottom
    cropEnabled = left > 0.001 || top > 0.001 || right < 0.999 || bottom < 0.999
  }

  private func makeCroppedSampleBuffer(_ original: CMSampleBuffer) -> CMSampleBuffer? {
    guard cropEnabled,
      let pixelBuffer = CMSampleBufferGetImageBuffer(original)
    else { return nil }

    let srcW = CVPixelBufferGetWidth(pixelBuffer)
    let srcH = CVPixelBufferGetHeight(pixelBuffer)
    let x = Int(cropLeft * CGFloat(srcW)) & ~1
    let y = Int(cropTop * CGFloat(srcH)) & ~1
    let w = max(64, Int((cropRight - cropLeft) * CGFloat(srcW)) & ~1)
    let h = max(64, Int((cropBottom - cropTop) * CGFloat(srcH)) & ~1)
    let cx = max(0, min(x, srcW - w))
    let cy = max(0, min(y, srcH - h))

    let fmt = CVPixelBufferGetPixelFormatType(pixelBuffer)
    var croppedPB: CVPixelBuffer?
    guard CVPixelBufferCreate(nil, w, h, fmt, nil, &croppedPB) == kCVReturnSuccess,
      let dst = croppedPB
    else { return nil }

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    CVPixelBufferLockBaseAddress(dst, [])
    defer {
      CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
      CVPixelBufferUnlockBaseAddress(dst, [])
    }

    let planeCount = CVPixelBufferGetPlaneCount(pixelBuffer)
    if planeCount == 0 {
      guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer),
        let dstBase = CVPixelBufferGetBaseAddress(dst)
      else { return nil }
      let srcBPR = CVPixelBufferGetBytesPerRow(pixelBuffer)
      let dstBPR = CVPixelBufferGetBytesPerRow(dst)
      let bpp = srcBPR / srcW
      for row in 0..<h {
        memcpy(dstBase + row * dstBPR, srcBase + (cy + row) * srcBPR + cx * bpp, w * bpp)
      }
    } else {
      for plane in 0..<planeCount {
        let scale = plane == 0 ? 1 : 2
        guard let srcBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, plane),
          let dstBase = CVPixelBufferGetBaseAddressOfPlane(dst, plane)
        else { continue }
        let srcBPR = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, plane)
        let dstBPR = CVPixelBufferGetBytesPerRowOfPlane(dst, plane)
        let pw = CVPixelBufferGetWidthOfPlane(dst, plane)
        let ph = CVPixelBufferGetHeightOfPlane(dst, plane)
        let bpp = srcBPR / (srcW / scale)
        for row in 0..<ph {
          memcpy(dstBase + row * dstBPR, srcBase + (cy/scale + row) * srcBPR + (cx/scale) * bpp, pw * bpp)
        }
      }
    }

    var timing = CMSampleTimingInfo()
    CMSampleBufferGetSampleTimingInfo(original, at: 0, timingInfoOut: &timing)
    var fmtDesc: CMFormatDescription?
    CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: dst, formatDescriptionOut: &fmtDesc)
    guard let fd = fmtDesc else { return nil }
    var out: CMSampleBuffer?
    CMSampleBufferCreateReadyWithImageBuffer(
      allocator: nil, imageBuffer: dst, formatDescription: fd,
      sampleTiming: &timing, sampleBufferOut: &out)
    return out
  }

  func setUp(
    sessionPreset: AVCaptureSession.Preset = .hd1280x720,
    position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation,
    completion: @escaping (Bool) -> Void
  ) {
    cameraQueue.async {
      let success = self.setUpCamera(
        sessionPreset: sessionPreset, position: position, orientation: orientation)
      DispatchQueue.main.async {
        completion(success)
      }
    }
  }

  func setUpCamera(
    sessionPreset: AVCaptureSession.Preset, position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation
  ) -> Bool {

    let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
    if authStatus == .denied || authStatus == .restricted {
      NSLog("YOLO VideoCapture: Camera permission denied or restricted. Cannot initialize camera.")
      return false
    }

    if authStatus == .notDetermined {
      NSLog(
        "YOLO VideoCapture: Camera permission not determined. Please request permission first.")
      return false
    }

    captureSession.beginConfiguration()
    captureSession.sessionPreset = sessionPreset

    guard let device = bestCaptureDevice(position: position) else {
      NSLog(
        "YOLO VideoCapture: No camera device available for position: %@",
        String(describing: position))
      captureSession.commitConfiguration()
      return false
    }

    captureDevice = device

    let input: AVCaptureDeviceInput
    do {
      input = try AVCaptureDeviceInput(device: device)
    } catch {
      NSLog(
        "YOLO VideoCapture: Failed to create AVCaptureDeviceInput: %@", error.localizedDescription)
      captureSession.commitConfiguration()
      return false
    }

    videoInput = input

    if captureSession.canAddInput(input) {
      captureSession.addInput(input)
    } else {
      NSLog("YOLO VideoCapture: Cannot add video input to capture session")
      captureSession.commitConfiguration()
      return false
    }
    var videoOrientaion = AVCaptureVideoOrientation.portrait
    switch orientation {
    case .portrait:
      videoOrientaion = .portrait
    case .landscapeLeft:
      videoOrientaion = .landscapeRight
    case .landscapeRight:
      videoOrientaion = .landscapeLeft
    default:
      videoOrientaion = .portrait
    }
    let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
    previewLayer.connection?.videoOrientation = videoOrientaion
    self.previewLayer = previewLayer

    let settings: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]

    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
    if captureSession.canAddOutput(videoOutput) {
      captureSession.addOutput(videoOutput)
    }
    if captureSession.canAddOutput(photoOutput) {
      captureSession.addOutput(photoOutput)
      photoOutput.isHighResolutionCaptureEnabled = true
      //            photoOutput.isLivePhotoCaptureEnabled = photoOutput.isLivePhotoCaptureSupported
    }

    // We want the buffers to be in portrait orientation otherwise they are
    // rotated by 90 degrees. Need to set this _after_ addOutput()!
    // let curDeviceOrientation = UIDevice.current.orientation
    let connection = videoOutput.connection(with: AVMediaType.video)
    connection?.videoOrientation = videoOrientaion
    if position == .front {
      connection?.isVideoMirrored = true
    }

    // Configure captureDevice
    guard let device = captureDevice else {
      NSLog("YOLO VideoCapture: captureDevice is nil, cannot configure")
      captureSession.commitConfiguration()
      return false
    }

    do {
      try device.lockForConfiguration()

      if device.isFocusModeSupported(AVCaptureDevice.FocusMode.continuousAutoFocus),
        device.isFocusPointOfInterestSupported
      {
        device.focusMode = AVCaptureDevice.FocusMode.continuousAutoFocus
        device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
      }
      device.exposureMode = AVCaptureDevice.ExposureMode.continuousAutoExposure
      device.unlockForConfiguration()
    } catch {
      NSLog("YOLO VideoCapture: device configuration failed: %@", error.localizedDescription)
      captureSession.commitConfiguration()
      return false
    }

    captureSession.commitConfiguration()
    return true
  }

  func start() {
    if !captureSession.isRunning {
      DispatchQueue.global().async {
        self.captureSession.startRunning()
      }
    }
  }

  func stop() {
    if captureSession.isRunning {
      DispatchQueue.global().async {
        self.captureSession.stopRunning()
      }
    }
  }

  func setZoomRatio(ratio: CGFloat) {
    guard let device = captureDevice else {
      NSLog("YOLO VideoCapture: Cannot set zoom: captureDevice is nil")
      return
    }
    do {
      try device.lockForConfiguration()
      defer {
        device.unlockForConfiguration()
      }
      device.videoZoomFactor = ratio
    } catch {
      NSLog("YOLO VideoCapture: Failed to set zoom ratio: %@", error.localizedDescription)
    }
  }

  private func predictOnFrame(sampleBuffer: CMSampleBuffer) {
    guard let predictor = predictor else {
      return
    }
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer
      if !frameSizeCaptured {
        let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        longSide = max(frameWidth, frameHeight)
        shortSide = min(frameWidth, frameHeight)
        frameSizeCaptured = true
      }

      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
      var imageOrientation: CGImagePropertyOrientation = .up
      //            switch UIDevice.current.orientation {
      //            case .portrait:
      //                imageOrientation = .up
      //            case .portraitUpsideDown:
      //                imageOrientation = .down
      //            case .landscapeLeft:
      //                imageOrientation = .up
      //            case .landscapeRight:
      //                imageOrientation = .up
      //            case .unknown:
      //                imageOrientation = .up
      //
      //            default:
      //                imageOrientation = .up
      //            }

      predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
      currentBuffer = nil
    }
  }

  func updateVideoOrientation(orientation: AVCaptureVideoOrientation) {
    guard let connection = videoOutput.connection(with: .video) else { return }

    connection.videoOrientation = orientation
    let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
    if currentInput?.device.position == .front {
      connection.isVideoMirrored = true
    } else {
      connection.isVideoMirrored = false
    }
    let o = connection.videoOrientation
    self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
  }

  deinit {
    if captureSession.isRunning {
      captureSession.stopRunning()
    }

    // Remove all inputs and outputs
    if let inputs = captureSession.inputs as? [AVCaptureInput] {
      for input in inputs {
        captureSession.removeInput(input)
      }
    }

    if let outputs = captureSession.outputs as? [AVCaptureOutput] {
      for output in outputs {
        captureSession.removeOutput(output)
      }
    }
  }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard inferenceOK else { return }
    let effective = makeCroppedSampleBuffer(sampleBuffer) ?? sampleBuffer
    predictOnFrame(sampleBuffer: effective)
  }
}

extension VideoCapture: AVCapturePhotoCaptureDelegate {
  @available(iOS 11.0, *)
  func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    guard let data = photo.fileDataRepresentation(),
      let image = UIImage(data: data)
    else {
      return
    }

    self.lastCapturedPhoto = image
  }
}

extension VideoCapture: ResultsListener, InferenceTimeListener {
  func on(inferenceTime: Double, fpsRate: Double) {
    DispatchQueue.main.async {
      self.delegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
    }
  }

  func on(result: YOLOResult) {
    DispatchQueue.main.async {
      self.delegate?.onPredict(result: result)
    }
  }
}

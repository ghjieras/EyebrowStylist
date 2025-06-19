//
//  ViewController.swift
//  EyebrowStylist
//
//  Created by 呂念浯 on 2025/6/19.
//

import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private var lastProcessTime: CFTimeInterval = 0
    private var captureSession: AVCaptureSession?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let faceDetectionRequest = VNDetectFaceLandmarksRequest()
    private let sequenceHandler = VNSequenceRequestHandler()
    private var overlayLayer: CALayer?
    
    // 儲存當前影像尺寸
    private var currentImageSize: CGSize = .zero
    private var imageOrientation: CGImagePropertyOrientation = .up
    
    // 用於平滑處理的歷史資料
    private var previousFaceRect: CGRect?
    private let smoothingFactor: CGFloat = 0.7  // 平滑因子，0-1 之間
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 基本的臉部偵測優化
        faceDetectionRequest.revision = VNDetectFaceLandmarksRequestRevision3
        
        checkCameraPermission()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        overlayLayer?.frame = view.bounds
    }
    
    func checkCameraPermission(){
        switch AVCaptureDevice.authorizationStatus(for: .video){
        case .authorized:
            print("相機權限已授權")
            setupCamera()
            
        case .notDetermined:
            print("請求相機權限")
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        self.setupCamera()
                    } else{
                        print("相機權限被拒絕")
                    }
                }
            }
        case .denied, .restricted:
            print("相機權限被拒絕或受限")
        @unknown default:
            print("未知的權限狀態")
        }
    }
    
    func setupCamera() {
        print("開始設定相機")
        
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .high
        
        guard let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            print("無法取得前置相機")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: frontCamera)
            
            if captureSession?.canAddInput(input) == true {
                captureSession?.addInput(input)
            }
            
            // 設定影像輸出用於臉部偵測
            let videoOutput = AVCaptureVideoDataOutput()
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            
            // 設定像素格式
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
            
            // 如果處理不及時就丟棄
            videoOutput.alwaysDiscardsLateVideoFrames = true
            
            if captureSession?.canAddOutput(videoOutput) == true {
                captureSession?.addOutput(videoOutput)
            }
            
            // 設定視訊連接方向
            if let connection = videoOutput.connection(with: .video) {
                connection.videoOrientation = .portrait
                // 前置相機需要鏡像
                if frontCamera.position == .front {
                    connection.isVideoMirrored = true
                }
            }
            
            setupPreviewLayer()
            captureSession?.startRunning()
            
        } catch {
            print("相機設定錯誤: \(error)")
        }
    }
    
    func setupPreviewLayer() {
        guard let captureSession = captureSession else { return }
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.frame = view.layer.bounds
        
        if let previewLayer = previewLayer {
            view.layer.addSublayer(previewLayer)
        }
        
        setupOverlayLayer()
        print("相機預覽層已設定")
    }
    
    func setupOverlayLayer() {
        overlayLayer?.removeFromSuperlayer()
        overlayLayer = CALayer()
        overlayLayer?.frame = view.bounds
        view.layer.addSublayer(overlayLayer!)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastProcessTime > 0.05 else { return }  // 提高更新頻率
        lastProcessTime = currentTime
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // 取得影像尺寸
        currentImageSize = CGSize(
            width: CVPixelBufferGetWidth(pixelBuffer),
            height: CVPixelBufferGetHeight(pixelBuffer)
        )
        
        // 設定影像方向（前置相機已鏡像）
        imageOrientation = .upMirrored
        
        // 執行臉部偵測
        let request = VNDetectFaceLandmarksRequest { [weak self] request, error in
            guard let results = request.results as? [VNFaceObservation], !results.isEmpty else {
                // 沒有偵測到臉部時清除畫面
                DispatchQueue.main.async {
                    self?.overlayLayer?.sublayers?.removeAll()
                }
                return
            }
            
            DispatchQueue.main.async {
                self?.handleFaceDetection(results: results)
            }
        }
        
        request.revision = VNDetectFaceLandmarksRequestRevision3
        
        try? sequenceHandler.perform([request], on: pixelBuffer, orientation: imageOrientation)
    }
    
    func handleFaceDetection(results: [VNFaceObservation]) {
        for face in results {
            print("偵測到臉部，信心度:\(face.confidence)")
            print("臉部位置: \(face.boundingBox)")
            
            if let landmarks = face.landmarks {
                // 詳細記錄偵測狀態
                var detectionInfo = "偵測狀態: "
                if let leftEyebrow = landmarks.leftEyebrow {
                    detectionInfo += "左眉(\(leftEyebrow.pointCount)點) "
                }
                if let rightEyebrow = landmarks.rightEyebrow {
                    detectionInfo += "右眉(\(rightEyebrow.pointCount)點) "
                }
                print(detectionInfo)
                
                drawFacialFeatures(face: face)
                
                // 顯示品質評分供參考
                let faceQuality = evaluateFaceQuality(face: face, landmarks: landmarks)
                print("臉部品質評分：\(faceQuality)")
            }
            break // 只處理第一張臉
        }
    }
    
    func drawFacialFeatures(face: VNFaceObservation) {
        overlayLayer?.sublayers?.removeAll()
        
        guard let landmarks = face.landmarks else { return }
        
        // 繪製臉部邊界框（用於調試）
        drawFaceBoundingBox(face: face)
        
        // 檢查角度和偵測狀態
        var isLargeAngle = false
        var yawDirection = ""  // "left" 或 "right"
        if let yaw = face.yaw {
            let yawDegrees = yaw.doubleValue * 180 / .pi
            isLargeAngle = abs(yawDegrees) > 35
            yawDirection = yawDegrees < 0 ? "left" : "right"
        }
        
        // 偵測到的眉毛數量
        var detectedEyebrows = 0
        
        // 只在確實偵測到左眉毛時才繪製
        if let leftEyebrow = landmarks.leftEyebrow, leftEyebrow.pointCount > 0 {
            let color = isLargeAngle ? UIColor.orange : UIColor.red
            drawEyebrowPoints(eyebrow: leftEyebrow, face: face, color: color)
            drawEyebrowPath(eyebrow: leftEyebrow, face: face, color: color.withAlphaComponent(0.5))
            detectedEyebrows += 1
            
            // 顯示左眉毛偵測狀態
            let statusLayer = CATextLayer()
            statusLayer.string = "L"
            statusLayer.fontSize = 14
            statusLayer.foregroundColor = color.cgColor
            statusLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
            statusLayer.frame = CGRect(x: 50, y: 10, width: 20, height: 20)
            statusLayer.alignmentMode = .center
            overlayLayer?.addSublayer(statusLayer)
        }
        
        // 只在確實偵測到右眉毛時才繪製
        if let rightEyebrow = landmarks.rightEyebrow, rightEyebrow.pointCount > 0 {
            let color = isLargeAngle ? UIColor.yellow : UIColor.blue
            drawEyebrowPoints(eyebrow: rightEyebrow, face: face, color: color)
            drawEyebrowPath(eyebrow: rightEyebrow, face: face, color: color.withAlphaComponent(0.5))
            detectedEyebrows += 1
            
            // 顯示右眉毛偵測狀態
            let statusLayer = CATextLayer()
            statusLayer.string = "R"
            statusLayer.fontSize = 14
            statusLayer.foregroundColor = color.cgColor
            statusLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
            statusLayer.frame = CGRect(x: 75, y: 10, width: 20, height: 20)
            statusLayer.alignmentMode = .center
            overlayLayer?.addSublayer(statusLayer)
        }
        
        // 繪製其他參考點（用於調試）
        if let leftEye = landmarks.leftEye, leftEye.pointCount > 0 {
            drawLandmarkPoints(landmark: leftEye, face: face, color: UIColor.green.withAlphaComponent(0.5))
        }
        if let rightEye = landmarks.rightEye, rightEye.pointCount > 0 {
            drawLandmarkPoints(landmark: rightEye, face: face, color: UIColor.cyan.withAlphaComponent(0.5))
        }
        
        // 顯示偵測狀態資訊
        let infoLayer = CATextLayer()
        var statusText = "偵測到 \(detectedEyebrows) 邊眉毛"
        if isLargeAngle {
            statusText += " (大角度\(yawDirection == "left" ? "左轉" : "右轉"))"
        }
        infoLayer.string = statusText
        infoLayer.fontSize = 14
        infoLayer.foregroundColor = UIColor.white.cgColor
        infoLayer.backgroundColor = UIColor.black.withAlphaComponent(0.7).cgColor
        infoLayer.frame = CGRect(x: 10, y: 35, width: 200, height: 25)
        infoLayer.alignmentMode = .left
        overlayLayer?.addSublayer(infoLayer)
    }
    
    func drawFaceBoundingBox(face: VNFaceObservation) {
        var convertedRect = convertNormalizedRect(face.boundingBox)
        
        // 應用平滑處理
        if let previousRect = previousFaceRect {
            convertedRect = CGRect(
                x: previousRect.origin.x + (convertedRect.origin.x - previousRect.origin.x) * (1 - smoothingFactor),
                y: previousRect.origin.y + (convertedRect.origin.y - previousRect.origin.y) * (1 - smoothingFactor),
                width: previousRect.width + (convertedRect.width - previousRect.width) * (1 - smoothingFactor),
                height: previousRect.height + (convertedRect.height - previousRect.height) * (1 - smoothingFactor)
            )
        }
        previousFaceRect = convertedRect
        
        let boxLayer = CAShapeLayer()
        boxLayer.path = UIBezierPath(rect: convertedRect).cgPath
        boxLayer.strokeColor = UIColor.green.cgColor
        boxLayer.lineWidth = 2
        boxLayer.fillColor = UIColor.clear.cgColor
        
        overlayLayer?.addSublayer(boxLayer)
        
        // 顯示臉部角度資訊
        if let yaw = face.yaw, let pitch = face.pitch, let roll = face.roll {
            let infoLayer = CATextLayer()
            let yawDeg = Int(yaw.doubleValue * 180 / .pi)
            let pitchDeg = Int(pitch.doubleValue * 180 / .pi)
            let rollDeg = Int(roll.doubleValue * 180 / .pi)
            
            infoLayer.string = "Yaw: \(yawDeg)° Pitch: \(pitchDeg)° Roll: \(rollDeg)°"
            infoLayer.fontSize = 12
            infoLayer.foregroundColor = UIColor.white.cgColor
            infoLayer.backgroundColor = UIColor.black.withAlphaComponent(0.7).cgColor
            infoLayer.frame = CGRect(x: convertedRect.minX,
                                    y: convertedRect.maxY + 5,
                                    width: 200,
                                    height: 20)
            infoLayer.alignmentMode = .left
            
            overlayLayer?.addSublayer(infoLayer)
        }
    }
    
    func drawEyebrowPoints(eyebrow: VNFaceLandmarkRegion2D, face: VNFaceObservation, color: UIColor) {
        let points = eyebrow.normalizedPoints
        
        for (index, point) in points.enumerated() {
            let convertedPoint = convertNormalizedPoint(point, inFace: face)
            
            let circleLayer = CAShapeLayer()
            let radius: CGFloat = 5
            let circlePath = UIBezierPath(arcCenter: convertedPoint,
                                         radius: radius,
                                         startAngle: 0,
                                         endAngle: .pi * 2,
                                         clockwise: true)
            
            circleLayer.path = circlePath.cgPath
            circleLayer.fillColor = color.cgColor
            
            overlayLayer?.addSublayer(circleLayer)
            
            // 添加點的編號標籤
            let textLayer = CATextLayer()
            textLayer.string = "\(index)"
            textLayer.fontSize = 10
            textLayer.foregroundColor = UIColor.white.cgColor
            textLayer.backgroundColor = UIColor.black.cgColor
            textLayer.frame = CGRect(x: convertedPoint.x - 10,
                                    y: convertedPoint.y - 20,
                                    width: 20,
                                    height: 15)
            textLayer.alignmentMode = .center
            
            overlayLayer?.addSublayer(textLayer)
        }
    }
    
    func drawLandmarkPoints(landmark: VNFaceLandmarkRegion2D, face: VNFaceObservation, color: UIColor) {
        let points = landmark.normalizedPoints
        
        for point in points {
            let convertedPoint = convertNormalizedPoint(point, inFace: face)
            
            let circleLayer = CAShapeLayer()
            let radius: CGFloat = 3
            let circlePath = UIBezierPath(arcCenter: convertedPoint,
                                         radius: radius,
                                         startAngle: 0,
                                         endAngle: .pi * 2,
                                         clockwise: true)
            
            circleLayer.path = circlePath.cgPath
            circleLayer.fillColor = color.cgColor
            
            overlayLayer?.addSublayer(circleLayer)
        }
    }
    
    func drawEyebrowPath(eyebrow: VNFaceLandmarkRegion2D, face: VNFaceObservation, color: UIColor) {
        let path = UIBezierPath()
        let points = eyebrow.normalizedPoints
        
        for (index, point) in points.enumerated() {
            let convertedPoint = convertNormalizedPoint(point, inFace: face)
            
            if index == 0 {
                path.move(to: convertedPoint)
            } else {
                path.addLine(to: convertedPoint)
            }
        }
        
        let shapeLayer = CAShapeLayer()
        shapeLayer.path = path.cgPath
        shapeLayer.strokeColor = color.cgColor
        shapeLayer.lineWidth = 3
        shapeLayer.fillColor = UIColor.clear.cgColor
        
        overlayLayer?.addSublayer(shapeLayer)
    }
    
    // 轉換 Vision 歸一化座標點到螢幕座標
    func convertNormalizedPoint(_ normalizedPoint: CGPoint, inFace face: VNFaceObservation) -> CGPoint {
        // 1. 將歸一化的臉部特徵點轉換到臉部邊界框內的位置
        let pointInFace = CGPoint(
            x: face.boundingBox.origin.x + normalizedPoint.x * face.boundingBox.width,
            y: face.boundingBox.origin.y + normalizedPoint.y * face.boundingBox.height
        )
        
        // 2. 轉換到螢幕座標
        return convertVisionToScreen(pointInFace)
    }
    
    // 轉換 Vision 歸一化矩形到螢幕座標
    func convertNormalizedRect(_ normalizedRect: CGRect) -> CGRect {
        let viewSize = view.bounds.size
        
        // Vision 座標系統：原點在左下角
        // UIKit 座標系統：原點在左上角
        // 需要翻轉 Y 軸，並且要考慮矩形的高度
        let screenRect = CGRect(
            x: normalizedRect.origin.x * viewSize.width,
            y: (1.0 - normalizedRect.origin.y - normalizedRect.height) * viewSize.height,
            width: normalizedRect.width * viewSize.width,
            height: normalizedRect.height * viewSize.height
        )
        
        return screenRect
    }
    
    // Vision 座標系統到螢幕座標系統的轉換
    func convertVisionToScreen(_ visionPoint: CGPoint) -> CGPoint {
        // Vision: (0,0) 在左下角，Y 軸向上
        // UIKit: (0,0) 在左上角，Y 軸向下
        
        let viewSize = view.bounds.size
        
        // 基本座標轉換
        var screenPoint = CGPoint(
            x: visionPoint.x * viewSize.width,
            y: (1.0 - visionPoint.y) * viewSize.height
        )
        
        // 如果是前置相機且已經在 connection 中設定了鏡像，就不需要再次鏡像
        // 因為 Vision 已經處理了鏡像的影像
        
        return screenPoint
    }
    
    func evaluateFaceQuality(face: VNFaceObservation, landmarks: VNFaceLandmarks2D) -> Float {
        var qualityScore: Float = face.confidence
        
        //  檢查是否有完整的眉毛資料
        if landmarks.leftEyebrow != nil && landmarks.rightEyebrow != nil {
            qualityScore += 0.2
        } else if landmarks.leftEyebrow != nil || landmarks.rightEyebrow != nil {
            // 即使只有一邊眉毛也給一些分數
            qualityScore += 0.1
        }
        
        let faceArea = face.boundingBox.width * face.boundingBox.height
        if faceArea > 0.1 {
            qualityScore += 0.2
        }
        
        //檢查是否有眼睛判斷是否為正面
        if landmarks.leftEye != nil && landmarks.rightEye != nil {
            qualityScore += 0.1
            
            // 檢查臉部是否太側面（透過檢查眼睛和鼻子的位置）
            if let nose = landmarks.nose {
                // 如果能偵測到鼻子，表示臉部較正面
                qualityScore += 0.1
            }
        }
        
        // 檢查臉部旋轉角度（透過 roll, pitch, yaw）
        if let roll = face.roll, let yaw = face.yaw, let pitch = face.pitch {
            // Roll: 頭部傾斜角度（左右歪頭）
            let rollDegrees = roll.doubleValue * 180 / .pi
            if abs(rollDegrees) < 30 {
                qualityScore += 0.05
            }
            
            // Yaw: 頭部左右轉動角度
            let yawDegrees = yaw.doubleValue * 180 / .pi
            if abs(yawDegrees) < 35 {  // 容許 35 度以內的左右轉動
                qualityScore += 0.05
            }
            // 移除大角度的懲罰，讓大角度也能保持一定的品質分數
            
            // Pitch: 頭部上下角度
            let pitchDegrees = pitch.doubleValue * 180 / .pi
            if abs(pitchDegrees) < 30 {
                qualityScore += 0.05
            }
            
            print("頭部角度 - Roll: \(rollDegrees)°, Yaw: \(yawDegrees)°, Pitch: \(pitchDegrees)°")
        }
        
        return min(max(qualityScore, 0), 1.0)  // 確保在 0-1 範圍內
    }
}

// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.*
import android.widget.FrameLayout
import android.widget.Toast
import android.view.ScaleGestureDetector
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.io.ByteArrayOutputStream
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.CountDownLatch
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.video.FallbackStrategy
import android.graphics.YuvImage
import android.graphics.Rect as GraphicsRect
import kotlin.math.max
import kotlin.math.min
import android.widget.TextView
import android.view.Gravity
import android.content.res.Configuration
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat

class YOLOView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : FrameLayout(context, attrs), DefaultLifecycleObserver {

    // Lifecycle owner for camera
    private var lifecycleOwner: LifecycleOwner? = null

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
        )
        private var previewUseCase: Preview? = null

        private const val TAG = "YOLOView"
        const val UDP_CHUNK_SIZE = 1400

        // Line thickness and corner radius
        private const val BOX_LINE_WIDTH = 8f
        private const val BOX_CORNER_RADIUS = 12f
        private const val KEYPOINT_LINE_WIDTH = 6f

        // Colors derived from Ultralytics
        private val ultralyticsColors = arrayOf(
            Color.argb(153, 4,   42,  255),
            Color.argb(153, 11,  219, 235),
            Color.argb(153, 243, 243, 243),
            Color.argb(153, 0,   223, 183),
            Color.argb(153, 17,  31,  104),
            Color.argb(153, 255, 111, 221),
            Color.argb(153, 255, 68,  79),
            Color.argb(153, 204, 237, 0),
            Color.argb(153, 0,   243, 68),
            Color.argb(153, 189, 0,   255),
            Color.argb(153, 0,   180, 255),
            Color.argb(153, 221, 0,   186),
            Color.argb(153, 0,   255, 255),
            Color.argb(153, 38,  192, 0),
            Color.argb(153, 1,   255, 179),
            Color.argb(153, 125, 36,  255),
            Color.argb(153, 123, 0,   104),
            Color.argb(153, 255, 27,  108),
            Color.argb(153, 252, 109, 47),
            Color.argb(153, 162, 255, 11)
        )

        // Pose
        private val posePalette = arrayOf(
            floatArrayOf(255f, 128f,  0f),
            floatArrayOf(255f, 153f,  51f),
            floatArrayOf(255f, 178f, 102f),
            floatArrayOf(230f, 230f,   0f),
            floatArrayOf(255f, 153f, 255f),
            floatArrayOf(153f, 204f, 255f),
            floatArrayOf(255f, 102f, 255f),
            floatArrayOf(255f,  51f, 255f),
            floatArrayOf(102f, 178f, 255f),
            floatArrayOf( 51f, 153f, 255f),
            floatArrayOf(255f, 153f, 153f),
            floatArrayOf(255f, 102f, 102f),
            floatArrayOf(255f,  51f,  51f),
            floatArrayOf(153f, 255f, 153f),
            floatArrayOf(102f, 255f, 102f),
            floatArrayOf( 51f, 255f,  51f),
            floatArrayOf(  0f, 255f,   0f),
            floatArrayOf(  0f,   0f, 255f),
            floatArrayOf(255f,   0f,   0f),
            floatArrayOf(255f, 255f, 255f),
        )

        private val kptColorIndices = intArrayOf(
            16,16,16,16,16,
            9, 9, 9, 9, 9, 9,
            0, 0, 0, 0, 0, 0
        )

        private val limbColorIndices = intArrayOf(
            0, 0, 0, 0,
            7, 7, 7,
            9, 9, 9, 9, 9,
            16,16,16,16,16,16,16
        )

        private val skeleton = arrayOf(
            intArrayOf(16, 14),
            intArrayOf(14, 12),
            intArrayOf(17, 15),
            intArrayOf(15, 13),
            intArrayOf(12, 13),
            intArrayOf(6, 12),
            intArrayOf(7, 13),
            intArrayOf(6, 7),
            intArrayOf(6, 8),
            intArrayOf(7, 9),
            intArrayOf(8, 10),
            intArrayOf(9, 11),
            intArrayOf(2, 3),
            intArrayOf(1, 2),
            intArrayOf(1, 3),
            intArrayOf(2, 4),
            intArrayOf(3, 5),
            intArrayOf(4, 6),
            intArrayOf(5, 7)
        )
    }

    // Callback to notify inference results externally
    private var inferenceCallback: ((YOLOResult) -> Unit)? = null
    
    // Streaming functionality
    private var streamConfig: YOLOStreamConfig? = null
    private var streamCallback: ((Map<String, Any>) -> Unit)? = null
    
    // Frame counter for streaming
    private val frameNumberCounter = AtomicLong(0)
    
    // Throttling variables for performance control
    private var lastInferenceTime: Long = 0
    private var targetFrameInterval: Long? = null // in nanoseconds
    private var throttleInterval: Long? = null // in nanoseconds
    
    // Inference frequency control variables
    private var inferenceFrameInterval: Long? = null // Target inference interval in nanoseconds
    private var frameSkipCount: Int = 0 // Current frame skip counter
    private var targetSkipFrames: Int = 0 // Number of frames to skip between inferences

    /** Set the callback */
    fun setOnInferenceCallback(callback: (YOLOResult) -> Unit) {
        this.inferenceCallback = callback
    }
    
    /** Set streaming configuration */
    fun setStreamConfig(config: YOLOStreamConfig?) {
        this.streamConfig = config
        setupThrottlingFromConfig()
    }

    /** Set streaming callback */
    fun setStreamCallback(callback: ((Map<String, Any>) -> Unit)?) {
        this.streamCallback = callback
    }

    // Callback to notify model load completion
    private var modelLoadCallback: ((Boolean) -> Unit)? = null

    /** Set model load completion callback (true: success) */
    fun setOnModelLoadCallback(callback: (Boolean) -> Unit) {
        this.modelLoadCallback = callback
    }

    // Use a PreviewView, forcing a TextureView under the hood
    private val previewView: PreviewView = PreviewView(context).apply {
        // Force TextureView usage so the overlay can be on top
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        scaleType = PreviewView.ScaleType.FILL_CENTER
    }

    // The overlay for bounding boxes
    private val overlayView: OverlayView = OverlayView(context)

    private var inferenceResult: YOLOResult? = null
    @Volatile private var predictor: Predictor? = null
    private var task: YOLOTask = YOLOTask.DETECT
    private var modelName: String = "Model"

    // Camera config
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private var camera: Camera? = null

    // New fields for proper teardown:
    private var cameraExecutor: ExecutorService? = null
    private var imageAnalysisUseCase: ImageAnalysis? = null
    
    // Flag to track if the view is stopped/disposed to prevent race conditions
    @Volatile
    private var isStopped = false

    // RTMP frame export
    @Volatile private var rtmpEnabled = false
    private var lastRtmpMs = 0L
    private val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var rtmpFrameCallback: ((ByteArray, Int, Int) -> Unit)? = null

    // UDP streaming — H.264 hardware-encoded via MediaCodec
    @Volatile private var udpTarget: InetSocketAddress? = null
    private var udpSocket: DatagramSocket? = null
    private val udpFrameSeq = AtomicInteger(0)
    @Volatile private var udpPipelineRunning = false
    @Volatile private var mediaCodec: MediaCodec? = null
    @Volatile private var h264OutputRunning = false
    private var h264OutputThread: Thread? = null
    @Volatile private var codecConfigData: ByteArray? = null  // SPS+PPS, prepended to every IDR
    private var nv12ScratchBuffer: ByteArray? = null  // reused per-frame to avoid allocation
    private val h264DroppedFrames = AtomicInteger(0)
    private val h264SubmittedFrames = AtomicInteger(0)
    private val cameraFrameCounter = AtomicInteger(0)
    @Volatile private var cameraTickMs = 0L

    // Native recording via VideoCapture<Recorder> — dedicated camera surface,
    // independent of ImageAnalysis, no frame drops, hardware-correct timestamps.
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    @Volatile private var nativeRecordingFilePath: String? = null
    @Volatile private var recordingFinalizeLatch: CountDownLatch? = null
    @Volatile private var recordingStopRequested = false
    var onRecordingStoppedUnexpectedly: (() -> Unit)? = null
    @Volatile var lastFrameWidth: Int = 0
    @Volatile var lastFrameHeight: Int = 0

    // Async inference — YOLO runs on its own thread so it doesn't block frame capture
    // Queue carries raw NV21 bytes so bitmap decoding happens off the camera thread
    private val inferenceQueue = ArrayBlockingQueue<Triple<ByteArray, Int, Int>>(1)
    @Volatile private var inferenceThreadRunning = false
    private var inferenceThread: Thread? = null

    // Zoom related
    private var currentZoomRatio = 1.0f
    private var minZoomRatio = 1.0f
    private var maxZoomRatio = 10.0f
    private lateinit var scaleGestureDetector: ScaleGestureDetector
    var onZoomChanged: ((Float) -> Unit)? = null

    // detection thresholds (can be changed externally via setters)
    private var confidenceThreshold = 0.25  // initial value
    private var iouThreshold = 0.7
    private var numItemsThreshold = 30
    private var showOverlays = true
    private lateinit var zoomLabel: TextView
    private lateinit var cameraButton: TextView
    private lateinit var confidenceLabel: TextView
    private var showUIControls = false

    init {
        // Clear any existing children
        removeAllViews()

        // 1) A container for the camera preview
        val previewContainer = FrameLayout(context).apply {
            layoutParams = LayoutParams(
                LayoutParams.MATCH_PARENT,
                LayoutParams.MATCH_PARENT
            )
        }

        // 2) Add the previewView to that container
        previewContainer.addView(previewView, LayoutParams(
            LayoutParams.MATCH_PARENT,
            LayoutParams.MATCH_PARENT
        ))

        // 3) Add that container
        addView(previewContainer)

        // 4) Add the overlay on top
        addView(overlayView, LayoutParams(
            LayoutParams.MATCH_PARENT,
            LayoutParams.MATCH_PARENT
        ))

        // Ensure overlay is visually above the preview container
        overlayView.elevation = 100f
        overlayView.translationZ = 100f
        previewContainer.elevation = 1f
        
        // Add zoom label
        zoomLabel = TextView(context).apply {
            layoutParams = LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.CENTER
            }
            text = "ZOOM: 1.0x"
            textSize = 28f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(200, 255, 0, 0))
            setPadding(20, 15, 20, 15)
            visibility = View.GONE
        }
        addView(zoomLabel)
        zoomLabel.elevation = 1000f
        
        // Add camera switch button
        cameraButton = TextView(context).apply {
            layoutParams = LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.TOP or Gravity.END
                topMargin = 100
                rightMargin = 50
            }
            text = "📷 CAMERA"
            textSize = 24f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(200, 0, 100, 200))
            setPadding(20, 15, 20, 15)
            visibility = View.GONE
            
            setOnClickListener {
                switchCamera()
            }
        }
        addView(cameraButton)
        cameraButton.elevation = 1000f
        
        // Add confidence threshold label
        confidenceLabel = TextView(context).apply {
            layoutParams = LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.BOTTOM or Gravity.START
                bottomMargin = 100
                leftMargin = 50
            }
            text = "Confidence: 0.50"
            textSize = 20f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(200, 200, 100, 0))
            setPadding(15, 10, 15, 10)
            visibility = View.GONE
        }
        addView(confidenceLabel)
        confidenceLabel.elevation = 1000f
        
        // Initialize scale gesture detector for pinch-to-zoom
        scaleGestureDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                val scale = detector.scaleFactor
                val newZoomRatio = currentZoomRatio * scale
                
                // Clamp zoom ratio between min and max
                val clampedZoomRatio = newZoomRatio.coerceIn(minZoomRatio, camera?.cameraInfo?.zoomState?.value?.maxZoomRatio ?: maxZoomRatio)
                
                camera?.cameraControl?.setZoomRatio(clampedZoomRatio)
                currentZoomRatio = clampedZoomRatio
                
                // Notify zoom change
                onZoomChanged?.invoke(currentZoomRatio)
                
                return true
            }
        })
    }

    // region threshold setters

    fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf
        predictor?.setConfidenceThreshold(conf)
        // Update the confidence label if UI controls are shown
        if (showUIControls) {
            post {
                confidenceLabel.text = "Confidence: ${String.format("%.2f", conf)}"
            }
        }
    }

    fun setIouThreshold(iou: Double) {
        iouThreshold = iou
        predictor?.setIouThreshold(iou)
    }

    fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        predictor?.setNumItemsThreshold(n)
    }
    
    fun setShowOverlays(show: Boolean) {
        showOverlays = show
    }

    fun setRtmpEnabled(enabled: Boolean) { rtmpEnabled = enabled }
    fun setRtmpFrameCallback(cb: ((ByteArray, Int, Int) -> Unit)?) { rtmpFrameCallback = cb }

    fun startNativeRecording(): Map<String, Any> {
        stopNativeRecordingSync()
        val vc = videoCapture ?: run {
            Log.e(TAG, "startNativeRecording: videoCapture not ready")
            return mapOf("success" to false, "filePath" to "")
        }
        val dir = context.getExternalFilesDir(null) ?: run {
            Log.e(TAG, "startNativeRecording: no external storage")
            return mapOf("success" to false, "filePath" to "")
        }
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val filePath = "${dir.absolutePath}/gymcam_$ts.mp4"
        nativeRecordingFilePath = filePath

        val latch = CountDownLatch(1)
        recordingFinalizeLatch = latch

        val outputOptions = FileOutputOptions.Builder(File(filePath)).build()
        val hasAudio = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                android.content.pm.PackageManager.PERMISSION_GRANTED
        var pending = vc.output.prepareRecording(context, outputOptions)
        if (hasAudio) pending = pending.withAudioEnabled()
        else Log.w(TAG, "RECORD_AUDIO not granted — recording without audio")
        recordingStopRequested = false
        activeRecording = pending
            .start(Executors.newSingleThreadExecutor()) { event: VideoRecordEvent ->
                if (event is VideoRecordEvent.Finalize) {
                    val expected = recordingStopRequested
                    if (event.hasError()) {
                        Log.e(TAG, "Recording finalize error: code=${event.error} expected=$expected")
                    }
                    if (!expected) {
                        // Camera session was interrupted (lifecycle, model reload, encoder error).
                        // Notify Dart so the record button resets.
                        mainHandler.post { onRecordingStoppedUnexpectedly?.invoke() }
                    }
                    latch.countDown()
                }
            }
        Log.i(TAG, "startNativeRecording → $filePath")
        return mapOf("success" to true, "filePath" to filePath)
    }

    // Blocking stop — must be called from a background thread.
    fun stopNativeRecordingSync(): String? {
        recordingStopRequested = true  // tell Finalize callback this was intentional
        activeRecording?.stop()
        activeRecording = null
        recordingFinalizeLatch?.await(10, TimeUnit.SECONDS)
        recordingFinalizeLatch = null
        val path = nativeRecordingFilePath
        nativeRecordingFilePath = null
        Log.i(TAG, "stopNativeRecording → $path")
        return path
    }

    fun setUdpTarget(host: String, port: Int, quality: Int = 40) {
        udpTarget = InetSocketAddress(host, port)
        if (!udpPipelineRunning) startUdpPipeline()
    }

    fun clearUdpTarget() {
        udpTarget = null
        stopUdpPipeline()
    }

    private fun startUdpPipeline() {
        udpPipelineRunning = true
        udpSocket = DatagramSocket()
        // H.264 encoder is started lazily on first frame (dimensions needed)
    }

    private fun startInferenceThread() {
        inferenceThreadRunning = true
        inferenceThread = Thread {
            while (inferenceThreadRunning) {
                val frame = inferenceQueue.poll(200, TimeUnit.MILLISECONDS) ?: continue
                val (nv21, w, h) = frame
                val p = predictor
                if (p == null || isStopped) continue
                try {
                    // Bitmap decode happens here, off the camera thread
                    val t0 = System.currentTimeMillis()
                    val bitmap = ImageUtils.toBitmapFromNv21(nv21, w, h) ?: continue
                    val decodeMs = System.currentTimeMillis() - t0

                    val isLandscape = context.resources.configuration.orientation ==
                        android.content.res.Configuration.ORIENTATION_LANDSCAPE
                    val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
                    (p as? BasePredictor)?.isFrontCamera = isFrontCamera

                    val t1 = System.currentTimeMillis()
                    val result = if (isLandscape) {
                        p.predict(bitmap, w, h, rotateForCamera = true, isLandscape = isLandscape)
                    } else {
                        p.predict(bitmap, h, w, rotateForCamera = true, isLandscape = isLandscape)
                    }
                    val yoloMs = System.currentTimeMillis() - t1
                    Log.d(TAG, "inference: decode=${decodeMs}ms yolo=${yoloMs}ms total=${decodeMs+yoloMs}ms")

                    val resultWithOriginalImage = if (streamConfig?.includeOriginalImage == true) {
                        result.copy(originalImage = bitmap)
                    } else {
                        result
                    }

                    inferenceResult = resultWithOriginalImage
                    inferenceCallback?.invoke(resultWithOriginalImage)

                    streamCallback?.let { callback ->
                        if (shouldProcessFrame()) {
                            updateLastInferenceTime()
                            val streamData = convertResultToStreamData(resultWithOriginalImage)
                            val enhancedStreamData = HashMap<String, Any>(streamData)
                            enhancedStreamData["timestamp"] = System.currentTimeMillis()
                            enhancedStreamData["frameNumber"] = frameNumberCounter.getAndIncrement()
                            callback.invoke(enhancedStreamData)
                        }
                    }

                    post { overlayView.invalidate() }
                } catch (e: Exception) {
                    Log.e(TAG, "Error during inference", e)
                }
            }
        }.also { it.isDaemon = true; it.name = "yolo-inference"; it.start() }
    }

    private fun stopInferenceThread() {
        inferenceThreadRunning = false
        inferenceThread?.join(500)
        inferenceThread = null
        inferenceQueue.clear()
    }

    private fun stopUdpPipeline() {
        udpPipelineRunning = false
        stopH264Encoder()
        udpSocket?.close()
        udpSocket = null
    }

    private fun startH264Encoder(w: Int, h: Int) {
        try {
            val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            val fmt = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h).apply {
                setInteger(MediaFormat.KEY_BIT_RATE, 4_000_000)
                setInteger(MediaFormat.KEY_FRAME_RATE, 30)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 0)
                setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar)
                setInteger(MediaFormat.KEY_PRIORITY, 0)
            }
            // Sync mode (no setCallback) so dequeueInputBuffer() works in feedH264Frame()
            codec.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            mediaCodec = codec
            Log.i(TAG, "H264 hardware encoder started ${w}x${h}")

            // Drain output on a dedicated thread
            h264OutputRunning = true
            h264OutputThread = Thread {
                val info = MediaCodec.BufferInfo()
                while (h264OutputRunning) {
                    val mc = mediaCodec ?: break
                    val idx = try { mc.dequeueOutputBuffer(info, 10_000L) }
                              catch (_: Exception) { break }
                    when {
                        idx >= 0 -> {
                            val buf = mc.getOutputBuffer(idx)
                            if (buf != null && info.size > 0) {
                                val data = ByteArray(info.size)
                                buf.get(data)
                                val isConfig   = (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0
                                val isKeyFrame = (info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0
                                when {
                                    isConfig -> {
                                        // Store SPS+PPS; don't send separately — will be
                                        // prepended to every IDR so late-joining receivers work
                                        codecConfigData = data
                                        Log.i(TAG, "H264 codec config captured ${data.size}B")
                                    }
                                    isKeyFrame -> {
                                        val config = codecConfigData
                                        if (config == null) Log.w(TAG, "IDR frame with no SPS/PPS — stream will not decode")
                                        sendH264Nal(if (config != null) config + data else data)
                                    }
                                    else -> sendH264Nal(data)
                                }
                            }
                            mc.releaseOutputBuffer(idx, false)
                        }
                        idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val fmt = mc.outputFormat
                            Log.i(TAG, "H264 format: $fmt")
                            // Extract SPS/PPS from csd-0 + csd-1 (device may never emit
                            // BUFFER_FLAG_CODEC_CONFIG, so this is the reliable path)
                            try {
                                val sps = fmt.getByteBuffer("csd-0")
                                val pps = fmt.getByteBuffer("csd-1")
                                if (sps != null && pps != null) {
                                    val spsBytes = ByteArray(sps.remaining()).also { sps.get(it) }
                                    val ppsBytes = ByteArray(pps.remaining()).also { pps.get(it) }
                                    codecConfigData = spsBytes + ppsBytes
                                    Log.i(TAG, "H264 SPS+PPS from csd: ${codecConfigData!!.size}B")
                                }
                            } catch (e: Exception) {
                                Log.w(TAG, "Could not extract csd: $e")
                            }
                        }
                    }
                }
            }.also { it.isDaemon = true; it.name = "h264-output"; it.start() }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start H264 encoder", e)
        }
    }

    private fun stopH264Encoder() {
        h264OutputRunning = false
        h264OutputThread?.join(500)
        h264OutputThread = null
        mediaCodec?.let { mc ->
            try { mc.flush(); mc.stop(); mc.release() } catch (_: Exception) {}
        }
        mediaCodec = null
        codecConfigData = null
    }

    private fun feedH264Frame(nv21: ByteArray, w: Int, h: Int) {
        val mc = mediaCodec ?: run {
            if (udpTarget != null && udpPipelineRunning) startH264Encoder(w, h)
            return
        }
        val idx = mc.dequeueInputBuffer(0)
        if (idx < 0) {
            val dropped = h264DroppedFrames.incrementAndGet()
            val submitted = h264SubmittedFrames.get()
            if (dropped % 30 == 0) {
                Log.w(TAG, "H264 encoder busy: dropped=$dropped submitted=$submitted (${100*dropped/(dropped+submitted+1)}% drop rate)")
            }
            return
        }
        val buf = mc.getInputBuffer(idx) ?: run { mc.queueInputBuffer(idx, 0, 0, 0, 0); return }
        buf.clear()
        // NV21 → NV12: Y plane is identical; UV plane swaps each V,U pair to U,V
        val ySize = w * h
        val nv12 = nv12ScratchBuffer?.takeIf { it.size == nv21.size }
            ?: ByteArray(nv21.size).also { nv12ScratchBuffer = it }
        System.arraycopy(nv21, 0, nv12, 0, ySize)
        var i = ySize
        while (i < nv21.size - 1) {
            nv12[i]     = nv21[i + 1]  // U
            nv12[i + 1] = nv21[i]      // V
            i += 2
        }
        buf.put(nv12)
        mc.queueInputBuffer(idx, 0, nv21.size, System.nanoTime() / 1000, 0)
        h264SubmittedFrames.incrementAndGet()
    }

    private fun sendH264Nal(nal: ByteArray) {
        val sock = udpSocket ?: return
        val target = udpTarget ?: return
        val totalChunks = (nal.size + UDP_CHUNK_SIZE - 1) / UDP_CHUNK_SIZE
        val seq = udpFrameSeq.getAndIncrement()
        val header = ByteArray(8)
        header[0] = (seq shr 24).toByte(); header[1] = (seq shr 16).toByte()
        header[2] = (seq shr 8).toByte();  header[3] = seq.toByte()
        for (i in 0 until totalChunks) {
            val start = i * UDP_CHUNK_SIZE
            val end = minOf(start + UDP_CHUNK_SIZE, nal.size)
            val chunk = ByteArray(8 + (end - start))
            System.arraycopy(header, 0, chunk, 0, 4)
            chunk[4] = (i shr 8).toByte(); chunk[5] = i.toByte()
            chunk[6] = (totalChunks shr 8).toByte(); chunk[7] = totalChunks.toByte()
            System.arraycopy(nal, start, chunk, 8, end - start)
            try { sock.send(DatagramPacket(chunk, chunk.size, target)) } catch (_: Exception) { return }
        }
    }

    fun setShowUIControls(show: Boolean) {
        showUIControls = show
        // Show/hide all UI controls
        val visibility = if (show) View.VISIBLE else View.GONE
        zoomLabel.visibility = visibility
        cameraButton.visibility = visibility
        confidenceLabel.visibility = visibility
    }
    
    fun setZoomLevel(zoomLevel: Float) {
        camera?.let { cam: Camera ->
            // Clamp zoom level between min and max
            val clampedZoomRatio = zoomLevel.coerceIn(minZoomRatio, cam.cameraInfo.zoomState.value?.maxZoomRatio ?: maxZoomRatio)
            
            cam.cameraControl.setZoomRatio(clampedZoomRatio)
            currentZoomRatio = clampedZoomRatio
            
            // Notify zoom change
            onZoomChanged?.invoke(currentZoomRatio)
        }
    }

    fun setTorchMode(enabled: Boolean) {
        camera?.let { cam ->
            if (cam.cameraInfo.hasFlashUnit()) {
                cam.cameraControl.enableTorch(enabled)
            }
        }
    }

    // endregion

    // region Model / Task

    fun setModel(modelPath: String, task: YOLOTask, useGpu: Boolean = true, callback: ((Boolean) -> Unit)? = null) {
        Executors.newSingleThreadExecutor().execute {
            try {
                val newPredictor = when (task) {
                    YOLOTask.DETECT -> ObjectDetector(context = context, modelPath = modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.SEGMENT -> Segmenter(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.CLASSIFY -> Classifier(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.POSE -> PoseEstimator(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.OBB -> ObbDetector(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                }
                
                // Apply thresholds to all predictor types
                newPredictor.apply {
                    setConfidenceThreshold(confidenceThreshold)
                    setIouThreshold(iouThreshold)
                    setNumItemsThreshold(numItemsThreshold)
                }

                post {
                    this.task = task
                    this.predictor = newPredictor
                    this.modelName = modelPath.substringAfterLast("/")
                    modelLoadCallback?.invoke(true)
                    callback?.invoke(true)
                    // Ensure camera starts after model loads if it's not already running
                    if (allPermissionsGranted() && lifecycleOwner != null && (camera == null || isStopped)) {
                        startCamera()
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load model: $modelPath. Camera will run without inference.", e)
                post {
                    // Clear predictor so the camera can keep running until a valid model is set
                    this.predictor = null
                    this.modelName = "No Model"
                    modelLoadCallback?.invoke(false)
                    callback?.invoke(false)
                }
            }
        }
    }

    private fun loadLabels(modelPath: String): List<String> {
        // Try to load labels from model metadata first
        val loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        if (loadedLabels != null) {
            return loadedLabels
        }

        // Return COCO dataset's 80 classes as a fallback
        // This is much more complete than the previous 7-class hardcoded list
        return listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        )
    }

    // endregion

    /**
     * Called when a LifecycleOwner is available for camera operations
     */
    fun onLifecycleOwnerAvailable(owner: LifecycleOwner) {
        this.lifecycleOwner = owner
        owner.lifecycle.addObserver(this)
        
        if (allPermissionsGranted() && (camera == null || isStopped)) {
            startCamera()
        }
    }
    
    // region camera init

    fun initCamera() {
        if (allPermissionsGranted()) {
            if (lifecycleOwner != null && (camera == null || isStopped)) {
                startCamera()
            }
        } else {
            val activity = context as? Activity ?: return
            ActivityCompat.requestPermissions(
                activity,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(context, "Camera permission not granted.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    fun startCamera() {
        isStopped = false

        try {
            cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()

                    // If recording is active, skip the full camera restart — unbindAll()
                    // would terminate the VideoCapture session (ERROR_SOURCE_INACTIVE).
                    // CameraX keeps the camera alive through lifecycle transitions automatically.
                    if (activeRecording != null) {
                        Log.w(TAG, "startCamera: recording in progress — skipping camera restart")
                        return@addListener
                    }

                    previewUseCase = Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                        .build()

                    imageAnalysisUseCase = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setResolutionSelector(
                            androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                                .setAspectRatioStrategy(
                                    androidx.camera.core.resolutionselector.AspectRatioStrategy(
                                        AspectRatio.RATIO_16_9,
                                        androidx.camera.core.resolutionselector.AspectRatioStrategy.FALLBACK_RULE_AUTO
                                    )
                                )
                                .setResolutionStrategy(
                                    androidx.camera.core.resolutionselector.ResolutionStrategy(
                                        android.util.Size(1920, 1080),
                                        androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER
                                    )
                                )
                                .build()
                        )
                        .build()

                    val recorder = Recorder.Builder()
                        .setQualitySelector(QualitySelector.fromOrderedList(
                            listOf(Quality.FHD, Quality.HD, Quality.SD),
                            FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)
                        ))
                        .build()
                    val vc = VideoCapture.withOutput(recorder)
                    videoCapture = vc

                    cameraExecutor = Executors.newSingleThreadExecutor()
                    imageAnalysisUseCase!!.setAnalyzer(cameraExecutor!!) { imageProxy ->
                        onFrame(imageProxy)
                    }
                    startInferenceThread()

                    val cameraSelector = CameraSelector.Builder()
                        .requireLensFacing(lensFacing)
                        .build()

                    cameraProvider.unbindAll()

                    try {
                        val owner = lifecycleOwner
                        if (owner == null) {
                            Log.e(TAG, "No LifecycleOwner available. Call onLifecycleOwnerAvailable() first.")
                            return@addListener
                        }

                        camera = cameraProvider.bindToLifecycle(
                            owner,
                            cameraSelector,
                            previewUseCase,
                            imageAnalysisUseCase,
                            vc
                        )

                        // Reset zoom to 1.0x when camera starts
                        currentZoomRatio = 1.0f
                        onZoomChanged?.invoke(currentZoomRatio)

                        previewUseCase?.setSurfaceProvider(previewView.surfaceProvider)

                        // Initialize zoom
                        camera?.let { cam: Camera ->
                            val cameraInfo = cam.cameraInfo
                            minZoomRatio = cameraInfo.zoomState.value?.minZoomRatio ?: 1.0f
                            maxZoomRatio = cameraInfo.zoomState.value?.maxZoomRatio ?: 1.0f
                            currentZoomRatio = cameraInfo.zoomState.value?.zoomRatio ?: 1.0f
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Use case binding failed", e)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error getting camera provider", e)
                }
            }, ContextCompat.getMainExecutor(context))
        } catch (e: Exception) {
            Log.e(TAG, "Error starting camera", e)
        }
    }

    fun setLensFacing(facing: Int) {
        lensFacing = facing
        // Restart camera if already started
        if (::cameraProviderFuture.isInitialized) {
            startCamera()
        }
    }

    fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    // endregion
    
    // Lifecycle methods from DefaultLifecycleObserver
    override fun onStart(owner: LifecycleOwner) {
        if (allPermissionsGranted()) {
            // Always restart camera on start if it's stopped or null
            // This ensures camera resumes when navigating back
            if (isStopped || camera == null) {
                startCamera()
            }
        }
    }

    override fun onResume(owner: LifecycleOwner) {
        if (allPermissionsGranted()) {
            // Double-check camera is running on resume
            if (isStopped || camera == null) {
                startCamera()
            }
        }
    }

    override fun onStop(owner: LifecycleOwner) {
        // Camera will be automatically stopped by CameraX when lifecycle stops
    }

    // region onFrame (per frame inference)

    private fun onFrame(imageProxy: ImageProxy) {
        if (isStopped) { imageProxy.close(); return }

        val frameStart = System.currentTimeMillis()
        val w = imageProxy.width
        val h = imageProxy.height
        lastFrameWidth = w
        lastFrameHeight = h
        val frameNum = cameraFrameCounter.incrementAndGet()
        if (frameNum % 30 == 0) {
            val now = System.currentTimeMillis()
            val elapsed = now - cameraTickMs
            val cameraFps = if (cameraTickMs > 0) (30_000.0 / elapsed).toInt() else 0
            cameraTickMs = now
            Log.i(TAG, "Camera fps≈$cameraFps  encoder submitted=${h264SubmittedFrames.get()} dropped=${h264DroppedFrames.get()}")
        }

        val needNv21 = rtmpEnabled || udpTarget != null || predictor != null
        val t0 = System.currentTimeMillis()
        val nv21: ByteArray? = if (needNv21) {
            try { ImageUtils.yuv420888ToNv21(imageProxy) } catch (_: Exception) { null }
        } else null
        val nv21Ms = System.currentTimeMillis() - t0

        // UDP path: H.264 hardware encoder — non-blocking, drops if encoder busy
        val t1 = System.currentTimeMillis()
        if (nv21 != null && udpTarget != null) {
            feedH264Frame(nv21, w, h)
        }
        val udpMs = System.currentTimeMillis() - t1

        // RTMP path: 33ms gate to avoid overwhelming FFmpeg
        if (nv21 != null && rtmpEnabled) {
            val nowMs = System.currentTimeMillis()
            if (nowMs - lastRtmpMs >= 33) {
                lastRtmpMs = nowMs
                val cb = rtmpFrameCallback
                if (cb != null) mainHandler.post { cb(nv21, w, h) }
            }
        }

        // Submit raw NV21 to inference thread — bitmap is created inside the thread,
        // not here, so the camera thread is not blocked by JPEG encode+decode.
        val t2 = System.currentTimeMillis()
        var inferencePushed = false
        if (!isStopped && predictor != null && shouldRunInference() && nv21 != null) {
            inferencePushed = inferenceQueue.offer(Triple(nv21, w, h))
        }
        val inferEnqMs = System.currentTimeMillis() - t2

        imageProxy.close()

        val totalMs = System.currentTimeMillis() - frameStart
        Log.d(TAG, "onFrame ${w}x${h}: nv21=${nv21Ms}ms h264Feed=${udpMs}ms inferEnq=${inferEnqMs}ms total=${totalMs}ms yoloPushed=$inferencePushed")
    }

    // endregion

    // region OverlayView

    private inner class OverlayView(context: Context) : View(context) {
        private val paint = Paint().apply { isAntiAlias = true }

        init {
            // Make background transparent
            setBackgroundColor(Color.TRANSPARENT)
            // Use hardware layer for better z-order 
            setLayerType(LAYER_TYPE_HARDWARE, null)

            // Raise overlay
            elevation = 1000f
            translationZ = 1000f

            setWillNotDraw(false)

            // Make overlay not intercept touch events
            isClickable = false
            isFocusable = false
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val result = inferenceResult ?: return
            
            // Only draw overlays if showOverlays is true
            if (!showOverlays) {
                return
            }

            val iw = result.origShape.width.toFloat()
            val ih = result.origShape.height.toFloat()

            val vw = width.toFloat()
            val vh = height.toFloat()
            
            // Get device orientation for debugging
            val orientation = context.resources.configuration.orientation
            val isLandscape = orientation == Configuration.ORIENTATION_LANDSCAPE
            

            // Scale factor from camera image to view
            val scaleX = vw / iw
            val scaleY = vh / ih
            val scale = max(scaleX, scaleY)
            

            val scaledW = iw * scale
            val scaledH = ih * scale

            val dx = (vw - scaledW) / 2f
            val dy = (vh - scaledH) / 2f
            
            // Check if using front camera
            val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
            

            when (task) {
                // ----------------------------------------
                // DETECT
                // ----------------------------------------
                YOLOTask.DETECT -> {
                    for (box in result.boxes) {
                        val alpha = (box.conf * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[box.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        // Use same coordinate calculation for all orientations
                        // since the image is now correctly oriented before inference
                        var left = box.xywh.left * scale + dx
                        var top = box.xywh.top * scale + dy
                        var right = box.xywh.right * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // Ensure coordinates are within view bounds and maintain aspect ratio
                        val boxWidth = right - left
                        val boxHeight = bottom - top
                        
                        // Adjust coordinates to maintain aspect ratio and stay within bounds
                        if (left < 0) {
                            left = 0f
                            right = left + boxWidth
                        }
                        if (right > vw) {
                            right = vw.toFloat()
                            left = right - boxWidth
                        }
                        if (top < 0) {
                            top = 0f
                            bottom = top + boxHeight
                        }
                        if (bottom > vh) {
                            bottom = vh.toFloat()
                            top = bottom - boxHeight
                        }
                        
                        // Flip horizontally for front camera (DETECT task)
                        if (isFrontCamera) {
                            val flippedLeft = vw - right
                            val flippedRight = vw - left
                            left = flippedLeft
                            right = flippedRight
                        }

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )

                        // Label text
                        val labelText = "${box.cls} ${"%.1f".format(box.conf * 100)}%"
                        paint.textSize = 40f
                        val fm = paint.fontMetrics
                        val textWidth = paint.measureText(labelText)
                        val textHeight = fm.bottom - fm.top
                        val pad = 8f

                        // Label background height is (text height + 2*padding)
                        val labelBoxHeight = textHeight + 2 * pad
                        // Place label on top of the box's upper edge
                        var labelBottom = top
                        var labelTop = labelBottom - labelBoxHeight

                        // Ensure label stays within bounds
                        if (labelTop < 0) {
                            labelTop = top
                            labelBottom = labelTop + labelBoxHeight
                        }

                        // Rectangle for label background
                        val labelLeft = left
                        val labelRight = left + textWidth + 2 * pad
                        val bgRect = RectF(labelLeft, labelTop, labelRight, labelBottom)

                        // Draw background
                        paint.style = Paint.Style.FILL
                        paint.color = newColor
                        canvas.drawRoundRect(bgRect, BOX_CORNER_RADIUS, BOX_CORNER_RADIUS, paint)

                        // Center text vertically within the rectangle
                        paint.color = Color.WHITE
                        // Center position = (bgRect.top + bgRect.bottom)/2
                        val centerY = (bgRect.top + bgRect.bottom) / 2
                        // Baseline = centerY - (fm.descent + fm.ascent)/2
                        val baseline = centerY - (fm.descent + fm.ascent) / 2
                        canvas.drawText(labelText, bgRect.left + pad, baseline, paint)
                    }
                }
                // ----------------------------------------
                // SEGMENT
                // ----------------------------------------
                YOLOTask.SEGMENT -> {
                    // Bounding boxes & labels
                    for (box in result.boxes) {
                        val alpha = (box.conf * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[box.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        // Draw bounding box
                        var left   = box.xywh.left   * scale + dx
                        var top    = box.xywh.top    * scale + dy
                        var right  = box.xywh.right  * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // For front camera POSE, apply horizontal flip
                        if (isFrontCamera) {
                            // Flip horizontally
                            val flippedLeft = vw - right
                            val flippedRight = vw - left
                            left = flippedLeft
                            right = flippedRight
                        }

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )

                        // Label background + text (vertically centered)
                        val labelText = "${box.cls} ${"%.1f".format(box.conf * 100)}%"
                        paint.textSize = 40f
                        val fm = paint.fontMetrics
                        val textWidth = paint.measureText(labelText)
                        val textHeight = fm.bottom - fm.top
                        val pad = 8f

                        val labelBoxHeight = textHeight + 2 * pad
                        val labelBoxWidth = textWidth + 2 * pad
                        
                        // Calculate initial label position (above the box)
                        var labelLeft = left
                        var labelTop = top - labelBoxHeight
                        var labelRight = labelLeft + labelBoxWidth
                        var labelBottom = top
                        
                        // Check top boundary
                        if (labelTop < 0) {
                            // Place label inside the top of the box
                            labelTop = top
                            labelBottom = labelTop + labelBoxHeight
                        }
                        
                        // Check left boundary
                        if (labelLeft < 0) {
                            labelLeft = 0f
                            labelRight = labelBoxWidth
                        }
                        
                        // Check right boundary
                        if (labelRight > vw) {
                            labelRight = vw.toFloat()
                            labelLeft = labelRight - labelBoxWidth
                            // If label is still too wide, align it with the right edge of the box
                            if (labelLeft < 0) {
                                labelLeft = maxOf(0f, right - labelBoxWidth)
                            }
                        }
                        
                        // Check bottom boundary (in case label was moved inside the box)
                        if (labelBottom > vh) {
                            labelBottom = vh.toFloat()
                            labelTop = labelBottom - labelBoxHeight
                        }
                        
                        val bgRect = RectF(labelLeft, labelTop, labelRight, labelBottom)

                        paint.style = Paint.Style.FILL
                        paint.color = newColor
                        canvas.drawRoundRect(bgRect, BOX_CORNER_RADIUS, BOX_CORNER_RADIUS, paint)

                        paint.color = Color.WHITE
                        val centerY = (labelTop + labelBottom) / 2
                        val baseline = centerY - (fm.descent + fm.ascent) / 2
                        canvas.drawText(labelText, labelLeft + pad, baseline, paint)
                    }

                    // Segmentation mask
                    result.masks?.combinedMask?.let { maskBitmap ->
                        val src = GraphicsRect(0, 0, maskBitmap.width, maskBitmap.height)
                        val dst = RectF(dx, dy, dx + scaledW, dy + scaledH)
                        val maskPaint = Paint().apply { alpha = 128 }
                        
                        if (isFrontCamera) {
                            // For front camera, flip the mask horizontally
                            canvas.save()
                            // Translate to center, flip horizontally, translate back
                            canvas.translate(vw / 2f, 0f)
                            canvas.scale(-1f, 1f)
                            canvas.translate(-vw / 2f, 0f)
                            canvas.drawBitmap(maskBitmap, src, dst, maskPaint)
                            canvas.restore()
                        } else {
                            canvas.drawBitmap(maskBitmap, src, dst, maskPaint)
                        }
                    }
                }
                // ----------------------------------------
                // CLASSIFY (display large in center)
                // ----------------------------------------
                YOLOTask.CLASSIFY -> {
                    result.probs?.let { probs ->
                        val alpha = (probs.top1Conf * 255).toInt().coerceIn(0, 255)
                        // Select color based on top1Index
                        val baseColor = ultralyticsColors[probs.top1Index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        val labelText = "${probs.top1Label} ${"%.1f".format(probs.top1Conf * 100)}%"
                        paint.textSize = 60f
                        val textWidth = paint.measureText(labelText)
                        val fm = paint.fontMetrics
                        val textHeight = fm.bottom - fm.top
                        val pad = 16f

                        // Screen center
                        val centerX = vw / 2f
                        val centerY = vh / 2f

                        val bgLeft   = centerX - (textWidth / 2) - pad
                        val bgTop    = centerY - (textHeight / 2) - pad
                        val bgRight  = centerX + (textWidth / 2) + pad
                        val bgBottom = centerY + (textHeight / 2) + pad

                        paint.color = newColor
                        paint.style = Paint.Style.FILL
                        val bgRect = RectF(bgLeft, bgTop, bgRight, bgBottom)
                        canvas.drawRoundRect(bgRect, 20f, 20f, paint)

                        paint.color = Color.WHITE
                        val baseline = centerY - (fm.descent + fm.ascent)/2
                        canvas.drawText(labelText, centerX - (textWidth / 2), baseline, paint)
                    }
                }
                // ----------------------------------------
                // POSE
                // ----------------------------------------
                YOLOTask.POSE -> {
                    // Bounding boxes
                    for (box in result.boxes) {
                        val alpha = (box.conf * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[box.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        var left   = box.xywh.left   * scale + dx
                        var top    = box.xywh.top    * scale + dy
                        var right  = box.xywh.right  * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // For front camera POSE, apply horizontal flip
                        if (isFrontCamera) {
                            // Flip horizontally
                            val flippedLeft = vw - right
                            val flippedRight = vw - left
                            left = flippedLeft
                            right = flippedRight
                        }

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )
                        
                        // Add label
                        val labelText = "${box.cls} ${"%.1f".format(box.conf * 100)}%"
                        paint.textSize = 40f
                        val fm = paint.fontMetrics
                        val textWidth = paint.measureText(labelText)
                        val textHeight = fm.bottom - fm.top
                        val pad = 8f
                        
                        val labelBoxHeight = textHeight + 2 * pad
                        val labelBoxWidth = textWidth + 2 * pad
                        
                        // Calculate initial label position (above the box)
                        var labelLeft = left
                        var labelTop = top - labelBoxHeight
                        var labelRight = labelLeft + labelBoxWidth
                        var labelBottom = top
                        
                        // Check top boundary
                        if (labelTop < 0) {
                            // Place label inside the top of the box
                            labelTop = top
                            labelBottom = labelTop + labelBoxHeight
                        }
                        
                        // Check left boundary
                        if (labelLeft < 0) {
                            labelLeft = 0f
                            labelRight = labelBoxWidth
                        }
                        
                        // Check right boundary
                        if (labelRight > vw) {
                            labelRight = vw.toFloat()
                            labelLeft = labelRight - labelBoxWidth
                            // If label is still too wide, align it with the right edge of the box
                            if (labelLeft < 0) {
                                labelLeft = maxOf(0f, right - labelBoxWidth)
                            }
                        }
                        
                        // Check bottom boundary
                        if (labelBottom > vh) {
                            labelBottom = vh.toFloat()
                            labelTop = labelBottom - labelBoxHeight
                        }
                        
                        val bgRect = RectF(labelLeft, labelTop, labelRight, labelBottom)
                        
                        // Draw label background
                        paint.style = Paint.Style.FILL
                        paint.color = newColor
                        canvas.drawRoundRect(bgRect, BOX_CORNER_RADIUS, BOX_CORNER_RADIUS, paint)
                        
                        // Draw label text
                        paint.color = Color.WHITE
                        val centerY = (labelTop + labelBottom) / 2
                        val baseline = centerY - (fm.descent + fm.ascent) / 2
                        canvas.drawText(labelText, labelLeft + pad, baseline, paint)
                    }

                    // Keypoints & skeleton
                    for (person in result.keypointsList) {
                        val points = arrayOfNulls<PointF>(person.xyn.size)
                        for (i in person.xyn.indices) {
                            val kp = person.xyn[i]
                            val conf = person.conf[i]
                            if (conf > 0.25f) {
                                val pxCam = kp.first * iw
                                val pyCam = kp.second * ih
                                var px = pxCam * scale + dx
                                var py = pyCam * scale + dy
                                
                                // For front camera POSE, apply horizontal flip
                                if (isFrontCamera) {
                                    px = vw - px  // Flip horizontally
                                }

                                val colorIdx = if (i < kptColorIndices.size) kptColorIndices[i] else 0
                                val rgbArray = posePalette[colorIdx % posePalette.size]
                                paint.color = Color.argb(
                                    255,
                                    rgbArray[0].toInt().coerceIn(0,255),
                                    rgbArray[1].toInt().coerceIn(0,255),
                                    rgbArray[2].toInt().coerceIn(0,255)
                                )
                                paint.style = Paint.Style.FILL
                                canvas.drawCircle(px, py, 8f, paint)

                                points[i] = PointF(px, py)
                            }
                        }

                        // Skeleton connection
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = KEYPOINT_LINE_WIDTH
                        for ((idx, bone) in skeleton.withIndex()) {
                            val i1 = bone[0] - 1  // 1-indexed to 0-indexed
                            val i2 = bone[1] - 1
                            val p1 = points.getOrNull(i1)
                            val p2 = points.getOrNull(i2)
                            if (p1 != null && p2 != null) {
                                val limbColorIdx = if (idx < limbColorIndices.size) limbColorIndices[idx] else 0
                                val rgbArray = posePalette[limbColorIdx % posePalette.size]
                                paint.color = Color.argb(
                                    255,
                                    rgbArray[0].toInt().coerceIn(0,255),
                                    rgbArray[1].toInt().coerceIn(0,255),
                                    rgbArray[2].toInt().coerceIn(0,255)
                                )
                                canvas.drawLine(p1.x, p1.y, p2.x, p2.y, paint)
                            }
                        }
                    }
                }
                // ----------------------------------------
                // OBB
                // ----------------------------------------
                YOLOTask.OBB -> {
                    for (obbRes in result.obb) {
                        val alpha = (obbRes.confidence * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[obbRes.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH

                        // Draw rotated rectangle (polygon) using path
                        val polygon = obbRes.box.toPolygon().map { pt ->
                            var x = pt.x * scaledW + dx
                            val y = pt.y * scaledH + dy
                            
                            // Flip horizontally for front camera
                            if (isFrontCamera) {
                                x = vw - x
                            }
                            
                            PointF(x, y)
                        }
                        if (polygon.size >= 4) {
                            val path = Path().apply {
                                moveTo(polygon[0].x, polygon[0].y)
                                for (p in polygon.drop(1)) {
                                    lineTo(p.x, p.y)
                                }
                                close()
                            }
                            canvas.drawPath(path, paint)

                            // Label text
                            val labelText = "${obbRes.cls} ${"%.1f".format(obbRes.confidence * 100)}%"
                            paint.textSize = 40f
                            paint.typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)

                            val fm = paint.fontMetrics
                            val textWidth = paint.measureText(labelText)
                            val textHeight = fm.bottom - fm.top
                            val padding = 10f
                            val cornerRadius = 8f

                            // Find bounding box of the OBB polygon
                            val minX = polygon.map { it.x }.minOrNull() ?: 0f
                            val maxX = polygon.map { it.x }.maxOrNull() ?: 0f
                            val minY = polygon.map { it.y }.minOrNull() ?: 0f
                            val maxY = polygon.map { it.y }.maxOrNull() ?: 0f
                            
                            val labelBoxHeight = textHeight + 2 * padding
                            val labelBoxWidth = textWidth + 2 * padding
                            
                            // Calculate initial label position (above the OBB)
                            var labelLeft = minX
                            var labelTop = minY - labelBoxHeight
                            var labelRight = labelLeft + labelBoxWidth
                            var labelBottom = minY
                            
                            // Check top boundary
                            if (labelTop < 0) {
                                // Place label inside the top of the OBB
                                labelTop = minY
                                labelBottom = labelTop + labelBoxHeight
                            }
                            
                            // Check left boundary
                            if (labelLeft < 0) {
                                labelLeft = 0f
                                labelRight = labelBoxWidth
                            }
                            
                            // Check right boundary
                            if (labelRight > vw) {
                                labelRight = vw.toFloat()
                                labelLeft = labelRight - labelBoxWidth
                                // If label is still too wide, align it with the OBB's right edge
                                if (labelLeft < 0) {
                                    labelLeft = maxOf(0f, maxX - labelBoxWidth)
                                }
                            }
                            
                            // Check bottom boundary
                            if (labelBottom > vh) {
                                labelBottom = vh.toFloat()
                                labelTop = labelBottom - labelBoxHeight
                            }

                            val bgRect = RectF(labelLeft, labelTop, labelRight, labelBottom)
                            paint.style = Paint.Style.FILL
                            paint.color = newColor
                            canvas.drawRoundRect(bgRect, cornerRadius, cornerRadius, paint)

                            // Center text vertically
                            paint.color = Color.WHITE
                            val centerY = (labelTop + labelBottom) / 2
                            val baseline = centerY - (fm.descent + fm.ascent) / 2
                            val textX = labelLeft + padding
                            canvas.drawText(labelText, textX, baseline, paint)
                        }
                    }
                }
            }
        }
        
        override fun onTouchEvent(event: MotionEvent?): Boolean {
            // Pass through all touch events
            return false
        }
    }
    
    // Scale listener for pinch-to-zoom
    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScaleBegin(detector: ScaleGestureDetector): Boolean {
            // Show zoom label when pinch starts (only if UI controls are not permanently shown)
            if (!showUIControls) {
                zoomLabel.visibility = View.VISIBLE
            }
            return true
        }
        
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val scaleFactor = detector.scaleFactor
            val newZoomRatio = currentZoomRatio * scaleFactor
            
            // Clamp zoom within min/max bounds
            val clampedZoom = newZoomRatio.coerceIn(minZoomRatio, maxZoomRatio)
            
            // Apply zoom to camera
            camera?.cameraControl?.setZoomRatio(clampedZoom)
            currentZoomRatio = clampedZoom
            
            // Update zoom label
            zoomLabel.text = String.format("%.1fx", currentZoomRatio)
            
            return true
        }
        
        override fun onScaleEnd(detector: ScaleGestureDetector) {
            // Hide zoom label after 2 seconds (only if UI controls are not permanently shown)
            if (!showUIControls) {
                zoomLabel.postDelayed({
                    zoomLabel.visibility = View.GONE
                }, 2000)
            }
        }
    }
    
    // Touch event handling for pinch-to-zoom
    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleGestureDetector.onTouchEvent(event)
        return true
    }
    
    // region Streaming functionality
    
    /**
     * Setup throttling parameters from streaming configuration
     */
    private fun setupThrottlingFromConfig() {
        streamConfig?.let { config ->
            // Setup maxFPS throttling (for result output)
            config.maxFPS?.let { maxFPS ->
                if (maxFPS > 0) {
                    targetFrameInterval = (1_000_000_000L / maxFPS) // Convert to nanoseconds
                }
            } ?: run {
                targetFrameInterval = null
            }

            // Setup throttleInterval (for result output)
            config.throttleIntervalMs?.let { throttleMs ->
                if (throttleMs > 0) {
                    throttleInterval = throttleMs * 1_000_000L // Convert ms to nanoseconds
                }
            } ?: run {
                throttleInterval = null
            }

            // Setup inference frequency control
            config.inferenceFrequency?.let { inferenceFreq ->
                if (inferenceFreq > 0) {
                    inferenceFrameInterval = (1_000_000_000L / inferenceFreq) // Convert to nanoseconds
                }
            } ?: run {
                inferenceFrameInterval = null
            }

            // Setup frame skipping
            config.skipFrames?.let { skipFrames ->
                if (skipFrames > 0) {
                    targetSkipFrames = skipFrames
                    frameSkipCount = 0 // Reset counter
                }
            } ?: run {
                targetSkipFrames = 0
                frameSkipCount = 0
            }

            // Initialize timing
            lastInferenceTime = System.nanoTime()
        }
    }
    
    /**
     * Check if we should run inference on this frame based on inference frequency control
     */
    private fun shouldRunInference(): Boolean {
        val now = System.nanoTime()
        
        // Check frame skipping control first (simpler, more deterministic)
        if (targetSkipFrames > 0) {
            frameSkipCount++
            if (frameSkipCount <= targetSkipFrames) {
                // Still skipping frames
                return false
            } else {
                // Reset counter and allow inference
                frameSkipCount = 0
                return true
            }
        }
        
        // Check inference frequency control (time-based)
        inferenceFrameInterval?.let { interval ->
            if (now - lastInferenceTime < interval) {
                return false
            }
        }
        
        return true
    }
    
    /**
     * Check if we should send results to Flutter based on output throttling settings
     */
    private fun shouldProcessFrame(): Boolean {
        val now = System.nanoTime()
        
        // Check maxFPS throttling
        targetFrameInterval?.let { interval ->
            if (now - lastInferenceTime < interval) {
                return false
            }
        }
        
        // Check throttleInterval
        throttleInterval?.let { interval ->
            if (now - lastInferenceTime < interval) {
                return false
            }
        }
        
        return true
    }
    
    /**
     * Update the last inference time (call this when actually processing)
     */
    private fun updateLastInferenceTime() {
        lastInferenceTime = System.nanoTime()
    }
    
    /**
     * Flattens keypoints data into a single array format: [x1, y1, conf1, x2, y2, conf2, ...]
     */
    private fun flattenKeypoints(keypoints: Keypoints): List<Double> {
        val flattened = mutableListOf<Double>()
        for (i in keypoints.xy.indices) {
            flattened.add(keypoints.xy[i].first.toDouble())
            flattened.add(keypoints.xy[i].second.toDouble())
            val confidence = if (i < keypoints.conf.size) {
                keypoints.conf[i].toDouble()
            } else {
                0.0
            }
            flattened.add(confidence)
        }
        return flattened
    }

    /**
     * Convert YOLOResult to a Map for streaming (ported from archived YOLOPlatformView)
     * Uses detection index correctly to avoid class index confusion
     */
    private fun convertResultToStreamData(result: YOLOResult): Map<String, Any> {
        val map = HashMap<String, Any>()
        val config = streamConfig ?: return emptyMap()
        
        // Convert detection results (if enabled)
        if (config.includeDetections) {
            val detections = ArrayList<Map<String, Any>>()

            if (config.includePoses && result.keypointsList.isNotEmpty() && result.boxes.isEmpty()) {
                for ((poseIndex, keypoints) in result.keypointsList.withIndex()) {
                    val detection = HashMap<String, Any>()
                    detection["classIndex"] = 0
                    detection["className"] = "person"
                    detection["confidence"] = 1.0
                    var minX = Float.MAX_VALUE
                    var minY = Float.MAX_VALUE
                    var maxX = Float.MIN_VALUE
                    var maxY = Float.MIN_VALUE
                    
                    for (kp in keypoints.xy) {
                        if (kp.first > 0 && kp.second > 0) {
                            minX = minOf(minX, kp.first)
                            minY = minOf(minY, kp.second)
                            maxX = maxOf(maxX, kp.first)
                            maxY = maxOf(maxY, kp.second)
                        }
                    }
                    val boundingBox = HashMap<String, Any>()
                    boundingBox["left"] = minX.toDouble()
                    boundingBox["top"] = minY.toDouble()
                    boundingBox["right"] = maxX.toDouble()
                    boundingBox["bottom"] = maxY.toDouble()
                    detection["boundingBox"] = boundingBox
                    
                    // Normalized bounding box
                    val normalizedBox = HashMap<String, Any>()
                    normalizedBox["left"] = (minX / result.origShape.width).toDouble()
                    normalizedBox["top"] = (minY / result.origShape.height).toDouble()
                    normalizedBox["right"] = (maxX / result.origShape.width).toDouble()
                    normalizedBox["bottom"] = (maxY / result.origShape.height).toDouble()
                    detection["normalizedBox"] = normalizedBox
                    
                    val keypointsFlat = flattenKeypoints(keypoints)
                    detection["keypoints"] = keypointsFlat

                    detections.add(detection)
                }
            }
            
            // Convert detection boxes - CRITICAL: use detectionIndex, not class index
            for ((detectionIndex, box) in result.boxes.withIndex()) {
                val detection = HashMap<String, Any>()
                detection["classIndex"] = box.index
                detection["className"] = box.cls
                detection["confidence"] = box.conf.toDouble()
                
                // Bounding box in original coordinates
                val boundingBox = HashMap<String, Any>()
                boundingBox["left"] = box.xywh.left.toDouble()
                boundingBox["top"] = box.xywh.top.toDouble()
                boundingBox["right"] = box.xywh.right.toDouble()
                boundingBox["bottom"] = box.xywh.bottom.toDouble()
                detection["boundingBox"] = boundingBox
                
                // Normalized bounding box (0-1)
                val normalizedBox = HashMap<String, Any>()
                normalizedBox["left"] = box.xywhn.left.toDouble()
                normalizedBox["top"] = box.xywhn.top.toDouble()
                normalizedBox["right"] = box.xywhn.right.toDouble()
                normalizedBox["bottom"] = box.xywhn.bottom.toDouble()
                detection["normalizedBox"] = normalizedBox
                
                // Add mask data for segmentation (if available and enabled)
                if (config.includeMasks && result.masks != null && detectionIndex < result.masks!!.masks.size) {
                    val maskData = result.masks!!.masks[detectionIndex] // Get mask for this detection
                    // Convert List<List<Float>> to List<List<Double>> for Flutter compatibility
                    val maskDataDouble = maskData.map { row ->
                        row.map { it.toDouble() }
                    }
                    detection["mask"] = maskDataDouble
                }
                
                // Add pose keypoints (if available and enabled)
                if (config.includePoses && result.keypointsList.isNotEmpty()) {
                    if (detectionIndex < result.keypointsList.size) {
                        val keypoints = result.keypointsList[detectionIndex]
                        val keypointsFlat = flattenKeypoints(keypoints)
                        detection["keypoints"] = keypointsFlat
                    }
                }
                
                detections.add(detection)
            }
            
            // Handle OBB results directly (same pattern as overlay: for obbRes in result.obb)
            for (obbRes in result.obb) {
                val detection = HashMap<String, Any>()
                detection["classIndex"] = obbRes.index
                detection["className"] = obbRes.cls
                detection["confidence"] = obbRes.confidence.toDouble()
                
                // Get OBB polygon points (4 corners of rotated rectangle)
                val polygon = obbRes.box.toPolygon()
                val imgWidth = result.origShape.width.toFloat()
                val imgHeight = result.origShape.height.toFloat()
                
                // Convert polygon points to pixel coordinates  
                val polygonPixels = polygon.map { point ->
                    mapOf(
                        "x" to (point.x * imgWidth).toDouble(),
                        "y" to (point.y * imgHeight).toDouble()
                    )
                }
                
                // Store polygon points directly for precise OBB cropping
                detection["polygon"] = polygonPixels
                
                // Also calculate AABB as fallback for compatibility (but Flutter should use polygon)
                var minX = Float.MAX_VALUE
                var maxX = Float.MIN_VALUE  
                var minY = Float.MAX_VALUE
                var maxY = Float.MIN_VALUE
                
                for (point in polygon) {
                    if (point.x < minX) minX = point.x
                    if (point.x > maxX) maxX = point.x
                    if (point.y < minY) minY = point.y
                    if (point.y > maxY) maxY = point.y
                }
                
                // Fallback bounding box (enlarged) - only use if polygon cropping fails
                val boundingBox = HashMap<String, Any>()
                boundingBox["left"] = (minX * imgWidth).toDouble()
                boundingBox["top"] = (minY * imgHeight).toDouble()
                boundingBox["right"] = (maxX * imgWidth).toDouble()
                boundingBox["bottom"] = (maxY * imgHeight).toDouble()
                detection["boundingBox"] = boundingBox
                
                // Normalized bounding box (0-1) - fallback
                val normalizedBox = HashMap<String, Any>()
                normalizedBox["left"] = minX.toDouble()
                normalizedBox["top"] = minY.toDouble()
                normalizedBox["right"] = maxX.toDouble()
                normalizedBox["bottom"] = maxY.toDouble()
                detection["normalizedBox"] = normalizedBox
                
                // Add OBB-specific data
                if (config.includeOBB) {
                    val points = polygon.map { point ->
                        mapOf(
                            "x" to point.x.toDouble(),
                            "y" to point.y.toDouble()
                        )
                    }
                    
                    val obbDataMap = mapOf(
                        "centerX" to obbRes.box.cx.toDouble(),
                        "centerY" to obbRes.box.cy.toDouble(),
                        "width" to obbRes.box.w.toDouble(),
                        "height" to obbRes.box.h.toDouble(),
                        "angle" to obbRes.box.angle.toDouble(),
                        "angleDegrees" to (obbRes.box.angle * 180.0 / Math.PI),
                        "area" to obbRes.box.area.toDouble(),
                        "points" to points,
                        "confidence" to obbRes.confidence.toDouble(),
                        "className" to obbRes.cls,
                        "classIndex" to obbRes.index
                    )
                    
                    detection["obb"] = obbDataMap
                }
                
                detections.add(detection)
            }
            
            map["detections"] = detections
        }

        // Add classification results (if available and enabled for CLASSIFY task)
        if (config.includeClassifications && result.probs != null && result.boxes.isEmpty()) {
            val probs = result.probs!!

            val top5Count = minOf(
                probs.top5Indices.size,
                probs.top5Labels.size,
                probs.top5Confs.size
            )
            val top5List = (0 until top5Count).map { index ->
                val classIdx = probs.top5Indices[index]
                val name = probs.top5Labels[index]
                val conf = probs.top5Confs[index]
                mapOf(
                    "class" to classIdx,
                    "name" to name,
                    "confidence" to conf.toDouble()
                )
            }

            // Add classification result to detections array (for compatibility with YOLOResult.fromMap)
            val detections = (map["detections"] as? List<Map<String, Any>>)?.toMutableList() ?: ArrayList()

            val classificationDetection = HashMap<String, Any>()
            classificationDetection["class"] = probs.top1Index
            classificationDetection["name"] = probs.top1Label
            classificationDetection["confidence"] = probs.top1Conf.toDouble()
            classificationDetection["top5"] = top5List

            // Full image bounding box for classification
            val boundingBox = HashMap<String, Any>()
            boundingBox["left"] = 0.0
            boundingBox["top"] = 0.0
            boundingBox["right"] = result.origShape.width.toDouble()
            boundingBox["bottom"] = result.origShape.height.toDouble()
            classificationDetection["boundingBox"] = boundingBox

            // Normalized bounding box (full image)
            val normalizedBox = HashMap<String, Any>()
            normalizedBox["left"] = 0.0
            normalizedBox["top"] = 0.0
            normalizedBox["right"] = 1.0
            normalizedBox["bottom"] = 1.0
            classificationDetection["normalizedBox"] = normalizedBox

            detections.add(classificationDetection)
            map["detections"] = detections
        }
        
        // Add performance metrics (if enabled)
        if (config.includeProcessingTimeMs) {
            val processingTimeMs = result.speed.toDouble()
            map["processingTimeMs"] = processingTimeMs
        }
        
        if (config.includeFps) {
            map["fps"] = result.fps?.toDouble() ?: 0.0
        }
        
        // Add original image (if available and enabled)
        if (config.includeOriginalImage) {
            result.originalImage?.let { bitmap ->
                val outputStream = java.io.ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                val imageData = outputStream.toByteArray()
                map["originalImage"] = imageData
            }
        }
        
        return map
    }
    
    // endregion
    
    /**
     * Capture current camera frame with detection overlays
     * Returns the captured image as a ByteArray (JPEG format)
     */
    fun captureFrame(): ByteArray? {
        try {
            // Create bitmap to hold the captured frame
            val width = width
            val height = height
            if (width <= 0 || height <= 0) {
                Log.e(TAG, "Invalid view dimensions for capture: ${width}x${height}")
                return null
            }
            
            // Create bitmap and canvas
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            
            // Method 1: Try to get bitmap from PreviewView directly
            var cameraFrameCaptured = false
            previewView.bitmap?.let { cameraBitmap ->
                // Draw the camera bitmap scaled to fit
                val matrix = Matrix()
                val scaleX = width.toFloat() / cameraBitmap.width
                val scaleY = height.toFloat() / cameraBitmap.height
                matrix.setScale(scaleX, scaleY)
                canvas.drawBitmap(cameraBitmap, matrix, null)
                cameraFrameCaptured = true
            }
            
            if (!cameraFrameCaptured) {
                // Method 2: Use hardware acceleration to capture the view
                Log.w(TAG, "PreviewView.bitmap is null, trying hardware capture")
                
                // Enable drawing cache temporarily
                isDrawingCacheEnabled = true
                buildDrawingCache()
                drawingCache?.let { cache ->
                    canvas.drawBitmap(cache, 0f, 0f, null)
                    cameraFrameCaptured = true
                }
                isDrawingCacheEnabled = false
                
                if (!cameraFrameCaptured) {
                    // Method 3: Last resort - draw the entire view hierarchy
                    Log.w(TAG, "Drawing cache failed, using draw method")
                    // Draw PreviewView first
                    previewView.draw(canvas)
                }
            }
            
            // Always draw the overlay on top
            overlayView.draw(canvas)
            
            // Convert bitmap to JPEG byte array
            val outputStream = java.io.ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
            val imageData = outputStream.toByteArray()
            
            // Clean up
            outputStream.close()
            bitmap.recycle()

            return imageData
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing frame", e)
            return null
        }
    }

    /**
     * Stop camera and inference (can be restarted later)
     */
    fun stop() {
        // Set stopped flag first to prevent new frames from being processed
        isStopped = true
        stopUdpPipeline()
        stopInferenceThread()

        try {
            imageAnalysisUseCase?.clearAnalyzer()
            if (::cameraProviderFuture.isInitialized) {
                try {
                    val cameraProvider = cameraProviderFuture.get(1, TimeUnit.SECONDS)
                    cameraProvider.unbindAll()
                } catch (e: Exception) {
                    Log.e(TAG, "Error getting camera provider for unbind", e)
                }
            }

            imageAnalysisUseCase = null

            previewUseCase?.setSurfaceProvider(null)
            previewUseCase = null

            cameraExecutor?.let { exec ->
                exec.shutdown()
                try {
                    if (!exec.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                        Log.w(TAG, "Executor didn't shut down in time; forcing shutdown")
                        exec.shutdownNow()
                        if (!exec.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                            Log.e(TAG, "Executor failed to terminate after forced shutdown")
                        }
                    }
                } catch (e: InterruptedException) {
                    Log.e(TAG, "Interrupted while waiting for executor shutdown", e)
                    exec.shutdownNow()
                    Thread.currentThread().interrupt()
                }
            }
            cameraExecutor = null

            camera = null
            
            // Close predictor safely - ensure no inference is running
            try {
                (predictor as? BasePredictor)?.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error closing predictor", e)
            }
            predictor = null
            inferenceCallback = null
            streamCallback = null
            inferenceResult = null
        } catch (e: Exception) {
            Log.e(TAG, "Error during YOLOView stop", e)
        }
    }

}

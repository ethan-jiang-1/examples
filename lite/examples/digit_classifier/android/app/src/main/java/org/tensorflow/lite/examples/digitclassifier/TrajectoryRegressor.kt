package org.tensorflow.lite.examples.digitclassifier

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

import org.json.JSONObject

class TrajectoryRegressor(private val context: Context) {
  private var interpreter: Interpreter? = null
  var isInitialized = false
    private set

  /** Executor to run inference task in the background */
  private val executorService: ExecutorService = Executors.newCachedThreadPool()

  private var inputImageWidth: Int = 0 // will be inferred from TF Lite model
  private var inputImageHeight: Int = 0 // will be inferred from TF Lite model
  private var modelInputSize: Int = 0 // will be inferred from TF Lite model

  private var input0_shape = arrayOf(1,200,3)
  private var input1_shape = arrayOf(1,200,3)
  private var output0_shape = arrayOf(1,4)
  private var output1_shape = arrayOf(1,4)

  private var model_filename: String = ""


  fun initialize(): Task<Void> {
    Log.i(TAG, "TrajectoryRegressor:initialize")
    val task = TaskCompletionSource<Void>()
    executorService.execute {
      try {

        initializeInterpreter()
        task.setResult(null)
      } catch (e: IOException) {
        task.setException(e)
      }
    }
    return task.task
  }

  private fun is_in_same_shape(shape1: IntArray, shape2: Array<Int>):Boolean {
    if (shape1.size == shape2.size) {
      //var i = 0
      for (i in 0..(shape1.size-1)) {
        if (shape1[i] != shape2[i])
          return false
      }
      return true
    }
    return false
  }

  @Throws(IOException::class)
  private fun initializeInterpreter() {
    Log.i(TAG, "TrajectoryRegressor:initializeInterpreter, Initial TFList started...")
    // Load the TF Lite model
    val assetManager = context.assets
    val model = loadModelFile(assetManager)

    // Initialize TF Lite Interpreter with NNAPI enabled

    val options = Interpreter.Options()
    //Ethan: disable NNAPI for now
    //options.setUseNNAPI(true)

    val interpreter = Interpreter(model, options)

    // Read input shape from model file
    //x_gyro dtype="FLOAT32" input0_shape(1, 200, 3)
    var input0 = interpreter.getInputTensor(0)
    //var input0_shape = input0.shape()
    //var input0_sign = input0.shapeSignature()

    //x_acc dtype="FLOAT32"  input1_shape(1, 200, 3)
    var input1 = interpreter.getInputTensor(1)
    //var input1_shape = input1.shape()
    //var input1_sign = input1.shapeSignature()

    //yhat_delta_q dtype="FLOAT32" output0_shape(1, 4)
    var output0 = interpreter.getOutputTensor(0)
    //var output0_shape = output0.shape()
    //var output0_sign = output0.shapeSignature()

    //yhat_delta_p dtype="FLOAT32" output1_shape(1, 3)
    var output1 = interpreter.getOutputTensor(1)
    //var output1_shape = output1.shape()
    //var output1_sign = output1.shapeSignature()

    Log.d(TAG, "Input0_shape same? "  + is_in_same_shape(input0.shape(), input0_shape).toString())
    Log.d(TAG, "Input1_shape same? "  + is_in_same_shape(input1.shape(), input1_shape).toString())
    Log.d(TAG, "Output0_shape same? " + is_in_same_shape(output0.shape(), output0_shape).toString())
    Log.d(TAG, "Output1_shape same? " + is_in_same_shape(output1.shape(), output1_shape).toString())


    //inputImageWidth = inputShape1[1]
    //inputImageHeight = inputShape1[2]
    //modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * PIXEL_SIZE

    // Finish interpreter initialization
    this.interpreter = interpreter
    isInitialized = true
    Log.d(TAG, "Initialized TFLite interpreter.")


    var capData:TflaCapData = TflaCapData(context)
    capData.parse("tfla_cap_data_0.json")
    capData.summary()
  }

  @Throws(IOException::class)
  private fun loadModelFile(assetManager: AssetManager): ByteBuffer {
    var model_filename = getModelFileName(assetManager)
    Log.i(TAG, "TrajectoryRegressor:loadModelFile " + model_filename)

    val fileDescriptor = assetManager.openFd(model_filename)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
  }

  @Throws(IOException::class)
  private fun readJsonAsset(assetManager: AssetManager, fileName: String): String {
    val file = assetManager.open(fileName)
    val formArray = ByteArray(file.available())
    file.read(formArray)
    file.close()
    return String(formArray)
  }

  private fun getModelFileName(assetManager: AssetManager): String {
    if (model_filename.length == 0) {
      var tfla_info = TflaInfo(context)
      tfla_info.parse("tfla_info.json")
      model_filename = tfla_info.get_model_filename1()
    }
    return model_filename
  }

  private fun classify(bitmap: Bitmap): String {
    Log.i(TAG, "TrajectoryRegressor:classify")

    if (!isInitialized) {
      throw IllegalStateException("TF Lite Interpreter is not initialized yet.")
    }

    var startTime: Long
    var elapsedTime: Long

    // Preprocessing: resize the input
    startTime = System.nanoTime()
    val resizedImage = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)
    val byteBuffer = convertBitmapToByteBuffer(resizedImage)
    elapsedTime = (System.nanoTime() - startTime) / 1000000
    Log.d(TAG, "Preprocessing time = " + elapsedTime + "ms")

    startTime = System.nanoTime()
    val result = Array(1) { FloatArray(OUTPUT_CLASSES_COUNT) }

    var loopmax = 100
    for (i in 1..loopmax) {
      interpreter?.run(byteBuffer, result)
    }
    elapsedTime = (System.nanoTime() - startTime) / 1000000

    Log.d(TAG, "Inference time = " + elapsedTime + "ms" + " within loop " + loopmax)
    Log.d(TAG, getOutputString(result[0]))

    return getOutputString(result[0])
  }

  fun classifyAsync(bitmap: Bitmap): Task<String> {
    Log.i(TAG, "TrajectoryRegressor:classifyAsync")
    val task = TaskCompletionSource<String>()
    executorService.execute {
      val result = classify(bitmap)
      task.setResult(result)
    }
    return task.task
  }

  fun close() {
    Log.i(TAG, "TrajectoryRegressor:close")
    executorService.execute {
      interpreter?.close()
      Log.d(TAG, "Closed TFLite interpreter.")
    }
  }

  private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
    val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
    byteBuffer.order(ByteOrder.nativeOrder())

    val pixels = IntArray(inputImageWidth * inputImageHeight)
    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    for (pixelValue in pixels) {
      val r = (pixelValue shr 16 and 0xFF)
      val g = (pixelValue shr 8 and 0xFF)
      val b = (pixelValue and 0xFF)

      // Convert RGB to grayscale and normalize pixel value to [0..1]
      val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
      byteBuffer.putFloat(normalizedPixelValue)
    }

    return byteBuffer
  }

  private fun getOutputString(output: FloatArray): String {
    val maxIndex = output.indices.maxBy { output[it] } ?: -1
    return "Prediction Result: %d\nConfidence: %2f".format(maxIndex, output[maxIndex])
  }

  companion object {
    private const val TAG = "TrajectoryRegressor"

    //private const val MODEL_FILE = "mnist.tflite"

    private const val FLOAT_TYPE_SIZE = 4
    private const val PIXEL_SIZE = 1

    private const val OUTPUT_CLASSES_COUNT = 10
  }
}

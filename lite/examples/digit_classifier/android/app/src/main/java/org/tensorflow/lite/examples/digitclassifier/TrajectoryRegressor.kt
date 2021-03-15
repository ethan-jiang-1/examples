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

  private var model_filename: String = ""

  private var inputs = Array(2){Array(1){Array(200){FloatArray(3)}}}

  private var pumper: PumpMgr? = null

  fun initialize(cur_pumper: PumpMgr): Task<Void> {
    pumper = cur_pumper
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

  private fun check_interpreter(interpreter: Interpreter): Boolean {

    val input0_shape = arrayOf(1,200,3)
    val input1_shape = arrayOf(1,200,3)
    val output0_shape = arrayOf(1,4)
    val output1_shape = arrayOf(1,4)

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

    var b1 = is_in_same_shape(input0.shape(), input0_shape)
    var b2 = is_in_same_shape(input1.shape(), input1_shape)
    var b3 = is_in_same_shape(output0.shape(), output0_shape)
    var b4 = is_in_same_shape(output1.shape(), output1_shape)

    Log.d(TAG, "CK:Input0_shape same? "  + b1.toString())
    Log.d(TAG, "CK:Input1_shape same? "  + b2.toString())
    Log.d(TAG, "CK:Output0_shape same? " + b3.toString())
    Log.d(TAG, "CK:Output1_shape same? " + b4.toString())

    return b1 and b2 and b3 and b4

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
    options.setNumThreads(2)

    val interpreter = Interpreter(model, options)

    check_interpreter(interpreter)

    // Finish interpreter initialization
    this.interpreter = interpreter
    isInitialized = true
    Log.d(TAG, "Initialized TFLite interpreter.")

  }

  @Throws(IOException::class)
  private fun loadModelFile(assetManager: AssetManager): ByteBuffer {
    model_filename = getModelFileName()
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

  fun getModelFileName(): String {
    if (model_filename.length == 0) {
      var tfla_info = TflaInfo(context)
      tfla_info.parse("tfla_info.json")
      model_filename = tfla_info.get_model_filename1()
    }
    return model_filename
  }

  private fun estimate(est_mode:String): String {
    Log.i(TAG, "TrajectoryRegressor:estimate")

    if (!isInitialized) {
      throw IllegalStateException("TF Lite Interpreter is not initialized yet.")
    }

    //if (!capData.has_parsed()) {
    //  throw IllegalAccessError("capData not parsed")
    //}
    var round = pumper!!.newRound()
    if (round == null) {
      return "FAIL: null"
    }
    if (round < 0) {
      Log.w(TAG, "no new data, skip")
      return "FAIL: no new data"
    }

    //prepare inputs
    Log.i(TAG, "initial iputs")
    pumper!!.feedInputs(inputs, round)


    //prepare outputs
    var output0 = Array(1){FloatArray(4)}
    var output1 = Array(1){FloatArray(3)}
    var outputs = HashMap<Int, Array<FloatArray>>()
    outputs.put(0, output0)
    outputs.put(1, output1)

    //run estimation 100 times
    Log.i(TAG, "start")
    var startTime = System.nanoTime()

    var loopEstimation = pumper!!.loopEstimate()!!
    if (loopEstimation) {
      Log.d(TAG, "loop 100 times to check performance")
      for (i in 0..100) {
        interpreter?.runForMultipleInputsOutputs(inputs, outputs as Map<Int, Any>)
      }

    } else {
      Log.d(TAG, "make estimation by regressor")
      interpreter?.runForMultipleInputsOutputs(inputs, outputs as Map<Int, Any>)
    }
    var elapsedTime = (System.nanoTime() - startTime) / 1000000


    //get outputs out of estimation
    var elapsedTimeMs = elapsedTime.toString()
    Log.i(TAG, "end: " + elapsedTimeMs +  " ms / 100 loop")

    pumper!!.respOutputs(outputs, round)

    return "OK: span100: " + elapsedTimeMs + " ms/100loops"
  }

  fun estimateAsyc(est_mode:String): Task<String> {
    Log.i(TAG, "TrajectoryRegressor:estimateAsyc")
    val task = TaskCompletionSource<String>()
    executorService.execute {
      val result = estimate(est_mode)
      task.setResult(result)
    }
    return task.task
  }

  fun close() {
    Log.i(TAG, "TrajectoryRegressor:close")
    pumper = null
    executorService.execute {
      interpreter?.close()
      Log.d(TAG, "Closed TFLite interpreter.")
    }
  }

  companion object {
    private const val TAG = "TrajectoryRegressor"

    //private const val MODEL_FILE = "mnist.tflite"

    private const val FLOAT_TYPE_SIZE = 4
    private const val PIXEL_SIZE = 1

    private const val OUTPUT_CLASSES_COUNT = 10
  }
}

package org.tensorflow.lite.examples.digitclassifier

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class TrajectoryRegressor(private val context: Context) {
  private var interpreter: Interpreter? = null
  var isInitialized = false
    private set

  /** Executor to run inference task in the background */
  private val executorService: ExecutorService = Executors.newCachedThreadPool()

  private var model_filename: String = ""
  private var model_sgn_filename: String = ""
  private var model_description:String = ""

  private var pumper: PumpMgr? = null
  private var mmsj: MwModelSgnJson? = null

  private var selected_mode = "D"  //"P", "F", "D",  "I"
  private var selected_options_str = ""

  private var fixed_batchsize = -1

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

  private fun selectInitModel() {
    var tfla_info = TflaInfo(context)
    tfla_info.parse("tfla_info.json")

    //if we like to try normal model
    model_filename = tfla_info.get_model_filename(selected_mode)
    model_sgn_filename = tfla_info.get_model_filename_sgn(selected_mode)

    mmsj = MwModelSgnJson(context)
    mmsj!!.parse(model_sgn_filename)

    model_description = mmsj!!.get_model_description()

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

  private fun tensor_signature(tensor: Tensor) {
    var dtype = tensor.dataType()
    var shape = tensor.shape()
    var sgn = tensor.shapeSignature()
    Log.i(TAG, dtype.toString() + "/" + shape.contentToString() + "/" + sgn.contentToString())
  }

  private fun check_interpreter_signature(interpreter: Interpreter): Boolean {

    //Not fixed
    //val input0_shape = arrayOf(1,200,3)
    //val input1_shape = arrayOf(1,200,3)
    //val output_p_shape = arrayOf(1,3)
    //val output_q_shape = arrayOf(1,4)
    //Fixed
    //val input0_shape = arrayOf(10,200,3)
    //val input1_shape = arrayOf(10,200,3)
    //val output_p_shape = arrayOf(10,3)
    //val output_q_shape = arrayOf(10,4)

    var ndx_x_gyro = mmsj!!.get_ndx_input_x_gyro()
    var ndx_x_acc = mmsj!!.get_ndx_input_x_acc()

    // Read input shape from model file
    //x_gyro dtype="FLOAT32" input0_shape(1, 200, 3)
    var input_gyro = interpreter.getInputTensor(ndx_x_gyro)
    Log.i(TAG, "x_gyro @" + ndx_x_gyro.toString())
    tensor_signature(input_gyro)

    //x_acc dtype="FLOAT32"  input1_shape(1, 200, 3)
    var input_acc = interpreter.getInputTensor(ndx_x_acc)
    Log.i(TAG, "x_acc @" + ndx_x_acc.toString())
    tensor_signature(input_acc)


    var ndx_y_delta_p = mmsj!!.get_ndx_output_y_delta_p()
    var ndx_y_delta_q = mmsj!!.get_ndx_output_y_delta_q()

    //yhat_delta_p dtype="FLOAT32" output1_shape(1, 3)
    var output_p = interpreter.getOutputTensor(ndx_y_delta_p)
    Log.i(TAG, "y_delta_p @" + ndx_y_delta_p.toString())
    tensor_signature(output_p)

    //yhat_delta_q dtype="FLOAT32" output0_shape(1, 4)
    var output_q = interpreter.getOutputTensor(ndx_y_delta_q)
    Log.i(TAG, "y_delta_q @" + ndx_y_delta_q.toString())
    tensor_signature(output_q)

    var shape_tensor = input_gyro.shape()
    var shape_mmsi = mmsj!!.get_input_x_gyro_shape()
    Log.i(TAG, "shape_tensor: " + shape_tensor.contentToString())
    Log.i(TAG, "shape mmsi: " + shape_mmsi.toString())
    if(shape_tensor[0] != shape_mmsi[0]) {
      Log.e(TAG, "shape mismatched")
    }
    if(shape_tensor[1] != shape_mmsi[1]) {
      Log.e(TAG, "shape mismatched")
    }
    if(shape_tensor[2] != shape_mmsi[2]) {
      Log.e(TAG, "shape mismatched")
    }

    if (shape_tensor[0] == 1) {
      fixed_batchsize = 1
    } else {
      fixed_batchsize = shape_tensor[0]
    }
    Log.i(TAG, "fixed_batch_size " + fixed_batchsize.toString())

    return true
  }


  public fun getSelectedOptionStr(): String {
    return selected_options_str
  }

  private fun get_options_classic(): Interpreter.Options
  {

    val options = Interpreter.Options()

    model_filename = getModelFileName()

    var iocs = ""
    if (model_filename.contains("_P.tflite")) {
      //iocs = "NNAPI/T4/BHO"
      iocs = "XNNPACK/T4/BHO"
    } else if (model_filename.contains("_F.tflite")) {
      //iocs = "/NNAPI/T4/BHO"
      iocs = "/XNNPACK/T4/BHO"
    } else if (model_filename.contains("_D.tflite")) {
      //iocs = "/NNAPI/T4/BHO"
      iocs = "/XNNPACK/T4/BHO"
    } else if (model_filename.contains("_I.tflite")) {
      iocs = ""
    }

    //Ethan: disable NNAPI for now
    if (iocs.contains("/NNAPI")) {
      Log.d(TAG, "Interpreter Options: use NNAPI ")
      options.setUseNNAPI(true)
    } else if (iocs.contains("/XNNPACK")) {
      Log.d(TAG, "Interpreter Options: use XNNPACK")
      options.setUseXNNPACK(true)
    }

    if (iocs.contains("/T2")) {
      Log.d(TAG, "Interpreter Options: Thread 2")
      options.setNumThreads(2)
    } else if (iocs.contains("/T4")) {
      Log.d(TAG, "Interpreter Options: Thread 4")
      options.setNumThreads(4)
    }

    if (iocs.contains("/FP16")) {
      options.setAllowFp16PrecisionForFp32(true)
    }
    if (iocs.contains("/BHO")) {
      options.setAllowBufferHandleOutput(true)
    }

    selected_options_str = "classic: " + iocs
    return options
  }

  @Throws(IOException::class)
  private fun initializeInterpreter() {
    Log.i(TAG, "TrajectoryRegressor:initializeInterpreter, Initial TFList started...")
    // Load the TF Lite model
    val assetManager = context.assets
    val model = loadModelFile(assetManager)

    // Initialize TF Lite Interpreter with NNAPI enabled

    val options = get_options_classic()

    val interpreter = Interpreter(model, options)

    check_interpreter_signature(interpreter)

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

  
  //ethan: alter the model_file_name
  fun getModelFileName(): String {
    if (model_filename.length == 0) {
      selectInitModel()
    }
    return model_filename
  }

  fun getModelDescription(): String {
    if (model_filename.length == 0) {
      selectInitModel()
    }
    return model_description
  }

  fun getModelSgnFileName(): String {
    if (model_sgn_filename.length == 0) {
      selectInitModel()
    }
    return model_sgn_filename
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

    assert(fixed_batchsize > 0)

    //prepare inputs
    var ti_inputs = Array(2){Array(fixed_batchsize){Array(200){FloatArray(3)}}}

    //prepare inputs
    Log.i(TAG, "initial iputs")
    pumper!!.feedInputs(ti_inputs, round, mmsj!!)

    //prepare outputs
    var ti_outputs = HashMap<Int, Array<FloatArray>>()

    var ndx_y_delta_p = mmsj!!.get_ndx_output_y_delta_p()
    var ndx_y_delta_q = mmsj!!.get_ndx_output_y_delta_q()

    var output0 = Array(fixed_batchsize){FloatArray(4)}
    var output1 = Array(fixed_batchsize){FloatArray(3)}
    ti_outputs.put(ndx_y_delta_q, output0)
    ti_outputs.put(ndx_y_delta_p, output1)

    //run estimation 100 times
    Log.i(TAG, "start")
    var startTime = System.nanoTime()

    var loopEstimation = pumper!!.loopEstimate()!!
    if (loopEstimation) {
      Log.d(TAG, "loop 100 times to check performance")
      var max = 100 / fixed_batchsize
      for (i in 0..max) {
        interpreter?.runForMultipleInputsOutputs(ti_inputs, ti_outputs as Map<Int, Any>)
      }

    } else {
      Log.d(TAG, "make estimation by regressor")
      interpreter?.runForMultipleInputsOutputs(ti_inputs, ti_outputs as Map<Int, Any>)
    }
    var elapsedTime = (System.nanoTime() - startTime) / 1000000


    //get outputs out of estimation
    var elapsedTimeMs = elapsedTime.toString()
    Log.i(TAG, "end: " + elapsedTimeMs +  " ms / 100 loop")

    pumper!!.respOutputs(ti_outputs, round, mmsj!!)

    var estimate_result = ""
    estimate_result += "Span100: " + elapsedTimeMs + " ms/100loops\n"
    estimate_result += "Comments\n"
    estimate_result += "Note\n"
    return estimate_result
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

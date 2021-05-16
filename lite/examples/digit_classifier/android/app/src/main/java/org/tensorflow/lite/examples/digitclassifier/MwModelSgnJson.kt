package org.tensorflow.lite.examples.digitclassifier

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.json.JSONArray
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

import org.json.JSONObject
import org.tensorflow.lite.Tensor
import java.nio.channels.NonReadableChannelException

class MwModelSgnJson(private val context: Context) {
  private var fileName:String = ""

  private var root = JSONObject("{}")

  private var input_x_gyro = -1
  private var input_x_acc = -1
  private var output_y_delta_p = -1
  private var output_y_delta_q = -1
  private var model_description = ""


  private fun parse_data() {
    val assetManager = context.assets

    var cap_data_str = readJsonAsset(assetManager, fileName)

    root = JSONObject(cap_data_str)

    input_x_gyro = root.getInt("input_x_gyro")
    input_x_acc = root.getInt("input_x_acc")
    output_y_delta_p = root.getInt("output_y_delta_p")
    output_y_delta_q = root.getInt("output_y_delta_q")
    model_description = root.getString("model_size")
  }

  @Throws(IOException::class)
  fun parse(jsonFileName: String) {
    Log.i(TAG, "TrajectoryRegressor:initializeInterpreter, Initial TFList started...")
    // Load the TF Lite model
    val assetManager = context.assets

    fileName = jsonFileName

    parse_data()
  }

  fun get_input_x_gyro():Int {
    return input_x_gyro
  }

  fun get_input_x_acc():Int {
    return input_x_acc
  }

  fun get_output_y_delta_p():Int {
    return output_y_delta_p
  }

  fun get_output_y_delta_q():Int {
    return output_y_delta_q
  }

  fun get_model_description(): String {
    return model_description
  }


  @Throws(IOException::class)
  private fun readJsonAsset(assetManager: AssetManager, fileName: String): String {
    val file = assetManager.open(fileName)
    val formArray = ByteArray(file.available())
    file.read(formArray)
    file.close()
    return String(formArray)
  }

  companion object {
    private const val TAG = "MwModelSgnJson"

  }

}

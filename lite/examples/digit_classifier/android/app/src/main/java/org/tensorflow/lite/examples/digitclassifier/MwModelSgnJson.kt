package org.tensorflow.lite.examples.digitclassifier

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import org.json.JSONArray
import java.io.IOException

import org.json.JSONObject

class MwModelSgnJson(private val context: Context) {
  private var fileName:String = ""

  private var root = JSONObject("{}")

  private var input_x_gyro_ndx = -1
  private var input_x_gyro_shape = JSONArray()

  private var input_x_acc_ndx = -1
  private var input_x_acc_shape = JSONArray()

  private var output_y_delta_p_ndx = -1
  private var output_y_delta_p_shape = JSONArray()

  private var output_y_delta_q_ndx = -1
  private var output_y_delta_q_shape = JSONArray()

  private var model_description = ""


  private fun parse_data() {
    val assetManager = context.assets

    var cap_data_str = readJsonAsset(assetManager, fileName)

    root = JSONObject(cap_data_str)

    input_x_gyro_ndx = root.getInt("input_x_gyro")
    input_x_gyro_shape = root.getJSONObject("input_" + input_x_gyro_ndx.toString()).getJSONArray("shape")

    input_x_acc_ndx = root.getInt("input_x_acc")
    input_x_acc_shape = root.getJSONObject("input_" + input_x_acc_ndx.toString()).getJSONArray("shape")


    output_y_delta_p_ndx = root.getInt("output_y_delta_p")
    output_y_delta_p_shape = root.getJSONObject("output_" + output_y_delta_p_ndx.toString()).getJSONArray("shape")

    output_y_delta_q_ndx = root.getInt("output_y_delta_q")
    output_y_delta_q_shape = root.getJSONObject("output_" + output_y_delta_q_ndx.toString()).getJSONArray("shape")


    var model_size = root.getString("model_size")
    var version_git = root.getString("version_git")
    var feed_mode = root.getString("feed_mode")

    var converter = root.getJSONObject("converter")
    var converter_model = converter.getJSONObject("model")
    var downsize_ratio = converter_model.getString("tflite_model.downsize_ratio")

    model_description = model_size + " " + downsize_ratio + "\n"
    model_description += version_git + " " + feed_mode
  }

  @Throws(IOException::class)
  fun parse(jsonFileName: String) {
    Log.i(TAG, "TrajectoryRegressor:initializeInterpreter, Initial TFList started...")
    // Load the TF Lite model
    val assetManager = context.assets

    fileName = jsonFileName

    parse_data()
  }

  fun get_ndx_input_x_gyro():Int {
    return input_x_gyro_ndx
  }

  fun get_input_x_gyro_shape(): JSONArray {
    return input_x_gyro_shape
  }

  fun get_ndx_input_x_acc():Int {
    return input_x_acc_ndx
  }

  fun get_input_x_acc_shape(): JSONArray {
    return input_x_acc_shape
  }

  fun get_ndx_output_y_delta_p():Int {
    return output_y_delta_p_ndx
  }

  fun get_output_y_delta_p_shape(): JSONArray {
    return output_y_delta_p_shape
  }

  fun get_ndx_output_y_delta_q():Int {
    return output_y_delta_q_ndx
  }

  fun get_output_y_delta_q_shape(): JSONArray {
    return output_y_delta_p_shape
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

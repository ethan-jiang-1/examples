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

class TflaInfo(private val context: Context) {
  private var fileName:String = ""

  private var root = JSONObject("{}")

  private fun parse_data() {
    val assetManager = context.assets

    var cap_data_str = readJsonAsset(assetManager, fileName)

    root = JSONObject(cap_data_str)
  }

  @Throws(IOException::class)
  fun parse(jsonFileName: String) {
    Log.i(TAG, "TrajectoryRegressor:initializeInterpreter, Initial TFList started...")
    // Load the TF Lite model
    val assetManager = context.assets

    fileName = jsonFileName

    parse_data()
  }


  fun get_model_filename(name:String): String{
    var node_name = "model" + name
    return root.getString(node_name)
  }
  fun get_model_filename_sgn(name:String): String{
    var node_name = "model" + name + "_sgn"
    return root.getString(node_name)
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
    private const val TAG = "TflaInfo"

  }

  fun summary() {

  }
}

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

class TflaCapData(private val context: Context) {
  private var inited: Boolean = false
  private var fileName:String = ""

  private var yhat_delta_p = Array(1){FloatArray(3)}
  private var yhat_delta_q = Array(1){FloatArray(4)}
  private var x_gyro = Array(1){Array(200){FloatArray(3)}}
  private var x_acc = Array(1){Array(200){FloatArray(3)}}

  private fun parse_yhat_delta_p(root:JSONObject) {
    var cap_syhat_delta_p = root.getJSONObject("syhat_delta_p")

    //var np_shape = cap_syhat_delta_p.getJSONArray("np_shape")
    //var np_size = cap_syhat_delta_p.getInt("np_size")
    //var np_ndim = cap_syhat_delta_p.getInt("np_ndim")
    var data = cap_syhat_delta_p.getJSONArray("data")

    var data0 = data[0]
    //val delta_p = FloatArray(3)
    for (i in 0..2) {
       var dval = (data0 as JSONArray).get(i)
       yhat_delta_p[0][i] = (dval as Double).toFloat()
    }
  }

  private fun parse_yhat_delta_q(root:JSONObject) {
    var cap_syhat_delta_q = root.getJSONObject("syhat_delta_q")

    //var np_shape = cap_syhat_delta_q.getJSONArray("np_shape")
    //var np_size = cap_syhat_delta_p.getInt("np_size")
    //var np_ndim = cap_syhat_delta_p.getInt("np_ndim")
    var data = cap_syhat_delta_q.getJSONArray("data")

    var data0 = data[0]
    //val delta_q = FloatArray(4)
    for (i in 0..3) {
      var dval = (data0 as JSONArray).get(i)
      yhat_delta_q[0][i] = (dval as Double).toFloat()
    }
  }

  private fun parse_x_gyro(root: JSONObject) {
    var cap_sx_gyro = root.getJSONObject("sx_gyro")

    var data = cap_sx_gyro.getJSONArray("data")
    for (i in 0..199) {
      var datai = data[i]
      for (j in 0..2) {
        var dval = (datai as JSONArray).get(j)
        x_gyro[0][i][j] = (dval as Double).toFloat()
      }
    }
  }


  private fun parse_x_acc(root: JSONObject) {
    var cap_sx_acc= root.getJSONObject("sx_acc")

    var data = cap_sx_acc.getJSONArray("data")
    for (i in 0..199) {
      var datai = data[i]
      for (j in 0..2) {
        var dval = (datai as JSONArray).get(j)
        x_acc[0][i][j] = (dval as Double).toFloat()
      }
    }
  }



  private fun parse_data() {
    val assetManager = context.assets

    var cap_data_str = readJsonAsset(assetManager, fileName)

    var root = JSONObject(cap_data_str)
    //var cap_fiename = root.getString("cap_filename")
    //var cap_order = root.getInt("cap_order")

    parse_x_gyro(root)
    parse_x_acc(root)
    parse_yhat_delta_p(root)
    parse_yhat_delta_q(root)

    print(yhat_delta_p)
    print(yhat_delta_q)
  }

  @Throws(IOException::class)
  fun parse(jsonFileName: String) {
    Log.i(TAG, "TrajectoryRegressor:initializeInterpreter, Initial TFList started...")
    // Load the TF Lite model
    val assetManager = context.assets

    fileName = jsonFileName

    parse_data()
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
    private const val TAG = "TflaCapData"

  }

  fun summary() {

  }
}

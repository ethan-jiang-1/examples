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

class PumpCapData(private val context: Context): PumpDataBase() {
  private var fileName:String = ""


  private var capData:TflaCapData? = null
  private var round_no:Int = 0

  fun init(filename:String) {
    // Initial captured data from json
    capData = TflaCapData(context)
    capData!!.parse(filename)
    capData!!.summary()
    Log.d(TAG, "capData parsed:" + capData!!.has_parsed().toString())

  }

  override fun newRound(): Int {
    round_no += 1
    Log.d(TAG, "newRpund @"+ round_no.toString())
    return round_no
  }

  override fun feedInputs(inputs: Array<Array<Array<FloatArray>>>, round:Int) {
    var x_gyro = capData!!.get_x_gyro()
    var x_acc  = capData!!.get_x_acc()
    for (i in 0..199) {
      for (j in 0..2) {
        inputs[1][0][i][j] = x_gyro[i][j]
        inputs[0][0][i][j] = x_acc[i][j]
      }
    }
    Log.d(TAG, "feedInputs @"+ round.toString())
  }

  override fun respOutputs(outputs:HashMap<Int, Array<FloatArray>>, round:Int) {
    var yhat_delta_p = outputs.get(1)?.get(0)
    var yhat_delta_q = outputs.get(0)?.get(0)

    Log.d(TAG, "respOutpus @"+ round.toString())
  }


  companion object {
    private const val TAG = "PumpDataCapData"

  }
}

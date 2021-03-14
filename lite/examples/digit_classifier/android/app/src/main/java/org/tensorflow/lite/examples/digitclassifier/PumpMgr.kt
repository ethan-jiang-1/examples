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

class PumpMgr(private val context: Context) {
  private var init_mode:String = ""

  private var pumperInterface: PumpDataBase? = null

  private var capData:PumpCapData? = null


  fun init(mode:String) {
    init_mode = mode

    Log.w(TAG, "mode:  " + mode.toString())
    if (mode == "capData:0") {
      capData = PumpCapData(context)
      capData!!.init("tfla_cap_data_0.json")
      pumperInterface = capData
    } else {
      Log.e(TAG, "mode not supported: " + mode)
    }
  }

  fun newRound(): Int? {
    return pumperInterface?.newRound()
  }

  fun feedInputs( inputs: Array<Array<Array<FloatArray>>>, round:Int): Unit? {
    return pumperInterface?.feedInputs(inputs, round)
  }

  fun respOutputs(outputs:HashMap<Int, Array<FloatArray>>, round:Int): Unit? {
    return pumperInterface?.respOutputs(outputs,round)
  }


  companion object {
    private const val TAG = "PumpMgr"

  }
}

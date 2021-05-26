package org.tensorflow.lite.examples.digitclassifier

import android.content.Context
import android.util.Log

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

  override fun feedInputs(inputs: Array<Array<Array<FloatArray>>>, round:Int, mmsj:MwModelSgnJson) {
    var x_gyro = capData!!.get_x_gyro()
    var x_acc  = capData!!.get_x_acc()

    var ndx_x_gyro = mmsj!!.get_ndx_input_x_gyro()
    var ndx_x_acc = mmsj!!.get_ndx_input_x_acc()

    for (i in 0..199) {
      for (j in 0..2) {
        inputs[ndx_x_gyro][0][i][j] = x_gyro[i][j]
        inputs[ndx_x_acc][0][i][j] = x_acc[i][j]
      }
    }
    Log.d(TAG, "feedInputs @"+ round.toString())
  }

  override fun respOutputs(outputs:HashMap<Int, Array<FloatArray>>, round:Int, mmsj:MwModelSgnJson) {
    var ndx_y_delta_p = mmsj!!.get_ndx_output_y_delta_p()
    var ndx_y_delta_q = mmsj!!.get_ndx_output_y_delta_q()

    var yhat_delta_p = outputs.get(ndx_y_delta_p)?.get(0)
    var yhat_delta_q = outputs.get(ndx_y_delta_q)?.get(0)

    Log.d(TAG, "respOutpus @"+ round.toString())
  }

  override fun loopEstimate(): Boolean {
    return true
  }

  companion object {
    private const val TAG = "PumpDataCapData"

  }
}

package org.tensorflow.lite.examples.digitclassifier

import android.util.Log

abstract class PumpDataBase {
    abstract fun newRound(): Int
    abstract fun feedInputs( inputs: Array<Array<Array<FloatArray>>>, round:Int)
    abstract fun respOutputs(outputs:HashMap<Int, Array<FloatArray>>, round:Int)
}
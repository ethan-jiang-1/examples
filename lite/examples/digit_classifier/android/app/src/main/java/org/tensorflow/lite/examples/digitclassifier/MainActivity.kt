package org.tensorflow.lite.examples.digitclassifier

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.util.Log
import android.widget.Button
import android.widget.TextView
import com.divyanshu.draw.widget.DrawView

class MainActivity : AppCompatActivity() {

  private var drawView: DrawView? = null
  private var clearButton: Button? = null
  private var startOneButton: Button? = null
  private var startMultiButton: Button? = null
  private var exitButton: Button? = null

  private var predictedTextView: TextView? = null
  private var modelTextView: TextView? = null
  private var pumpTextView: TextView? = null

  private var pumper: PumpMgr? = null

  val FINISH = "finish_key_extra"


  // the digitClassifier
  //private var digitClassifier = DigitClassifier(this)
  // private var digitClassifier = null
//  private val digitClassifier by lazy {Log.w(TAG, "Lazy loading digitClassifier")
//                                       DigitClassifier(this)}

  //the trajectory regressor
  private val trajectoryRegressor by lazy {Log.w(TAG, "Lazy loading trajectoryRegressor")
                                          TrajectoryRegressor(this)}

  @SuppressLint("ClickableViewAccessibility")
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.tfe_dc_activity_main)

    Log.w(TAG, "MainActivity:onCreate")

    // Setup view instances
    drawView = findViewById(R.id.draw_view)
    drawView?.setStrokeWidth(10.0f)
    drawView?.setColor(Color.WHITE)
    drawView?.setBackgroundColor(Color.BLACK)

    clearButton = findViewById(R.id.clear_button)
    startOneButton = findViewById(R.id.start_one_button)
    startMultiButton = findViewById(R.id.start_multi_button)
    exitButton = findViewById(R.id.exit_button)

    predictedTextView = findViewById(R.id.predicted_text)
    modelTextView = findViewById(R.id.model_text)
    pumpTextView = findViewById(R.id.pump_text)

    // Setup clear drawing button
    clearButton?.setOnClickListener {
      Log.w(TAG, "MainActivity:clearButton clicked")
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.tfe_dc_prediction_text_placeholder)
    }

    //Setup start one button
    startOneButton?.setOnClickListener{
      Log.w(TAG, "MainActivity:startButton clicked")
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.tfe_dc_start_one_button_text)
      
      estimateTrajectory("one")
    }

    //Setup start multi button
    startMultiButton?.setOnClickListener{
      Log.w(TAG, "MainActivity:stopButton clicked")
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.tfe_dc_start_multi_button_text)

      estimateTrajectory("multi")
    }

    //Setup exit button
    exitButton?.setOnClickListener{
      Log.w(TAG, "MainActivity:resetButton clicked")
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.tfe_dc_exit_button_text)

      val intent = Intent(this, MainActivity::class.java)
      intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
      intent.putExtra(FINISH, true)
      finish()
    }

    var finish = getIntent().getBooleanExtra(FINISH, false) //default false if not set by argument
    if(finish) {
      finish()
      return
    }


    //setup classifier
    //setup_classifer()

    //setup estimator
    setup_estimator()
  }

  private fun setup_classifer() {

    // Setup classification trigger so that it classify after every stroke drew
    drawView?.setOnTouchListener { _, event ->
      // As we have interrupted DrawView's touch event,
      // we first need to pass touch events through to the instance for the drawing to show up
      drawView?.onTouchEvent(event)

      // Then if user finished a touch event, run classification
      //if (event.action == MotionEvent.ACTION_UP) {
      //  classifyDrawing()
      //}

      true
    }

    // Setup digit classifier
    Log.w(TAG, "initial digtitClassifier in MainActivatiy")
    //digitClassifier = DigitClassifier(this)
    //digitClassifier
    //  .initialize()
    //  .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit classifier.", e) }

  }

  private fun setup_estimator() {
    //Init pumper
    pumper = PumpMgr(this)
    val pump_mode = "capData:tfla_cap_data_0.json"
    var init_ret = pumper!!.init(pump_mode)

    // Setup trajector Regressor
    Log.w(TAG, "initial trajectorRegressor in MainActivatiy: " + init_ret)
    trajectoryRegressor
      .initialize(pumper!!)
      .addOnFailureListener { e -> Log.e(TAG, "Error to setting up trajectory regressor.", e) }

    modelTextView?.text = trajectoryRegressor.getModelFileName() + trajectoryRegressor.getInterpreterOptionsControllStr()
    pumpTextView?.text = pump_mode
  }

  override fun onDestroy() {
//    if (digitClassifier.isInitialized) {
//      digitClassifier.close()
//    }
    if (trajectoryRegressor.isInitialized) {
      trajectoryRegressor.close()
    }
    super.onDestroy()
  }

  private fun classifyDrawing() {
//    val bitmap = drawView?.getBitmap()
//
//    if ((bitmap != null) && (digitClassifier.isInitialized)) {
//      digitClassifier
//        .classifyAsync(bitmap)
//        .addOnSuccessListener { resultText -> predictedTextView?.text = resultText }
//        .addOnFailureListener { e ->
//          predictedTextView?.text = getString(
//            R.string.tfe_dc_classification_error_message,
//            e.localizedMessage
//          )
//          Log.e(TAG, "Error classifying drawing.", e)
//        }
//    }
  }


  private fun estimateTrajectory(est_mode:String) {
    if (trajectoryRegressor.isInitialized) {
       trajectoryRegressor
         .estimateAsyc(est_mode)
         .addOnSuccessListener { resultText -> predictedTextView?.text = resultText }
    }
  }

  companion object {
    private const val TAG = "MainActivity"
  }
}

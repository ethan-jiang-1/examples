package org.tensorflow.lite.examples.digitclassifier

import android.annotation.SuppressLint
import android.graphics.Color
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import com.divyanshu.draw.widget.DrawView

class MainActivity : AppCompatActivity() {

  private var drawView: DrawView? = null
  private var clearButton: Button? = null
  private var startButton: Button? = null
  private var stopButton: Button? = null
  private var resetButton: Button? = null
  private var predictedTextView: TextView? = null

  // the digitClassifier
  //private var digitClassifier = DigitClassifier(this)
  //private var digitClassifier = null
  private val digitClassifier by lazy {Log.w(TAG, "Lazy loading digitClassifier")
                                       DigitClassifier(this)}

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
    startButton = findViewById(R.id.start_button)
    stopButton = findViewById(R.id.stop_button)
    resetButton = findViewById(R.id.reset_button)

    predictedTextView = findViewById(R.id.predicted_text)

    // Setup clear drawing button
    clearButton?.setOnClickListener {
      Log.w(TAG, "MainActivity:clearButton clicked")
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.tfe_dc_prediction_text_placeholder)
    }

    //Setup start button
    startButton?.setOnClickListener{
      Log.w(TAG, "MainActivity:startButton clicked")
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.tfe_dc_start_button_text)
    }

    //Setup start button
    stopButton?.setOnClickListener{
      Log.w(TAG, "MainActivity:stopButton clicked")
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.tfe_dc_stop_button_text)
    }

    //Setup start button
    resetButton?.setOnClickListener{
      Log.w(TAG, "MainActivity:resetButton clicked")
      drawView?.clearCanvas()
      predictedTextView?.text = getString(R.string.tfe_dc_reset_button_text)
    }


    // Setup classification trigger so that it classify after every stroke drew
    drawView?.setOnTouchListener { _, event ->
      // As we have interrupted DrawView's touch event,
      // we first need to pass touch events through to the instance for the drawing to show up
      drawView?.onTouchEvent(event)

      // Then if user finished a touch event, run classification
//      if (event.action == MotionEvent.ACTION_UP) {
//        classifyDrawing()
//      }

      true
    }

    // Setup digit classifier
    Log.w(TAG, "initial digtitClassifier in MainActivatiy")
    //digitClassifier = DigitClassifier(this)
    digitClassifier
      .initialize()
      .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit classifier.", e) }

    // Setup digit classifier
    Log.w(TAG, "initial trajectorRegressor in MainActivatiy")
    trajectoryRegressor
      .initialize()
      .addOnFailureListener { e -> Log.e(TAG, "Error to setting up trajectory regressor.", e) }


  }

  override fun onDestroy() {
    if (digitClassifier.isInitialized) {
      digitClassifier.close()
    }
    if (trajectoryRegressor.isInitialized) {
      trajectoryRegressor.close()
    }
    super.onDestroy()
  }

  private fun classifyDrawing() {
    val bitmap = drawView?.getBitmap()

    if ((bitmap != null) && (digitClassifier.isInitialized)) {
      digitClassifier
        .classifyAsync(bitmap)
        .addOnSuccessListener { resultText -> predictedTextView?.text = resultText }
        .addOnFailureListener { e ->
          predictedTextView?.text = getString(
            R.string.tfe_dc_classification_error_message,
            e.localizedMessage
          )
          Log.e(TAG, "Error classifying drawing.", e)
        }
    }
  }

  companion object {
    private const val TAG = "MainActivity"
  }
}

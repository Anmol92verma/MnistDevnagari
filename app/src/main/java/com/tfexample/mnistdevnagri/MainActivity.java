package com.tfexample.mnistdevnagri;

import android.app.Activity;
import android.graphics.PointF;
import android.os.Bundle;
import android.support.v4.util.ArrayMap;
import android.text.SpannableStringBuilder;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import com.tfexample.mnistdevnagri.draw.DrawModel;
import com.tfexample.mnistdevnagri.draw.DrawView;
import com.tfexample.mnistdevnagri.tflite.Classifier;
import com.tfexample.mnistdevnagri.tflite.TensorFlowImageClassifier;

import java.io.IOException;
import java.util.List;

/**
 * [[0, '/Users/anmolverma/ml/nhcd/consonants/32'],
 * [1, '/Users/anmolverma/ml/nhcd/consonants/35'],
 * [2, '/Users/anmolverma/ml/nhcd/consonants/34'],
 * [3, '/Users/anmolverma/ml/nhcd/consonants/33'],
 * [4, '/Users/anmolverma/ml/nhcd/consonants/20'],
 * [5, '/Users/anmolverma/ml/nhcd/consonants/18'],
 * [6, '/Users/anmolverma/ml/nhcd/consonants/27'],
 * [7, '/Users/anmolverma/ml/nhcd/consonants/9'],
 * [8, '/Users/anmolverma/ml/nhcd/consonants/11'],
 * [9, '/Users/anmolverma/ml/nhcd/consonants/7'],
 * [10, '/Users/anmolverma/ml/nhcd/consonants/29'],
 * [11, '/Users/anmolverma/ml/nhcd/consonants/16'],
 * [12, '/Users/anmolverma/ml/nhcd/consonants/6'],
 * [13, '/Users/anmolverma/ml/nhcd/consonants/28'],
 * [14, '/Users/anmolverma/ml/nhcd/consonants/17'],
 * [15, '/Users/anmolverma/ml/nhcd/consonants/1'],
 * [16, '/Users/anmolverma/ml/nhcd/consonants/10'],
 * [17, '/Users/anmolverma/ml/nhcd/consonants/19'],
 * [18, '/Users/anmolverma/ml/nhcd/consonants/26'],
 * [19, '/Users/anmolverma/ml/nhcd/consonants/8'],
 * [20, '/Users/anmolverma/ml/nhcd/consonants/21'],
 * [21, '/Users/anmolverma/ml/nhcd/consonants/36'],
 * [22, '/Users/anmolverma/ml/nhcd/consonants/31'],
 * [23, '/Users/anmolverma/ml/nhcd/consonants/30'],
 * [24, '/Users/anmolverma/ml/nhcd/consonants/24'],
 * [25, '/Users/anmolverma/ml/nhcd/consonants/23'],
 * [26, '/Users/anmolverma/ml/nhcd/consonants/4'],
 * [27, '/Users/anmolverma/ml/nhcd/consonants/15'],
 * [28, '/Users/anmolverma/ml/nhcd/consonants/3'],
 * [29, '/Users/anmolverma/ml/nhcd/consonants/12'],
 * [30, '/Users/anmolverma/ml/nhcd/consonants/2'],
 * [31, '/Users/anmolverma/ml/nhcd/consonants/13'],
 * [32, '/Users/anmolverma/ml/nhcd/consonants/5'],
 * [33, '/Users/anmolverma/ml/nhcd/consonants/14'],
 * [34, '/Users/anmolverma/ml/nhcd/consonants/22'],
 * [35, '/Users/anmolverma/ml/nhcd/consonants/25'],
 * [36, '/Users/anmolverma/ml/nhcd/numerals/9'],
 * [37, '/Users/anmolverma/ml/nhcd/numerals/0'],
 * [38, '/Users/anmolverma/ml/nhcd/numerals/7'],
 * [39, '/Users/anmolverma/ml/nhcd/numerals/6'],
 * [40, '/Users/anmolverma/ml/nhcd/numerals/1'],
 * [41, '/Users/anmolverma/ml/nhcd/numerals/8'],
 * [42, '/Users/anmolverma/ml/nhcd/numerals/4'],
 * [43, '/Users/anmolverma/ml/nhcd/numerals/3'],
 * [44, '/Users/anmolverma/ml/nhcd/numerals/2'],
 * [45, '/Users/anmolverma/ml/nhcd/numerals/5'],
 * [46, '/Users/anmolverma/ml/nhcd/vowels/9'],
 * [47, '/Users/anmolverma/ml/nhcd/vowels/11'],
 * [48, '/Users/anmolverma/ml/nhcd/vowels/7'],
 * [49, '/Users/anmolverma/ml/nhcd/vowels/6'],
 * [50, '/Users/anmolverma/ml/nhcd/vowels/1'],
 * [51, '/Users/anmolverma/ml/nhcd/vowels/10'],
 * [52, '/Users/anmolverma/ml/nhcd/vowels/8'],
 * [53, '/Users/anmolverma/ml/nhcd/vowels/4'],
 * [54, '/Users/anmolverma/ml/nhcd/vowels/3'],
 * [55, '/Users/anmolverma/ml/nhcd/vowels/12'],
 * [56, '/Users/anmolverma/ml/nhcd/vowels/2'],
 * [57, '/Users/anmolverma/ml/nhcd/vowels/5']]
 */

/**
 * [[0, '/Users/anmolverma/ml/nhcd/numerals/9'],
 * [1, '/Users/anmolverma/ml/nhcd/numerals/0'],
 * [2, '/Users/anmolverma/ml/nhcd/numerals/7'],
 * [3, '/Users/anmolverma/ml/nhcd/numerals/6'],
 * [4, '/Users/anmolverma/ml/nhcd/numerals/1'],
 * [5, '/Users/anmolverma/ml/nhcd/numerals/8'],
 * [6, '/Users/anmolverma/ml/nhcd/numerals/4'],
 * [7, '/Users/anmolverma/ml/nhcd/numerals/3'],
 * [8, '/Users/anmolverma/ml/nhcd/numerals/2'],
 * [9, '/Users/anmolverma/ml/nhcd/numerals/5']]
 */

public class MainActivity extends Activity implements View.OnClickListener, View.OnTouchListener {

    private static final int PIXEL_WIDTH = 28;

    // ui elements
    private Button clearBtn, classBtn;
    private TextView resText;
    // views
    private DrawModel drawModel;
    private DrawView drawView;
    private PointF mTmpPiont = new PointF();

    private float mLastX;
    private float mLastY;
    private Classifier classifier;
    private ArrayMap<Integer, String> labelMap;

    @Override
    // In the onCreate() method, you perform basic application startup logic that should happen
    //only once for the entire life of the activity.
    protected void onCreate(Bundle savedInstanceState) {
        //initialization
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        prepareLabels();

        //get drawing view from XML (where the finger writes the number)
        drawView = (DrawView) findViewById(R.id.draw);
        //get the model object
        drawModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);

        //init the view with the model object
        drawView.setModel(drawModel);
        // give it a touch listener to activate when the user taps
        drawView.setOnTouchListener(this);

        //clear button
        //clear the drawing when the user taps
        clearBtn = (Button) findViewById(R.id.btn_clear);
        clearBtn.setOnClickListener(this);

        //class button
        //when tapped, this performs classification on the drawn image
        classBtn = (Button) findViewById(R.id.btn_class);
        classBtn.setOnClickListener(this);

        // res text
        //this is the text that shows the output of the classification
        resText = (TextView) findViewById(R.id.tfRes);

        // tensorflow
        //load up our saved model to perform inference from local storage
        try {
            classifier = TensorFlowImageClassifier.create(getAssets(), "converted_model.tflite", "labels.txt", PIXEL_WIDTH);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void prepareLabels() {
        labelMap = new ArrayMap<Integer,String>();
        labelMap.put(0, "Users/anmolverma/ml/nhcd/consonants/32");
        labelMap.put(1, "Users/anmolverma/ml/nhcd/consonants/35");
        labelMap.put(2, "Users/anmolverma/ml/nhcd/consonants/34");
        labelMap.put(3, "Users/anmolverma/ml/nhcd/consonants/33");
        labelMap.put(4, "Users/anmolverma/ml/nhcd/consonants/20");
        labelMap.put(5, "Users/anmolverma/ml/nhcd/consonants/18");
        labelMap.put(6, "Users/anmolverma/ml/nhcd/consonants/27");
        labelMap.put(7, "Users/anmolverma/ml/nhcd/consonants/9");
        labelMap.put(8, "Users/anmolverma/ml/nhcd/consonants/11");
        labelMap.put(9, "Users/anmolverma/ml/nhcd/consonants/7");
        labelMap.put(10, "Users/anmolverma/ml/nhcd/consonants/29");
        labelMap.put(11, "Users/anmolverma/ml/nhcd/consonants/16");
        labelMap.put(12, "Users/anmolverma/ml/nhcd/consonants/6");
        labelMap.put(13, "Users/anmolverma/ml/nhcd/consonants/28");
        labelMap.put(14, "Users/anmolverma/ml/nhcd/consonants/17");
        labelMap.put(15, "Users/anmolverma/ml/nhcd/consonants/1");
        labelMap.put(16, "Users/anmolverma/ml/nhcd/consonants/10");
        labelMap.put(17, "Users/anmolverma/ml/nhcd/consonants/19");
        labelMap.put(18, "Users/anmolverma/ml/nhcd/consonants/26");
        labelMap.put(19, "Users/anmolverma/ml/nhcd/consonants/8");
        labelMap.put(20, "Users/anmolverma/ml/nhcd/consonants/21");
        labelMap.put(21, "Users/anmolverma/ml/nhcd/consonants/36");
        labelMap.put(22, "Users/anmolverma/ml/nhcd/consonants/31");
        labelMap.put(23, "Users/anmolverma/ml/nhcd/consonants/30");
        labelMap.put(24, "Users/anmolverma/ml/nhcd/consonants/24");
        labelMap.put(25, "Users/anmolverma/ml/nhcd/consonants/23");
        labelMap.put(26, "Users/anmolverma/ml/nhcd/consonants/4");
        labelMap.put(27, "Users/anmolverma/ml/nhcd/consonants/15");
        labelMap.put(28, "Users/anmolverma/ml/nhcd/consonants/3");
        labelMap.put(29, "Users/anmolverma/ml/nhcd/consonants/12");
        labelMap.put(30, "Users/anmolverma/ml/nhcd/consonants/2");
        labelMap.put(31, "Users/anmolverma/ml/nhcd/consonants/13");
        labelMap.put(32, "Users/anmolverma/ml/nhcd/consonants/5");
        labelMap.put(33, "Users/anmolverma/ml/nhcd/consonants/14");
        labelMap.put(34, "Users/anmolverma/ml/nhcd/consonants/22");
        labelMap.put(35, "Users/anmolverma/ml/nhcd/consonants/25");
        labelMap.put(36, "Users/anmolverma/ml/nhcd/numerals/9");
        labelMap.put(37, "Users/anmolverma/ml/nhcd/numerals/0");
        labelMap.put(38, "Users/anmolverma/ml/nhcd/numerals/7");
        labelMap.put(39, "Users/anmolverma/ml/nhcd/numerals/6");
        labelMap.put(40, "Users/anmolverma/ml/nhcd/numerals/1");
        labelMap.put(41, "Users/anmolverma/ml/nhcd/numerals/8");
        labelMap.put(42, "Users/anmolverma/ml/nhcd/numerals/4");
        labelMap.put(43, "Users/anmolverma/ml/nhcd/numerals/3");
        labelMap.put(44, "Users/anmolverma/ml/nhcd/numerals/2");
        labelMap.put(45, "Users/anmolverma/ml/nhcd/numerals/5");
        labelMap.put(46, "Users/anmolverma/ml/nhcd/vowels/9");
        labelMap.put(47, "Users/anmolverma/ml/nhcd/vowels/11");
        labelMap.put(48, "Users/anmolverma/ml/nhcd/vowels/7");
        labelMap.put(49, "Users/anmolverma/ml/nhcd/vowels/6");
        labelMap.put(50, "Users/anmolverma/ml/nhcd/vowels/1");
        labelMap.put(51, "Users/anmolverma/ml/nhcd/vowels/10");
        labelMap.put(52, "Users/anmolverma/ml/nhcd/vowels/8");
        labelMap.put(53, "Users/anmolverma/ml/nhcd/vowels/4");
        labelMap.put(54, "Users/anmolverma/ml/nhcd/vowels/3");
        labelMap.put(55, "Users/anmolverma/ml/nhcd/vowels/12");
        labelMap.put(56, "Users/anmolverma/ml/nhcd/vowels/2");
        labelMap.put(57, "Users/anmolverma/ml/nhcd/vowels/5");
    }

    //the activity lifecycle

    @Override
    //OnResume() is called when the user resumes his Activity which he left a while ago,
    // //say he presses home button and then comes back to app, onResume() is called.
    protected void onResume() {
        drawView.onResume();
        super.onResume();
    }

    @Override
    //OnPause() is called when the user receives an event like a call or a text message,
    // //when onPause() is called the Activity may be partially or completely hidden.
    protected void onPause() {
        drawView.onPause();
        super.onPause();
    }

    @Override
    public void onClick(View view) {
        //when the user clicks something
        if (view.getId() == R.id.btn_clear) {
            //if its the clear button
            //clear the drawing
            drawModel.clear();
            drawView.reset();
            drawView.invalidate();
            //empty the text view
            resText.setText("");
        } else if (view.getId() == R.id.btn_class) {
            //if the user clicks the classify button
            //get the pixel data and store it in an array
            int[] pixels = drawView.getPixelData();
            SpannableStringBuilder textToShow = new SpannableStringBuilder();
            List<Classifier.Recognition> recog = classifier.recognizeImage(pixels);
            for (int i = 0; i < recog.size(); i++) {
                textToShow.append(recog.get(i).toString()+" "+labelMap.get(Integer.parseInt(recog.get(i).getId())));
                textToShow.append("\n");
            }
            resText.setText(textToShow.toString());
        }
    }

    @Override
    //this method detects which direction a user is moving
    //their finger and draws a line accordingly in that
    //direction
    public boolean onTouch(View v, MotionEvent event) {
        //get the action and store it as an int
        int action = event.getAction() & MotionEvent.ACTION_MASK;
        //actions have predefined ints, lets match
        //to detect, if the user has touched, which direction the users finger is
        //moving, and if they've stopped moving

        //if touched
        if (action == MotionEvent.ACTION_DOWN) {
            //begin drawing line
            processTouchDown(event);
            return true;
            //draw line in every direction the user moves
        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;
            //if finger is lifted, stop drawing
        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }

    //draw line down

    private void processTouchDown(MotionEvent event) {
        //calculate the x, y coordinates where the user has touched
        mLastX = event.getX();
        mLastY = event.getY();
        //user them to calcualte the position
        drawView.calcPos(mLastX, mLastY, mTmpPiont);
        //store them in memory to draw a line between the
        //difference in positions
        float lastConvX = mTmpPiont.x;
        float lastConvY = mTmpPiont.y;
        //and begin the line drawing
        drawModel.startLine(lastConvX, lastConvY);
    }

    //the main drawing function
    //it actually stores all the drawing positions
    //into the drawmodel object
    //we actually render the drawing from that object
    //in the drawrenderer class
    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        drawView.calcPos(x, y, mTmpPiont);
        float newConvX = mTmpPiont.x;
        float newConvY = mTmpPiont.y;
        drawModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        drawView.invalidate();
    }

    private void processTouchUp() {
        drawModel.endLine();
    }


}
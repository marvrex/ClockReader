#include <iostream>
#include <opencv2/opencv.hpp>

//if CAMERA is used within the code, replace it with this:
#define CAMERA

using namespace std;
using namespace cv;

vector<int> sortContourIndices(vector<float> areas);
static inline float angleBetweenLinesInRadians(Point2f line1Start, Point2f line1End, Point2f line2Start, Point2f line2End);

int main () {

	Mat img;
	Mat picture;

	VideoCapture capture(0);
	
	// load image if capture is not available
	if (!capture.isOpened()) {
		cout << "Error: capture device not available. Trying to load image file." << endl;
		flush(cout);
		picture = imread("../clock1400.jpg", CV_LOAD_IMAGE_COLOR);
	}

	//blue
	namedWindow("blue", WINDOW_AUTOSIZE);
	int thresholdBlue = 143;
	createTrackbar("Threshold", "blue", &thresholdBlue, 255, 0);

	//endless loop
	for(;;) {
		// load current frame
		if (capture.isOpened())
			capture >> img;
		else
			img = picture.clone();

		//extract RGB channels
		Mat RGB_channel[3];
		split(img, RGB_channel);

		//blue channel
		Mat blueImg = RGB_channel[0];
		threshold(blueImg, blueImg, thresholdBlue, 255, CV_THRESH_BINARY);

		//closing
		erode(blueImg, blueImg, Mat(), Point(-1,-1), 5, 0, morphologyDefaultBorderValue());
		dilate(blueImg, blueImg, Mat(), Point(-1,-1), 5, 0, morphologyDefaultBorderValue());

		//opening
		dilate(blueImg, blueImg, Mat(), Point(-1,-1), 5, 0, morphologyDefaultBorderValue());
		erode(blueImg, blueImg, Mat(), Point(-1,-1), 5, 0, morphologyDefaultBorderValue());

		//find contours on the blue channel
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		findContours(blueImg, contours, hierarchy, RETR_LIST, CV_CHAIN_APPROX_NONE);
		Mat contoursImg = Mat::zeros(blueImg.size(), CV_8UC3);

		if (contours.size() == 0)
			continue;

		//get image moments
		vector<Moments> mu(contours.size() );
		for( int i = 0; i < contours.size(); i++ ){
			mu[i] = moments( contours[i], false );
		}

		//mass center
		vector<Point2f> masscenter( contours.size() );
		for( int i = 0; i < contours.size(); i++ ){ 
			masscenter[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); 
		}

		for (int i = 0; i < masscenter.size(); i++) {
			circle(contoursImg, masscenter[i], 3, Scalar(0, 0, 255), CV_FILLED, 8, 0);
		}

		imshow("blue", blueImg);
		imshow("image", img);

		// calculate contour areas
		vector<float> contourAreas(contours.size());
		for(int i = 0; i < contours.size(); i++) {
			contourAreas[i] = contourArea(contours[i]);
			cout << i << ": " << contourAreas[i] << endl;
		}


		int contourCircle(0), contourTriangle(0), contourHandBig(0), contourHandSmall(0);

		cout << "sorted indices:" << endl;
		vector<int> sortedIndices = sortContourIndices(contourAreas);
		for(int i = 0; i < sortedIndices.size(); i++) {
			cout << i << ": " << sortedIndices[i] << endl;
		}
		cout << endl;
		if (sortedIndices.size() > 3) {
			contourCircle = sortedIndices[sortedIndices.size()-1];
			contourHandBig = sortedIndices[sortedIndices.size()-2];
			contourHandSmall = sortedIndices[sortedIndices.size()-3];
			contourTriangle = sortedIndices[sortedIndices.size()-4];
		}

		// draw contours
		for (int i= 0; i < contours.size(); i++) {
			Scalar color(255,0,255);
			if (i == contourCircle)
				color = Scalar(255,0,0);
			else if (i == contourTriangle)
				color = Scalar(0,255,0);
			else if (i == contourHandBig || i == contourHandSmall)
				color = Scalar(0,0,255);

			drawContours(contoursImg, contours, i, color, 2, 8, noArray());
		}
		

		float angleBigHand = angleBetweenLinesInRadians(
			masscenter[contourCircle], masscenter[contourTriangle],
			masscenter[contourCircle], masscenter[contourHandBig]
			);
		cout << "angleBigHand: " << angleBigHand << endl;

		float angleSmallHand = angleBetweenLinesInRadians(
			masscenter[contourCircle], masscenter[contourTriangle],
			masscenter[contourCircle], masscenter[contourHandSmall]
			);
		cout << "angleSmallHand: " << angleSmallHand << endl;

		int hour = int(angleSmallHand / 360 * 12);
		int minute = int(angleBigHand / 360 * 60);

		ostringstream text;
		text << hour << ":" << minute;
		putText(contoursImg, text.str(), Point(40, 40), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1, 8, false);


		imshow("contours", contoursImg);


		//break with ESC
		if (cvWaitKey(1) == 27)
			break;
	}

	capture.release();
	return 0;
}

// sort by area
struct comparator {
        inline bool operator() (const pair<float, int>& pair1, const pair<float, int>& pair2) {
                return pair1.first < pair2.first;
        }
};

vector<int> sortContourIndices(vector<float> areas) {
	vector<pair<float, int> > areaPairs (areas.size());

	// fill in pairs with values from areas
	for (int i = 0; i < areas.size(); i++) {
		areaPairs[i] = make_pair(areas[i], i);
	}

	sort(areaPairs.begin(), areaPairs.end(), comparator());

	// extract indices
	vector<int> indices(areaPairs.size());
	for (int i = 0; i < areaPairs.size(); i++) {
		indices[i] = areaPairs[i].second;
	}

	return indices;
}

static inline float angleBetweenLinesInRadians(Point2f line1Start, Point2f line1End, Point2f line2Start, Point2f line2End) {
	double angle1 = atan2(line1Start.y-line1End.y, line1Start.x-line1End.x);
	double angle2 = atan2(line2Start.y-line2End.y, line2Start.x-line2End.x);
	double result = (angle2-angle1) * 180 / 3.14;
	if (result<0) {
		result+=360;
	}
	return result;
}
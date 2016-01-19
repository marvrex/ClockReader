#include <iostream>
#include <opencv2/opencv.hpp>

#define VK_ESCAPE 27

using namespace std;
using namespace cv;

vector<int> sortContourIndices(vector<float> areas);
static inline float angleBetweenLinesInRadians(Point2f line1Start, Point2f line1End, Point2f line2Start, Point2f line2End);

void opening(cv::Mat mat)
{
	erode(mat, mat, Mat(), Point(-1,-1), 5, 0, morphologyDefaultBorderValue());
	dilate(mat, mat, Mat(), Point(-1,-1), 5, 0, morphologyDefaultBorderValue());
}

void closing(cv::Mat mat)
{
	dilate(mat, mat, Mat(), Point(-1,-1), 5, 0, morphologyDefaultBorderValue());
	erode(mat, mat, Mat(), Point(-1,-1), 5, 0, morphologyDefaultBorderValue());
}

int main (int argc, char* argv[]) {

	Mat img;
	Mat picture;
	VideoCapture* capture_p(0);
	
	if (argc == 1) {
		capture_p = new VideoCapture(0);
		if (!capture_p->isOpened()) {
			cout << "Failed to open video capture device. Don't want to use the video capture? Load an image from a file by using: " << endl << "$ " << (argv[0] == "" ? "ClockReader" : argv[0]) << " <path-to-image-file>" << endl;
			return 1;
		}
	} else {
		cout << "Loading image from file: " << argv[1] << endl;
		flush(cout);
		picture = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	}

	//blue
	namedWindow("blue", WINDOW_AUTOSIZE);
	int thresholdBlue = 143;
	createTrackbar("Threshold", "blue", &thresholdBlue, 255, 0);

	//endless loop
	for(;;) {
		// load current frame
		if (capture_p != 0) {
			 if (!capture_p->isOpened()) {
				 cout << "Error: capture device not available...";
				 continue;
			 } else
				*capture_p >> img;
		} else
			img = picture.clone();

		// extract RGB channels
		Mat RGB_channel[3];
		split(img, RGB_channel);

		// blue channel
		Mat blueImg = RGB_channel[0];
		
		// apply binary threshold
		threshold(blueImg, blueImg, thresholdBlue, 255, CV_THRESH_BINARY);

		// opening (which is a closing for the other color)
		closing(blueImg);

		// closing (closing black = opening white)
		opening(blueImg);
		
		// show input images
		imshow("blue", blueImg);
		imshow("image", img);

		// find contours on the blue channel
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		// CV_RETR_CCOMP -> two-level hierarchy
		// might need to use CV_RETR_TREE to build a full hierarchy
		findContours(blueImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		Mat contoursImg = Mat::zeros(blueImg.size(), CV_8UC3);

		// there must be at least two contours for a working clock reading
		// -> and even more if it's not 12:00
		if (contours.size() < 2) {
			if (cvWaitKey(1) == VK_ESCAPE)
				break;
			continue;
		}

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

		// draw all centers of mass
		for (int i = 0; i < masscenter.size(); i++) {
			circle(contoursImg, masscenter[i], 3, Scalar(0, 0, 255), CV_FILLED, 8, 0);
		}

		// calculate contour areas
		vector<float> contourAreas(contours.size());
		for(int i = 0; i < contours.size(); i++) {
			contourAreas[i] = contourArea(contours[i]);
		}

		// sort contours by area
		vector<int> sortedIndices = sortContourIndices(contourAreas);
		
		int contourCircle(-1), contourTriangle(-1), contourHandBig(-1), contourHandSmall(-1);
		
		// the biggest contour is the circle -- the background of the clock
		contourCircle = sortedIndices[sortedIndices.size()-1];
		// every other contour on the clock has to be a children of this contour
		// so: find all contours with the circle as their parent
		vector<int> childContours;
		for (int i = 0; i < sortedIndices.size()-1; i++) {
			if (hierarchy[sortedIndices[i]][3] == contourCircle)
				childContours.push_back(sortedIndices[i]);
		}
		
		// the children are still sorted by size
		// the two biggest contours are most likely the hands of the clock
		if (childContours.size() > 0)
			contourHandBig = childContours[childContours.size()-1];
		
		if (childContours.size() > 1)
			contourHandSmall = childContours[childContours.size()-2];
		else
			// there is only one child contour
			contourHandSmall = contourHandBig; // 12:00
		
		// the contour that is farther from the center of the circle than the big hand of the clock has to be the triangle. If there is no such contour then the big hand shadows the triangle
		if (childContours.size() > 1) {
			Point2f centerCircle = masscenter[contourCircle];
			Point2f centerBig = masscenter[contourHandBig];
			double distBig = norm(centerCircle - centerBig);
			// the triangle is one of the three biggest child contours
			for (int i = childContours.size()-2; i >= 0 && i >= childContours.size()-3; i--) {
				Point2f center = masscenter[childContours[i]];
				float dist = norm(center - centerCircle);
				if (dist > distBig) {
					contourTriangle = childContours[i];
				}
			}
		}
		// no contour is farther from the center than the big hand
		if (contourTriangle == -1)
			contourTriangle = contourHandBig;
		// if the second largest contour is also the farthest contour
		else if (contourTriangle == contourHandSmall) // 13:05, 14:10, ...
			contourHandSmall = contourHandBig;

		// draw the contours of the clock
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
// 		cout << "angleBigHand: " << angleBigHand << endl;

		float angleSmallHand = angleBetweenLinesInRadians(
			masscenter[contourCircle], masscenter[contourTriangle],
			masscenter[contourCircle], masscenter[contourHandSmall]
			);
// 		cout << "angleSmallHand: " << angleSmallHand << endl;

		int hour = int(angleSmallHand / 360 * 12);
		int minute = int(angleBigHand / 360 * 60);

		ostringstream text;
		text << hour << ":" << minute;
		putText(contoursImg, text.str(), Point(40, 40), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1, 8, false);


		imshow("contours", contoursImg);


		//break with ESC
		if (cvWaitKey(1) == VK_ESCAPE)
			break;
	}
	
	if (capture_p != 0) {
		capture_p->release();
		delete capture_p;
	}
	return 0;
}

// sort by area
struct comparatorArea {
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

	sort(areaPairs.begin(), areaPairs.end(), comparatorArea());

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
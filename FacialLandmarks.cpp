#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

struct FaceLandmarksModelLoader
{
    static constexpr double kEar = 0.3;
    static constexpr double kMouth = 0.7;

    cv::CascadeClassifier face_cascade;
    cv::Ptr<cv::face::Facemark> facemark;
};

struct Img
{
    static constexpr double kTargetWidth = 160.0;

    cv::Mat image;
    cv::Mat mask;
};

struct VideoCaptureReader
{
    cv::VideoCapture cap;

    static constexpr double kWidth = 320.0;
    static constexpr double kHeight = 240.0;
    
    cv::Mat frame;
    cv::Mat flipped;
    cv::Mat gray;
};

void DrawLandmarks(cv::Mat &, std::vector<cv::Point2f> &, int, int, cv::Scalar &, bool = true);

double distanceCalculate(double, double, double, double);

void LoadModels(FaceLandmarksModelLoader&);

void ReadImg(Img&);

void InitReader(VideoCaptureReader&);

void ReadFrame(VideoCaptureReader&);

double ComputeAspectRatio(std::vector<cv::Point2f> &, int, int, bool = false);

void RenderAll(Img&, VideoCaptureReader &r, FaceLandmarksModelLoader &loader);

int main()
{
    Img logo;
    ReadImg(logo);

    FaceLandmarksModelLoader loader;
    LoadModels(loader);

    VideoCaptureReader webcam;
    InitReader(webcam);

    for (;;)
    {
        ReadFrame(webcam);



        if (cv::waitKey(10) >= 0)
            break;

        RenderAll(logo, webcam, loader);
    }
}

void ResizeImg(cv::Mat &src, cv::Mat &dst, double scale_factor)
{
    double height = static_cast<double>(src.rows);
    double target_height = scale_factor * height;

    int w = static_cast<int>(Img::kTargetWidth);
    int h = static_cast<int>(target_height);

    cv::Size dsize{ w, h };

    cv::resize(src, dst, dsize);
}

void DrawLandmarks(cv::Mat &img, std::vector<cv::Point2f> &lm, int start, int end, cv::Scalar& sc, bool is_closed)
{
    std::vector<cv::Point> pt;

    for (size_t i = start; i <= end; i++)
    {
        pt.push_back(cv::Point{static_cast<int>(lm[i].x * 2), static_cast<int>(lm[i].y * 2)});
    }

    cv::polylines(img, pt, is_closed, sc, 1);
}

double distanceCalculate(double x1, double y1, double x2, double y2)
{
    double x = x1 - x2;
    double y = y1 - y2;
    double dist;

    dist = pow(x, 2) + pow(y, 2);
    dist = sqrt(dist);

    return dist;
}

void LoadModels(FaceLandmarksModelLoader &loader)
{
    loader.face_cascade = cv::CascadeClassifier{"haarcascade_frontalface_alt2.xml"};
    loader.facemark = cv::face::FacemarkLBF::create();
    loader.facemark->loadModel("lbfmodel.yaml");
}

void ReadImg(Img &img)
{
    cv::Mat output_image = cv::imread("TTTechAuto_Logo_RGB_large.png", cv::IMREAD_UNCHANGED);

    double img_width = static_cast<double>(output_image.cols);

    double scale_factor = Img::kTargetWidth / img_width;

    cv::Mat resized;
    ResizeImg(output_image, resized, scale_factor);

    std::vector<cv::Mat> channels;
    cv::split(resized, channels);

    std::vector<cv::Mat> rgb{ channels[0], channels[1], channels[2] };
    cv::merge(rgb, img.image);
    img.mask = channels[3];
}

void InitReader(VideoCaptureReader &r)
{
    r.cap = cv::VideoCapture{ 0 };
    r.cap.set(cv::CAP_PROP_FRAME_WIDTH, VideoCaptureReader::kWidth);
    r.cap.set(cv::CAP_PROP_FRAME_HEIGHT, VideoCaptureReader::kHeight);
}

void ReadFrame(VideoCaptureReader &r)
{
    r.cap >> r.frame;
    cv::flip(r.frame, r.flipped, 1);
    cv::cvtColor(r.flipped, r.gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(r.gray, r.gray);
}

double ComputeAspectRatio(std::vector<cv::Point2f>& keypoints, int start, int end, bool is_mouth)
{
    std::vector<cv::Point2f> pts;
    for (size_t i = start; i <= end; i++)
    {
        pts.push_back(cv::Point2f{ keypoints[i].x, keypoints[i].y });
    }

    double ear = 0.0;

    if (is_mouth)
    {
        double a = distanceCalculate(pts[1].x, pts[1].y, pts[11].x, pts[11].y);
        double b = distanceCalculate(pts[2].x, pts[2].y, pts[10].x, pts[10].y);
        double c = distanceCalculate(pts[3].x, pts[3].y, pts[9].x, pts[9].y);
        double d = distanceCalculate(pts[4].x, pts[4].y, pts[8].x, pts[8].y);
        double e = distanceCalculate(pts[5].x, pts[5].y, pts[7].x, pts[7].y);
        double f = distanceCalculate(pts[0].x, pts[0].y, pts[6].x, pts[6].y);

        ear = (a + b + c + d + e) / (2.0 * f);
    }
    else
    {
        double a = distanceCalculate(pts[1].x, pts[1].y, pts[5].x, pts[5].y);
        double b = distanceCalculate(pts[2].x, pts[2].y, pts[4].x, pts[4].y);
        double c = distanceCalculate(pts[0].x, pts[0].y, pts[3].x, pts[3].y);

        ear = (a + b) / (2.0 * c);
    }

    return ear;
}

void RenderAll(Img &img, VideoCaptureReader &r, FaceLandmarksModelLoader &loader)
{
    const int kBgWidth{ 640 };
    const int kBgHeight{ 480 };

    cv::Mat bg{ kBgHeight, kBgWidth, CV_8UC3, cv::Scalar{0, 0, 0} };


    //
    
    std::vector<cv::Rect> faces;
    loader.face_cascade.detectMultiScale(r.gray, faces);

    std::vector<std::vector<cv::Point2f>> shapes;
    
    bool success = loader.facemark->fit(r.gray, faces, shapes);

    if (success)
    {
        for (size_t i = 0; i < shapes.size(); i++)
        {
            if (shapes[i].size() == 68)
            {
                double left_ear = ComputeAspectRatio(shapes[i], 36, 41);
                double right_ear = ComputeAspectRatio(shapes[i], 42, 47);
                double average_ear = (left_ear + right_ear) / 2;
                // std::cout << average_ear << std::endl;
                double mear = ComputeAspectRatio(shapes[i], 48, 59, true);

                if (average_ear < FaceLandmarksModelLoader::kEar)
                {
                    cv::Scalar sc{ 0, 255, 255 };
                    DrawLandmarks(bg, shapes[i], 36, 41, sc);
                    DrawLandmarks(bg, shapes[i], 42, 47, sc);
                }
                else
                {
                    cv::Scalar sc{ 255, 255, 255 };
                    DrawLandmarks(bg, shapes[i], 36, 41, sc);
                    DrawLandmarks(bg, shapes[i], 42, 47, sc);
                }

                if (mear > FaceLandmarksModelLoader::kMouth)
                {
                    cv::Scalar sc{ 0, 255, 255 };
                    DrawLandmarks(bg, shapes[i], 48, 59, sc);
                    DrawLandmarks(bg, shapes[i], 60, 67, sc);
                }
                else
                {
                    cv::Scalar sc{ 255, 255, 255 };
                    DrawLandmarks(bg, shapes[i], 48, 59, sc);
                    DrawLandmarks(bg, shapes[i], 60, 67, sc);
                }

                cv::Scalar sc{ 255, 255, 255 };
                DrawLandmarks(bg, shapes[i], 27, 30, sc, false);
                DrawLandmarks(bg, shapes[i], 30, 35, sc);

                DrawLandmarks(bg, shapes[i], 17, 21, sc, false);
                DrawLandmarks(bg, shapes[i], 22, 26, sc, false);

                DrawLandmarks(bg, shapes[i], 0, 16, sc, false);
            }
        }
    }

    //


    cv::Mat roi = bg(cv::Rect{kBgWidth - img.image.cols, kBgHeight - img.image.rows, img.image.cols, img.image.rows});

    img.image.copyTo(roi, img.mask);

    cv::imshow("webcam", bg);
}

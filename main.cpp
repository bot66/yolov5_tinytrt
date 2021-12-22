#include "Trt.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <dirent.h>
#include <time.h>
#include <vector>
#define N 1
#define W 640
#define H 640
#define C 3
#define CLASS_NUM 80
#define IOU_THRESH 0.45f
#define CONF_THRESH 0.25f

using namespace std;
using namespace cv;


struct Object
{
    Rect_<float> rect;
    int label;
    float conf;
    Object(Rect_<float> rect, int label, float conf):rect(rect),label(label),conf(conf){};
    Object(){};
    ~Object(){};

};


void quickSortObjects(vector<Object> &objects, int left, int right)
{
    Object pivot=objects[left];
    int i=left;
    int j=right;
    if (left < right)
    {
        while(i<j)
        {
            while(i<j && objects[j].conf <= pivot.conf)
            {
                --j;
            }
            objects[i]=objects[j];
            while (i<j && objects[i].conf >= pivot.conf)
            {
                ++i;
            }
            objects[j]=objects[i];   
        }
        objects[i]=pivot;
        quickSortObjects(objects,left,i-1);
        quickSortObjects(objects,i+1,right);
    }
}

vector<string>readFolder(const string &image_path)
{
    vector<string> image_names;
    auto dir = opendir(image_path.c_str());
 
    if ((dir) != NULL)
    {
        struct dirent *entry;
        entry = readdir(dir);
        while (entry)
        {
            auto temp = image_path +"/" +  entry->d_name;
            if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            {
                entry = readdir(dir);
                continue;
            }
            image_names.push_back(temp);
            entry = readdir(dir);
        }
    }
    return image_names;
}

void preprocess(Mat &image)
{
    cvtColor(image,image,COLOR_BGR2RGB);
    float imageWidth= image.cols;
    float imageHeight=image.rows;
    float r = min(H/imageHeight,W/imageWidth);
    float unpadW=round(imageWidth*r);
    float unpadH=round(imageHeight*r);
    float padW=W-unpadW;
    float padH=H-unpadH;
    int top=round(padH/2-0.1), bottom =round(padH/2+0.1);
    int left=round(padW/2-0.1), right=round(padW/2+0.1);

    resize(image, image, Size(unpadW, unpadH));
    copyMakeBorder(image,image,top,bottom,left,right,BORDER_CONSTANT,Scalar(114,114,114));

    assert(image.cols==W && image.rows==H);

    image.convertTo(image, CV_32FC3, 1.0 / 255); //div 255
  
    vector<float> mean_value{0, 0, 0}; //RGB
    vector<float> std_value{1,1,1};

    vector<Mat> rgbChannels(3);
    split(image, rgbChannels);
    for (size_t i = 0; i < rgbChannels.size(); i++)
    {
        rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0 / std_value[i], (0.0 - mean_value[i]) / std_value[i]);
    }
    merge(rgbChannels, image);

}

vector<float> prepareInput(vector<Mat> &vec_img) {

    vector<float> result(N*C*H*W);
    float *data = result.data();
    int index = 0;
    for (const Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        //HWC TO CHW
        int channelLength = W*H;

        vector<Mat> split_img = {
                Mat(H, W, CV_32FC1, data + channelLength * index),   //R
                Mat(H, W, CV_32FC1, data + channelLength * (index + 1)), //G
                Mat(H, W, CV_32FC1, data + channelLength * (index + 2)) //B    
        };
        index += 3;
        split(src_img, split_img);	
    }
    return result;	
}

void nonMaxSupression( vector<float> &output, vector<Object> &results,float conf_thresh, float iou_thresh)
{
    //Box (center x, center y, width, height)
    int step=5+CLASS_NUM;
    vector<Object> temp;
    for (auto i=output.begin();i!=output.end(); i+=step)
    {
        float pObj=*(i+4);
        auto maxProbPos=max_element(i+5,i+step-1);
        int label=maxProbPos-i;
        label-=5;
        float conf=(*maxProbPos) * pObj;
        if (conf>conf_thresh)
        {
            temp.emplace_back(Rect_<float>(*i - (*(i+2))/2,
                                            *(i+1)-(*(i+3))/2,
                                            *(i+2),
                                            *(i+3)),label,conf);    // conf = obj_conf * cls_conf

        }
    }
    if (temp.size()==0)
        return;
     //sort highest to lowest by confidence
    quickSortObjects(temp,0,temp.size()-1);

    for (size_t i=0;i<temp.size();++i)
    {
        int keep=1;
        for (size_t j=0;j<results.size();++j)
        {
            float intersectionArea=((results[j].rect)&(temp[i].rect)).area();
            float unionArea=temp[i].rect.area()+results[j].rect.area()-intersectionArea;
            if (intersectionArea/unionArea > iou_thresh)
            {
                keep=0;
            }
        }
        if (keep)
            results.push_back(temp[i]);
    }


}

void drawObjects(Mat& bgr, const vector<Object>& objects, const string &savePath)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    float imageWidth= bgr.cols;
    float imageHeight=bgr.rows;
    float r = min(H/imageHeight,W/imageWidth);
    float padW=(W-(r*bgr.cols))/2;
    float padH=(H-(r*bgr.rows))/2;


    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object &obj = objects[i];

        float x=(obj.rect.x-padW)<0?0:(obj.rect.x-padW);
        x/=r;
        float y=(obj.rect.y-padH)<0?0:(obj.rect.y-padH);
        y/=r;
        float w=obj.rect.width/r;
        float h=obj.rect.height/r;

        w =(w+x)>(imageWidth-1)?(imageWidth-x-1):w;
        h =(h+y)>(imageHeight-1)?(imageHeight-y-1):h;        
        
        cout <<" class: " << class_names[obj.label] 
             <<" xywh: " << x <<"," << y << "," << w << "," << h << endl;
        rectangle(bgr, Rect_<float>(x,y,w,h), Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.conf * 100);

        int baseLine = 0;
        Size label_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        y=y - label_size.height - baseLine;
        if(y<0)
            y=0;
        rectangle(bgr, Rect(Point(x, y), Size(label_size.width, label_size.height + baseLine)),
                      Scalar(255, 255, 255), -1);

        putText(bgr, text, Point(x, y + label_size.height),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
    }
    imwrite(savePath,bgr);
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        cout << "usage:" << "./yolov5_tinytrt <onnx_model> <engine_model> <input_folder> <output_folder> " << endl;
        return 1;
    }
    Trt trt;
    trt.CreateEngine(argv[1],argv[2],1,0);

    vector<string> imagePaths=readFolder(argv[3]);
    for (auto p:imagePaths)
    {
        cout <<"Image:" <<p<< endl;
        Mat image=imread(p);
        Mat drawImage=image.clone();
        preprocess(image);
        vector<Mat> images;
        images.push_back(image);
        auto inputs= prepareInput(images);

        vector<float> output;
        int outputIndex=4;
        trt.CopyFromHostToDevice(inputs,0);
        double startTime = clock();
        trt.Forward();    
        double endTime = clock();
        cout <<"Forward time:" <<(double)(endTime - startTime)*1000/CLOCKS_PER_SEC << "ms" << endl; 
        trt.CopyFromDeviceToHost(output, outputIndex); 

        vector<Object> objects;
        nonMaxSupression(output,objects,CONF_THRESH,IOU_THRESH);

        string savePath=argv[4];
        savePath=savePath + "/"+p.substr(p.find("/")+1);
        drawObjects(drawImage,objects,savePath);
    }

    return 0;

}

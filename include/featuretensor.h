#ifndef FEATURETENSOR_H
#define FEATURETENSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
//#include <NvInfer.h>
//#include <NvOnnxParser.h>
#include "model.hpp"
#include "datatype.h"
//#include "cuda_runtime_api.h"

using std::vector;
//using nvinfer1::ILogger;

class FeatureTensor {
public:
    //FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const int featureDim, int gpuID, ILogger* gLogger);
    FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const int featureDim, int gpuID);

    ~FeatureTensor();

public:
    bool getRectsFeature(const cv::Mat& img, DETECTIONS& det);
    bool getRectsFeature(DETECTIONS& det);
    void loadOnnx(std::string onnxPath);
    //int getResult(float*& buffer);
    cv::Mat doInference(vector<cv::Mat>& imgMats);

private:
    cv::Mat doInference_run(vector<cv::Mat> imgMats);
    void stream2det(cv::Mat stream, DETECTIONS& det);

private:
    //nvinfer1::IRuntime* runtime;
    //nvinfer1::ICudaEngine* engine;
    //nvinfer1::IExecutionContext* context;
    cv::dnn::Net dnn_engine;
    const int maxBatchSize;
    const cv::Size imgShape;
    const int featureDim;

private:
    int curBatchSize;
    //const int inputStreamSize, outputStreamSize;
    bool initFlag;
    cv::Mat outputBuffer;
    //float* const inputBuffer;
    //float* const outputBuffer;
    int inputIndex, outputIndex;
    void* buffers[2];
    //cudaStream_t cudaStream;
    // BGR format
    float means[3], std[3];
    const std::string inputName, outputName;
    //ILogger* gLogger;
};

#endif

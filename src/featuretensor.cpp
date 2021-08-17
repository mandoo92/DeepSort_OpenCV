#include "featuretensor.h"
#include <fstream>

//using namespace nvinfer1;

#define INPUTSTREAM_SIZE (maxBatchSize*3*imgShape.area())
#define OUTPUTSTREAM_SIZE (maxBatchSize*featureDim)

FeatureTensor::FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const int featureDim, int gpuID) 
        : maxBatchSize(maxBatchSize), imgShape(imgShape), featureDim(featureDim), 
        inputName("input"), outputName("output") {
    //cudaSetDevice(gpuID);
    //this->gLogger = gLogger;
    //runtime = nullptr;
    //engine = nullptr;
    //context = nullptr; 

    means[0] = 0.485, means[1] = 0.456, means[2] = 0.406;
    std[0] = 0.229, std[1] = 0.224, std[2] = 0.225;

    initFlag = false;
}

FeatureTensor::~FeatureTensor() {
    if (initFlag) {
        // cudaStreamSynchronize(cudaStream);
        //cudaStreamDestroy(cudaStream);
        //cudaFree(buffers[inputIndex]);
        //cudaFree(buffers[outputIndex]);
    }
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& det) {
    std::vector<cv::Mat> mats;
    for (auto& dbox : det) {
        cv::Rect rect = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                                 int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
        rect.width = rect.height * 0.5;
        rect.x = (rect.x >= 0 ? rect.x : 0);
        rect.y = (rect.y >= 0 ? rect.y : 0);
        rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y));
        cv::Mat tempMat = img(rect).clone();
        cv::resize(tempMat, tempMat, imgShape);
        mats.push_back(tempMat);
    }
    cv::Mat out = doInference(mats);
    // decode output to det
    stream2det(out, det);
    return true;
}

bool FeatureTensor::getRectsFeature(DETECTIONS& det) {
    return true;
}

void FeatureTensor::loadOnnx(std::string onnxPath) {
    //dnn_engine = new cv::dnn::Net();

    //dnn_engine.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //dnn_engine.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    //static auto _engine = cv::dnn::readNetFromONNX(onnxPath);
    dnn_engine = cv::dnn::readNetFromONNX(onnxPath);
}

//int FeatureTensor::getResult(float*& buffer) {
//    if (buffer != nullptr)
//        delete buffer;
//    int curStreamSize = curBatchSize*featureDim;
//    buffer = new float[curStreamSize];
//    for (int i = 0; i < curStreamSize; ++i) {
//        buffer[i] = outputBuffer[i];
//    }
//    return curStreamSize;
//}

cv::Mat FeatureTensor::doInference(vector<cv::Mat>& imgMats) {
    cv::Mat out;
    int mat_size = imgMats.size();
    if (mat_size > 0) {
        if (mat_size != 32) {
            for (size_t i = 0; i < 32 - mat_size; ++i)
                imgMats.push_back(cv::Mat(cv::Size(64, 128), CV_8UC3, cv::Scalar(255, 255, 255)));
        }

        out = doInference_run(imgMats);
    }

    return out;
}


cv::Mat FeatureTensor::doInference_run(vector<cv::Mat> imgMats) {
    //cudaMemcpyAsync(buffers[inputIndex], inputBuffer, inputStreamSize * sizeof(float), cudaMemcpyHostToDevice, cudaStream);
    //Dims4 inputDims{curBatchSize, 3, imgShape.height, imgShape.width};
    //context->setBindingDimensions(0, inputDims);
    //
    //context->enqueueV2(buffers, cudaStream, nullptr);
    //cudaMemcpyAsync(outputBuffer, buffers[outputIndex], outputStreamSize * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
    // cudaStreamSynchronize(cudaStream);
    auto blob = cv::dnn::blobFromImages(imgMats, 0.225, cv::Size(64, 128), cv::Scalar(0.485, 0.456, 0.406), true);

    dnn_engine.setInput(blob, "input");
    cv::Mat out = dnn_engine.forward(this->outputName);

    return out;
}

void FeatureTensor::stream2det(cv::Mat stream, DETECTIONS& det) {
    int i = 0;
    for (DETECTION_ROW& dbox : det) {
        for (int j = 0; j < featureDim; ++j) {
            float data = stream.at<float>(j, i);
            dbox.feature[j] = data;
        }
        i++;
    }
}

/*
    This file is part of EqVIO.

    EqVIO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqVIO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqVIO.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "eqvio/LoopTimer.h"
#include "eqvio/VIOFilter.h"
#include "eqvio/VIOFilterSettings.h"
#include "eqvio/VIOSimulator.h"
#include "eqvio/VIOWriter.h"
#include "eqvio/dataserver/dataservers.h"

#include "GIFT/KeyPointFeatureTracker.h"
#include "GIFT/PointFeatureTracker.h"
#include "GIFT/Visualisation.h"
#include "opencv2/highgui/highgui.hpp"

#include "yaml-cpp/yaml.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

#include "argparse/argparse.hpp"

#include "eqvio/VIOVisualiser.h"

VisionMeasurement convertGIFTFeaturesRGBD(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp, const cv::Mat& depthImage);
VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp);

int main(int argc, char const* argv[]) {
    // TODO: Add options for writing state and images
    argparse::ArgumentParser program("EqVIO standalone");
    program.add_argument("-d", "--display")
        .help("Display a live plot of the system state")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--mode").default_value(std::string("asl")).action([](const std::string& value) {
        static const std::vector<std::string> choices = {"asl", "uzhfpv", "anu", "ros", "hilti"};
        if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw std::invalid_argument("Invalid mode requested.");
    });
    program.add_argument("--depthFlag")
        .help("measure depth or not")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--simvis")
        .help("Simulate the vision measurements from ground truth.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--simimu")
        .help("Simulate the IMU measurements from ground truth.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("dataset").help("The name of the dataset.");
    program.add_argument("config").help("The yaml file with the filter and feature tracker configuration.");
    program.add_argument("-o", "--output")
        .help("Directory for storing the filter output files.")
        .default_value(std::string("-"));
    program.add_argument("-c", "--camera")
        .help("File containing camera intrinsics and extrinsics.")
        .default_value(std::string("-"));
    program.add_argument("--timing")
        .help("Record the processing time information.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--start")
        .help("Start processing a number of seconds after the start of the data.")
        .default_value(0.0)
        .scan<'g', double>();
    program.add_argument("--stop")
        .help("Stop processing a number of seconds after the start of the data.")
        .default_value(-1.0)
        .scan<'g', double>();
    program.add_argument("--limitRate")
        .help("Limit the playback rate of the data. Default 0.0 (no limit).")
        .default_value(0.0)
        .scan<'g', double>();

    // Read the arguments
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cout << err.what() << '\n' << program;
        exit(0);
    }

    const std::string configFileName = program.get("config");
    YAML::Node eqvioConfig = YAML::LoadFile(configFileName);
    bool writeStateFlag = configOrDefault(eqvioConfig, "main:writeState", true);
    const double cameraLag = configOrDefault(eqvioConfig, "main:cameraLag", 0.0);

    const bool displayFlag = program.get<bool>("--display");
    const bool simvisFlag = program.get<bool>("--simvis");
    const bool simimuFlag = program.get<bool>("--simimu");
    const bool writeTimeFlag = program.get<bool>("--timing");
    const bool depthFlag = program.get<bool>("--depthFlag");
    
    if (depthFlag) {
    std::cout << "Using depth" << std::endl;

    } else {
        std::cout << "Not using depth" << std::endl;
    }

    std::unique_ptr<DataServerBase> dataServer;
    dataServer = std::make_unique<ThreadedDataServer>(
        createDatasetReader(program.get("--mode"), program.get("dataset"), cameraLag), eqvioConfig["sim"]);

    // Read the camera file if provided
    if (program.is_used("--camera")) {
        dataServer->readCamera(program.get<std::string>("--camera"));
    }

    
    // Set up the timer.
    loopTimer.initialise(
        {"correction", "features", "preprocessing", "propagation", "total", "total vision update", "write output"});

    const double limitRateSetting = program.get<double>("--limitRate");
    double startTime = program.get<double>("--start");
    double stopTime = program.get<double>("--stop");
    {
        double initialDataTime = dataServer->nextTime();
        startTime = startTime > 0 ? initialDataTime + startTime : startTime;
        stopTime = stopTime > 0 ? initialDataTime + stopTime : stopTime;
    }

    // Read I/O settings

    if (displayFlag && !EQVIO_BUILD_VISUALISATION) {
        std::cout << "Visualisations have been requested, but the necessary module is not set to be built."
                  << std::endl;
    }
    VIOVisualiser visualiser(displayFlag);

    
    // Initialise the filter
    VIOFilter::Settings filterSettings(eqvioConfig["eqf"]);
    if (dataServer->cameraExtrinsics()) {
        std::cout << "Camera extrinsics were provided by the dataset, overriding those in the filter settings."
                  << std::endl;
        filterSettings.cameraOffset = *dataServer->cameraExtrinsics();
    }
    std::cout << "The camera extrinsics (pose of the camera w.r.t. the IMU) are:\n"
              << filterSettings.cameraOffset.asMatrix() << std::endl;
    VIOFilter filter(filterSettings);
    ////////////////////////////////////////////////////////////////////////////////////
    // Initialise the feature tracker
    GIFT::PointFeatureTracker featureTracker = GIFT::PointFeatureTracker(dataServer->camera());
    // GIFT::KeyPointFeatureTracker featureTracker = GIFT::KeyPointFeatureTracker(dataServer->camera);
    featureTracker.settings.configure(eqvioConfig["GIFT"]);

    // Set up output files

    std::stringstream outputFileNameStream, internalFileNameStream;
    std::time_t t0 = std::time(nullptr);
    if (program.is_used("--output")) {
        outputFileNameStream << program.get<std::string>("--output");
        if (!writeStateFlag) {
            writeStateFlag = true;
            std::cout
                << "User provided an output file, so the output writing flag from the configuration will be ignored."
                << std::endl;
        }
    } else {
        if (depthFlag)
        {
            outputFileNameStream << "EQVIO_depthoutput_" << std::put_time(std::localtime(&t0), "%F_%T") << "/";
        }else{
            outputFileNameStream << "EQVIO_output_" << std::put_time(std::localtime(&t0), "%F_%T") << "/";
        }
    }
    VIOWriter vioWriter(outputFileNameStream.str());

    
    int imuDataCounter = 0, visionDataCounter = 0;
    const std::chrono::steady_clock::time_point loopStartTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point rateLimitTimer = std::chrono::steady_clock::now();

    // Add a new variable to store the latest depth image
    cv::Mat latestDepthImage;

    while (true) {
        MeasurementType measType = dataServer->nextMeasurementType();

        if (measType == MeasurementType::None) {
            // End of data
            break;
        }


        if (measType == MeasurementType::DepthImage) {
            // Process depth image 
            StampedDepthImage imageDepth;
            imageDepth = dataServer->getDepthImage();

            //record current depth image
            latestDepthImage = imageDepth.depthImage;
            // int latest_depth = latestDepthImage.depth();

            if (startTime > 0 && imageDepth.stamp < startTime) {
                continue;
            }
            
            std::cout << "Read depth image with timestamp: " << imageDepth.stamp << std::endl;
            // std::cout << "Read depth image with depth: " << latestDepthImage << std::endl;
        }   
        else if (measType == MeasurementType::Image) {
            loopTimer.startLoop();

            StampedImage imageData;
            VisionMeasurement measData;

            loopTimer.startTiming("total");

            if (simvisFlag) {
                measData = dataServer->getSimVision();
                if (startTime > 0 && measData.stamp < startTime) {
                    continue;
                }
            } else {
                imageData = dataServer->getImage();
                if (startTime > 0 && imageData.stamp < startTime) {
                    continue;
                }
                loopTimer.startTiming("features");
                const VisionMeasurement& featurePrediction =
                    filter.getFeaturePredictions(dataServer->camera(), imageData.stamp);
                featureTracker.processImage(imageData.image, featurePrediction.ocvCoordinates());
                //use depth flag to process feature for rgbd and rgb differently
                if(depthFlag){
                    measData = convertGIFTFeaturesRGBD(featureTracker.outputFeatures(), imageData.stamp, latestDepthImage);
                }else{
                    measData = convertGIFTFeatures(featureTracker.outputFeatures(), imageData.stamp);
                }
                
                
                measData.cameraPtr = dataServer->camera();
                loopTimer.endTiming("features");
            }

            loopTimer.startTiming("total vision update");
            //process data differently depend on depthflag
            if (depthFlag){
                filter.processVisionDataRGBD(measData);
            }else{
                filter.processVisionDataRGB(measData);
            }
            

            loopTimer.endTiming("total vision update");
            loopTimer.endTiming("total");
            ++visionDataCounter;

            if (displayFlag && !simvisFlag) {
                visualiser.displayFeatureImage(featureTracker.outputFeatures(), imageData.image);
            }

            // Output filter data
            loopTimer.startTiming("write output");
            VIOState estimatedState = filter.stateEstimate();
            if (writeStateFlag) {
                vioWriter.writeStates(filter.getTime(), estimatedState);
                vioWriter.writeFeatures(measData);
            }

            loopTimer.endTiming("write output");
            if (writeTimeFlag) {
                vioWriter.writeTiming(loopTimer.getLoopTimingData());
            }

            // Optionally visualise the filter data
            if (displayFlag) {
                visualiser.updateMapDisplay(estimatedState);
            }

            // Limit the loop rate
            if (limitRateSetting > 0) {
                std::this_thread::sleep_until(rateLimitTimer + std::chrono::duration<double>(1.0 / limitRateSetting));
                rateLimitTimer = std::chrono::steady_clock::now();
            }

        } else if (measType == MeasurementType::IMU) {
            IMUVelocity imuData;
            if (simimuFlag) {
                imuData = dataServer->getSimIMU();
            } else {
                imuData = dataServer->getIMU();
            }
            assert(!imuData.acc.hasNaN());
            assert(!imuData.gyr.hasNaN());
            assert(imuData.stamp > 0);

            if (startTime > 0 && imuData.stamp < startTime) {
                continue;
            }

            filter.processIMUData(imuData);
            ++imuDataCounter;

        }

        if (stopTime > 0 && filter.getTime() > stopTime) {
            break;
        }
    }

    const auto elapsedTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - loopStartTime);
    std::cout << "Processed " << imuDataCounter << " IMU and " << visionDataCounter << " vision measurements.\n"
              << "Time taken: " << elapsedTime.count() * 1e-3 << " seconds." << std::endl;

    return 0;
}
 

// VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp) {
VisionMeasurement convertGIFTFeaturesRGBD(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp, const cv::Mat& depthImage) {
    VisionMeasurement measurement;
    measurement.stamp = stamp;

    for (const GIFT::Feature& f : GIFTFeatures) {
    // Get the feature's image coordinates
    cv::Point2f pt = f.camCoordinates;
    // Check if the point is within the depth image
    if (pt.x >= 0 && pt.x < depthImage.cols && pt.y >= 0 && pt.y < depthImage.rows) {
        // Round and convert coordinates to integer
        int x = static_cast<int>(std::round(pt.x));
        int y = static_cast<int>(std::round(pt.y));
        // Get the depth at the feature's image coordinates
        uint16_t depthInMillimeters = depthImage.at<uint16_t>(y, x);
        float depthInMeters = static_cast<float>(depthInMillimeters) / 1000.0f; // adjust the scaling factor as necessary

        // Add the feature to the measurement, including its depth
        measurement.camCoordinates[f.idNumber] = Eigen::Vector2d(f.camCoordinatesEigen().x(), f.camCoordinatesEigen().y());
        measurement.depthValue[f.idNumber] = depthInMeters;
        // std::cout << "depth is " << depthInMeters << std::endl;
        // std::cout << "Depth image type: " << depthImage.type() << std::endl;

    }
    }
    return measurement;

}

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp) {
    VisionMeasurement measurement;
    measurement.stamp = stamp;
    std::transform(
        GIFTFeatures.begin(), GIFTFeatures.end(),
        std::inserter(measurement.camCoordinates, measurement.camCoordinates.begin()),
        [](const GIFT::Feature& f) { return std::make_pair(f.idNumber, f.camCoordinatesEigen()); });
    return measurement;
}
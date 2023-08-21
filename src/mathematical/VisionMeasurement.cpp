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

#include "eqvio/mathematical/VisionMeasurement.h"
#include "eqvio/mathematical/VIOState.h"

using namespace std;
using namespace Eigen;

std::vector<int> VisionMeasurement::getIds() const {
    std::vector<int> ids(camCoordinates.size());
    transform(camCoordinates.begin(), camCoordinates.end(), ids.begin(), [](const auto& cc) { return cc.first; });
    return ids;
}

std::map<int, cv::Point2f> VisionMeasurement::ocvCoordinates() const {
    std::map<int, cv::Point2f> ocvPoints;
    for (const auto& [id, pt] : camCoordinates) {
        ocvPoints[id] = cv::Point2f(pt.x(), pt.y());
    }
    return ocvPoints;
}


// void VIO_eqf::removeLandmarkByIndex(const int& idx) {
//     xi0.cameraLandmarks.erase(xi0.cameraLandmarks.begin() + idx);
//     X.id.erase(X.id.begin() + idx);
//     X.Q.erase(X.Q.begin() + idx);
//     removeRows(Sigma, VIOSensorState::CompDim + 3 * idx, 3);
//     removeCols(Sigma, VIOSensorState::CompDim + 3 * idx, 3);
// }

// void VIO_eqf::removeLandmarkById(const int& id) {
//     const auto it = find_if(
//         xi0.cameraLandmarks.begin(), xi0.cameraLandmarks.end(), [&id](const Landmark& lm) { return lm.id == id; });
//     assert(it != xi0.cameraLandmarks.end());
//     const int idx = distance(xi0.cameraLandmarks.begin(), it);
//     removeLandmarkByIndex(idx);
// }

// Used to remove measurement pairs with a specific id
void VisionMeasurement::removeEntryById(const int& id) {
    // Erase the entry with id from camCoordinates
    auto it_cam = camCoordinates.find(id);
    if (it_cam != camCoordinates.end())
        camCoordinates.erase(it_cam);

    // Erase the entry with id from depthValue
    auto it_depth = depthValue.find(id);
    if (it_depth != depthValue.end())
        depthValue.erase(it_depth);
}

CSVLine& operator>>(CSVLine& line, VisionMeasurement& vision) {
    line >> vision.stamp;
    int numBearings;
    line >> numBearings;
    for (int i = 0; i < numBearings; ++i) {
        int id;
        Vector2d y;
        line >> id >> y;
        vision.camCoordinates[id] = y;
    }
    return line;
}

CSVLine& operator<<(CSVLine& line, const VisionMeasurement& vision) {
    line << vision.stamp;
    line << vision.camCoordinates.size();
    for (const pair<const int, Vector2d>& cc : vision.camCoordinates) {
        line << cc.first << cc.second;
    }
    return line;
}

VisionMeasurement operator-(const VisionMeasurement& y1, const VisionMeasurement& y2) {
    VisionMeasurement yDiff;
    

    for (const pair<const int, Vector2d>& cc1 : y1.camCoordinates) {
        const auto it2 = y2.camCoordinates.find(cc1.first);
        if (it2 != y2.camCoordinates.end()) {
            yDiff.camCoordinates[cc1.first] = cc1.second - it2->second;
        }
    }
    assert(y1.cameraPtr == y2.cameraPtr);
    // assert(y1.depthValue.size() == y2.depthValue.size());
    yDiff.cameraPtr = y1.cameraPtr;

    //add depth 
    // Subtract depth values
    for (const auto& depth1 : y1.depthValue) {
        const auto it2 = y2.depthValue.find(depth1.first);
        if (it2 != y2.depthValue.end()) {
            yDiff.depthValue[depth1.first] = depth1.second - it2->second;
        }
    }
    return yDiff;
}

VisionMeasurement::operator Eigen::VectorXd() const {
    // vector<int> ids = getIds();
    // Eigen::VectorXd result = Eigen::VectorXd(2 * ids.size());
    // for (size_t i = 0; i < ids.size(); ++i) {
    //     result.segment<2>(2 * i) = camCoordinates.at(ids[i]);
    // }
    // return result;
    //changed to 3D
    vector<int> ids = getIds();

    Eigen::VectorXd result = Eigen::VectorXd(3 * ids.size()); // Adjusting the size for 3D coordinates

    for (size_t i = 0; i < ids.size(); ++i) {
        result.segment<2>(3 * i) = camCoordinates.at(ids[i]); //  2D coordinates
        result(3 * i + 2) = depthValue.at(ids[i]); //  corresponding depth value
    }
    return result;
}




// VisionMeasurement::operator Eigen::VectorXd() const {
//     vector<int> ids = getIds();
//     Eigen::VectorXd result = Eigen::VectorXd(2 * ids.size());
//     for (size_t i = 0; i < ids.size(); ++i) {
//         result.segment<2>(2 * i) = camCoordinates.at(ids[i]);
//     }
//     return result;

// }

// VisionMeasurement operator+(const VisionMeasurement& y, const Eigen::VectorXd& eta) {
//     assert(eta.rows() == 2 * y.camCoordinates.size());
//     VisionMeasurement result = y;
//     size_t i = 0;
//     for (auto& pixelCoords : result.camCoordinates) {
//         pixelCoords.second += eta.segment<2>(2 * i);
//         ++i;
//     }
//     return result;
// }

VisionMeasurement operator+(const VisionMeasurement& y, const Eigen::VectorXd& eta) {
    assert(eta.rows() == static_cast<Eigen::Index>(3 * y.camCoordinates.size())); // Cast to match signedness

    // assert(eta.rows() == 3 * y.camCoordinates.size()); // 3 instead of 2
    VisionMeasurement result = y;
    size_t i = 0;
    for (auto& pixelCoords : result.camCoordinates) {
        pixelCoords.second += eta.segment<2>(3 * i); // 3 * i instead of 2 * i
        result.depthValue[pixelCoords.first] += eta(3 * i + 2); // add corresponding depth value
        ++i;
    }
    return result;
}
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

#include "eqvio/mathematical/EqFMatrices.h"
#include <iostream>

using namespace Eigen;
using namespace std;
using namespace liepp;

Eigen::MatrixXd EqFCoordinateSuite::stateMatrixADiscrete(
    const VIOGroup& X, const VIOState& xi0, const IMUVelocity& imuVel, const double& dt) const {
    // Compute using numerical differentiation

    auto a0Discrete = [&](const VectorXd& epsilon) {
        const auto xi_e = stateChart.inv(epsilon, xi0);
        const auto xi_hat = stateGroupAction(X, xi0);
        const auto xi = stateGroupAction(X, xi_e);
        const auto LambdaTilde =
            liftVelocityDiscrete(xi, imuVel, dt) * liftVelocityDiscrete(xi_hat, imuVel, dt).inverse();
        const auto xi_e1 = stateGroupAction(X * LambdaTilde * X.inverse(), xi_e);
        const VectorXd epsilon1 = stateChart(xi_e1, xi0);
        return epsilon1;
    };

    const auto A0tD = numericalDifferential(a0Discrete, Eigen::VectorXd::Zero(xi0.Dim()));
    return A0tD;
}

const Eigen::MatrixXd EqFCoordinateSuite::outputMatrixC(
    const VIOState& xi0, const VIOGroup& X, const VisionMeasurement& y, const bool useEquivariance) const {
    // Rows and their corresponding output components
    // [2i, 2i+2): Landmark measurement i

    // Cols and their corresponding state components
    // I am using zero indexing and half open ranges
    // [0,2): Gravity vector (deviation from e3)
    // [2,5) Body-fixed velocity
    // [5+3i,5+3(i+1)): Body-fixed landmark i

    const int M = xi0.cameraLandmarks.size();
    const vector<int> ids = y.getIds();
    const int N = ids.size();
    
    MatrixXd Cstar_rgb = MatrixXd::Zero(2 * N, VIOSensorState::CompDim + Landmark::CompDim * M);
  
    const VisionMeasurement yHat = measureSystemState(stateGroupAction(X, xi0), y.cameraPtr);

    for (int i = 0; i < M; ++i) {
        const int& idNum = xi0.cameraLandmarks[i].id;
        const Vector3d& qi0 = xi0.cameraLandmarks[i].p;
        const auto it_y = find(ids.begin(), ids.end(), idNum);
        const auto it_Q = find(X.id.begin(), X.id.end(), idNum);
        assert(it_Q != X.id.end());
        const int k = distance(X.id.begin(), it_Q);
        if (it_y != ids.end()) {
            assert(*it_y == *it_Q);
            assert(X.id[k] == idNum);

            const int j = distance(ids.begin(), it_y);

            // CStar.block<2, 3>(2 * j, VIOSensorState::CompDim + 3 * i) =
            //     useEquivariance ? outputMatrixCiStar(qi0, X.Q[k], y.cameraPtr, y.camCoordinates.at(idNum))
            //                     : outputMatrixCi(qi0, X.Q[k], y.cameraPtr);

            Eigen::Matrix<double, 2, 3> matrixRGB= useEquivariance ? outputMatrixCiStar(qi0, X.Q[k], y.cameraPtr, y.camCoordinates.at(idNum))
                                                     : outputMatrixCi(qi0, X.Q[k], y.cameraPtr);

            // Store RGB values
            Cstar_rgb.block<2, 3>(2 * j, VIOSensorState::CompDim + 3 * i) = matrixRGB;
        }
    }
    
    vector<int> validDepthIds;
    int validDepthCount = 0;
    // for (int i = 0; i < M; ++i) {
    //     const int& idNum = xi0.cameraLandmarks[i].id;
    //     if (y.depthValue.find(idNum) != y.depthValue.end() && !std::isnan(y.depthValue.at(idNum)) && y.depthValue.at(idNum) != 0) {
    //         validDepthCount++;
    //         validDepthIds.push_back(idNum);
    //     }
    // }

    std::unordered_map<int, int> idToIndexMap;
    for (int i = 0; i < M; ++i) {
        const int& idNum = xi0.cameraLandmarks[i].id;
        if (y.depthValue.find(idNum) != y.depthValue.end() && 
            !std::isnan(y.depthValue.at(idNum)) && 
            y.depthValue.at(idNum) != 0) {

            validDepthCount++;
            validDepthIds.push_back(idNum);
            idToIndexMap[idNum] = i; // Only insert id into the map if it's valid in the depth measurements
        }
    }
    std::sort(validDepthIds.begin(), validDepthIds.end());
    //Create the matrix with the determined size
    MatrixXd Cstar_depth = MatrixXd::Zero(validDepthCount, VIOSensorState::CompDim + Landmark::CompDim * M);

    

    for (size_t depthIdx = 0; depthIdx < validDepthIds.size(); ++depthIdx) {
        const int& idNum = validDepthIds[depthIdx];
        
        // Check if idNum exists in X.id and y's depth values
        const auto it_y = find(ids.begin(), ids.end(), idNum); 
        const auto it_Q = find(X.id.begin(), X.id.end(), idNum);

        assert(it_Q != X.id.end());
        const int k = distance(X.id.begin(), it_Q);
        
        if (it_y != ids.end()) {
            assert(*it_y == *it_Q);
            assert(X.id[k] == idNum);

            const liepp::SOT3d& QHat = X.Q[k];
            const Vector3d& qi0 = xi0.cameraLandmarks[k].p; 

            // const Vector3d qHat = QHat.inverse() * qi0;
            
            const double depthValue = y.depthValue.at(idNum);
            // std::cout <<"depth \n" << depthValue <<std::endl;
            // // std::cout << "Current idNum: \n " << idNum << std::endl;
            // std::cout << "use qvalue to calculate depth: \n" << qHat.norm() << std::endl;

            Eigen::Matrix<double, 1, 3> matrixDepth = outputMatrixci_depth(qi0, QHat, depthValue);
            // std::cout<<"depth matrix" <<matrixDepth<<std::endl;

            //k = idNum -1
            Cstar_depth.block<1, 3>(depthIdx, VIOSensorState::CompDim + 3 * k) = matrixDepth;
            // std::cout<<Cstar_depth<<std::endl;
        }
    } 

    MatrixXd CStar(Cstar_rgb.rows() + Cstar_depth.rows(), Cstar_rgb.cols());

   
    CStar << Cstar_rgb, Cstar_depth;

    assert(!CStar.hasNaN());
    return CStar;
}

const Eigen::Matrix<double, 2, 3> EqFCoordinateSuite::outputMatrixCi(
    const Eigen::Vector3d& q0, const liepp::SOT3d& QHat, const GIFT::GICameraPtr& camPtr) const {
    const Vector3d qHat = QHat.inverse() * q0;
    const Vector2d yHat = camPtr->projectPoint(qHat);
    return outputMatrixCiStar(q0, QHat, camPtr, yHat);
}

const Eigen::Matrix<double, 1, 3> EqFCoordinateSuite::outputMatrixci_depth(
    const Eigen::Vector3d& q0, const liepp::SOT3d& QHat,  const double measurement_depth) const {

    return outputMatrixCiStarDepth(q0,QHat, measurement_depth);
}


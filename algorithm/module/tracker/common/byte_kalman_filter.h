#pragma once

#include "byte_data_type.h"

namespace byte_kalman {

class KalmanFilter
{
public:
	static const double chi2inv95[10];
	KalmanFilter();

	/*******************************************************
	 * 
	 * @brief initialize a kalman filter by measurement
	 * 
	 * @param [in] measurement measurement states
	 * 
	 * @return pair of initial state and covariance
	 * 
	*******************************************************/
	KAL_DATA initiate(const DETECTBOX& measurement);

	/*******************************************************
	 * 
	 * @brief pridict current states by pervious states
	 * 
	 * @param [out] mean predicted states value
	 * @param [out] covariance covariance matrix
	 * 
	 * @return null
	 * 
	*******************************************************/
	void predict(KAL_MEAN& mean, KAL_COVA& covariance);

	/*******************************************************
	 * 
	 * @brief pridict current states by pervious states
	 * 
	 * @param [out] mean predicted states value
	 * @param [out] covariance covariance matrix
	 * 
	 * @return null
	 * 
	*******************************************************/
	KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);

	/*******************************************************
	 * 
	 * @brief states update
	 * 
	 * @param [in] mean predicted states value
	 * @param [in] covariance covariance matrix
	 * @param [in] measurement measurement states
	 * 
	 * @return pair of updated state and covariance
	 * 
	*******************************************************/
	KAL_DATA update(const KAL_MEAN& mean,
		const KAL_COVA& covariance,
		const DETECTBOX& measurement);

	/*******************************************************
	 * 
	 * @brief pridict current states by pervious states
	 * 
	 * @param [out] mean predicted states value
	 * @param [out] covariance covariance matrix
	 * 
	 * @return null
	 * 
	*******************************************************/
	Eigen::Matrix<float, 1, -1> gating_distance(
		const KAL_MEAN& mean,
		const KAL_COVA& covariance,
		const std::vector<DETECTBOX>& measurements,
		bool only_position = false);

private:
	Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
	Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
	float _std_weight_position;
	float _std_weight_velocity;
};

}	// namespace KalmanFilter
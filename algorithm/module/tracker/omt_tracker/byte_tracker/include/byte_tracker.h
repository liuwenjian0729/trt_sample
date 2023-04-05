#pragma once
#include"track.h"
#include"data_type.h"

namespace byte_tracker {

class ByteTracker
{
public:
    ByteTracker(int frame_rate=30, int track_buffer=30);
    ~ByteTracker(){};

	std::vector<ByteTarget> update(const std::vector<base::DetectBox>& objects);
	cv::Scalar get_color(int idx);

private:
	std::vector<ByteTarget*> joint_targets(std::vector<ByteTarget*> &tlista, std::vector<ByteTarget> &tlistb);
	std::vector<ByteTarget> joint_targets(std::vector<ByteTarget> &tlista, std::vector<ByteTarget> &tlistb);

	std::vector<ByteTarget> sub_targets(std::vector<ByteTarget> &tlista, std::vector<ByteTarget> &tlistb);
	void remove_duplicate_targets(std::vector<ByteTarget> &resa, std::vector<ByteTarget> &resb, std::vector<ByteTarget> &targetsa, std::vector<ByteTarget> &targetsb);
	void linear_assignment(std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);
	std::vector<std::vector<float> > iou_distance(std::vector<ByteTarget*> &atracks, std::vector<ByteTarget> &btracks, int &dist_size, int &dist_size_size);
	std::vector<std::vector<float> > iou_distance(std::vector<ByteTarget> &atracks, std::vector<ByteTarget> &btracks);
	std::vector<std::vector<float> > ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs);

	double lapjv(const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);	

private:
	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;

	std::vector<ByteTarget> tracked_targets;
	std::vector<ByteTarget> lost_targets;
	std::vector<ByteTarget> removed_targets;
	byte_kalman::KalmanFilter kalman_filter;
	std::vector<std::vector<double>> init_theta_sum;
    std::vector<std::vector<int>> init_theta_num;

	int init_theta_grid_h = 4; 
	int init_theta_grid_w = 6; 
};

}   // namespace    byte_tracker

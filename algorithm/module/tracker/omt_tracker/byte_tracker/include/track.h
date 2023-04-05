#pragma once
#include<vector>
#include<opencv2/opencv.hpp>
#include"byte_kalman_filter.h"
#include"track.h"

namespace byte_tracker {

enum TrackState { New = 0, Tracked, Lost, Removed };

class ByteTarget
{
public:
	ByteTarget(std::vector<float> tlwh_, float score);
	~ByteTarget();

	std::vector<float> static tlbr_to_tlwh(std::vector<float> &tlbr);
	void static multi_predict(std::vector<ByteTarget*> &targets, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	std::vector<float> tlwh_to_xyah(std::vector<float> tlwh_tmp);
	std::vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(ByteTarget &new_track, int frame_id, bool new_id = false);
	void update(ByteTarget &new_track, int frame_id);

public:
    TrackState state;      // < state of object
	bool is_activated;     // < is activated state or not
	int track_id;
	char uuid_[37];         // < unique identification of object

	std::vector<float> _tlwh;
	std::vector<float> tlwh;    // < position of object represented by [top, left, width, height]
	std::vector<float> tlbr;    // < position of object represented by [top, left, bottem, right]
	int frame_id;          // < current frame id
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;      // < kalman best estimate states
	KAL_COVA covariance;    // < kalman covariance matrix
	float score;    // < confidence of object

	byte_kalman::KalmanFilter kalman_filter;	// < kalman filter
};

}   //namespace byte_tracker
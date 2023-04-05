#include<uuid/uuid.h>
#include "track.h"

namespace byte_tracker {

//////////////////////////////////////////////////////////////////
// ByteTarget::ByteTarget()
//////////////////////////////////////////////////////////////////
ByteTarget::ByteTarget(std::vector<float> tlwh_, float score):
	frame_id(0),
	start_frame(0),
	tracklet_len(0),
	is_activated(false),
	track_id(0),
	state(TrackState::New),
	score(score)
{
	_tlwh.resize(4);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	tlwh.resize(4);
	tlbr.resize(4);

	static_tlwh();
	static_tlbr();

	uuid_t uuid_track;
    uuid_generate(uuid_track); 
    uuid_unparse(uuid_track, uuid_);
}

//////////////////////////////////////////////////////////////////
// ByteTarget::~ByteTarget()
//////////////////////////////////////////////////////////////////
ByteTarget::~ByteTarget()
{
}

//////////////////////////////////////////////////////////////////
// ByteTarget::activate()
//////////////////////////////////////////////////////////////////
void ByteTarget::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id)
{
	this->kalman_filter = kalman_filter;
	this->track_id = this->next_id();

	std::vector<float> _tlwhtmp(4);
	_tlwhtmp[0] = this->_tlwh[0];
	_tlwhtmp[1] = this->_tlwh[1];
	_tlwhtmp[2] = this->_tlwh[2];
	_tlwhtmp[3] = this->_tlwh[3];
	std::vector<float> xyah = tlwh_to_xyah(_tlwhtmp);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.initiate(xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	if (frame_id == 1)
	{
		this->is_activated = true;
	}
	//this->is_activated = true;
	this->frame_id = frame_id;
	this->start_frame = frame_id;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::re_activate()
//////////////////////////////////////////////////////////////////
void ByteTarget::re_activate(ByteTarget &new_track, int frame_id, bool new_id)
{
	std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	this->is_activated = true;
	this->frame_id = frame_id;
	this->score = new_track.score;
	if (new_id)
		this->track_id = next_id();
}

//////////////////////////////////////////////////////////////////
// ByteTarget::update()
//////////////////////////////////////////////////////////////////
void ByteTarget::update(ByteTarget &new_track, int frame_id)
{
	this->frame_id = frame_id;
	this->tracklet_len++;

	std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->state = TrackState::Tracked;
	this->is_activated = true;

	this->score = new_track.score;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::static_tlwh()
//////////////////////////////////////////////////////////////////
void ByteTarget::static_tlwh()
{
	if (this->state == TrackState::New)
	{
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		return;
	}

	tlwh[0] = mean[0];
	tlwh[1] = mean[1];
	tlwh[2] = mean[2];
	tlwh[3] = mean[3];

	tlwh[2] *= tlwh[3];
	tlwh[0] -= tlwh[2] / 2;
	tlwh[1] -= tlwh[3] / 2;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::static_tlbr()
//////////////////////////////////////////////////////////////////
void ByteTarget::static_tlbr()
{
	tlbr.clear();
	tlbr.assign(tlwh.begin(), tlwh.end());
	tlbr[2] += tlbr[0];
	tlbr[3] += tlbr[1];
}

//////////////////////////////////////////////////////////////////
// ByteTarget::tlwh_to_xyah()
//////////////////////////////////////////////////////////////////
std::vector<float> ByteTarget::tlwh_to_xyah(std::vector<float> tlwh_tmp)
{
	std::vector<float> tlwh_output = tlwh_tmp;
	tlwh_output[0] += tlwh_output[2] / 2;
	tlwh_output[1] += tlwh_output[3] / 2;
	tlwh_output[2] /= tlwh_output[3];
	return tlwh_output;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::to_xyah()
//////////////////////////////////////////////////////////////////
std::vector<float> ByteTarget::to_xyah()
{
	return tlwh_to_xyah(tlwh);
}

//////////////////////////////////////////////////////////////////
// ByteTarget::tlbr_to_tlwh()
//////////////////////////////////////////////////////////////////
std::vector<float> ByteTarget::tlbr_to_tlwh(std::vector<float> &tlbr)
{
	tlbr[2] -= tlbr[0];
	tlbr[3] -= tlbr[1];
	return tlbr;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::mark_lost()
//////////////////////////////////////////////////////////////////
void ByteTarget::mark_lost()
{
	state = TrackState::Lost;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::mark_removed()
//////////////////////////////////////////////////////////////////
void ByteTarget::mark_removed()
{
	state = TrackState::Removed;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::next_id()
//////////////////////////////////////////////////////////////////
int ByteTarget::next_id()
{
	static int _count = 0;
	_count++;
	return _count;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::end_frame()
//////////////////////////////////////////////////////////////////
int ByteTarget::end_frame()
{
	return this->frame_id;
}

//////////////////////////////////////////////////////////////////
// ByteTarget::multi_predict()
//////////////////////////////////////////////////////////////////
void ByteTarget::multi_predict(std::vector<ByteTarget*> &targets, byte_kalman::KalmanFilter &kalman_filter)
{
	for (int i = 0; i < targets.size(); i++)
	{
		if (targets[i]->state != TrackState::Tracked)
		{
			targets[i]->mean[7] = 0;
		}

		kalman_filter.predict(targets[i]->mean, targets[i]->covariance);
		// if(targets[i]->is_movement)
		// {
		// 	kalman_filter.predict(targets[i]->mean, targets[i]->covariance);
		// 	targets[i]->static_tlwh();
		// 	targets[i]->static_tlbr();
		// }
	}
}

}   // namespace byte_tracker

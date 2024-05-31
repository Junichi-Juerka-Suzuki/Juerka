/*
 * common.h
 *
 *  Created on: Nov 28, 2020
 *      Author: jun
 */

#pragma once
#include <cstddef>
#include <cstdint>

namespace Juerka::CommonNet
{
	using std::size_t;
	using std::uint64_t;

	using neuron_t = std::uint_fast16_t;
	using synapse_t = std::uint_fast32_t;
	using synapse_count_t = std::uint_fast16_t;
	using step_time_t = std::int_fast64_t;
	using elec_t = double;

	inline constexpr static step_time_t TIME_END = 1000; //msec.
	inline constexpr static size_t ALIGNMENT = 64;

	inline constexpr static size_t INPUT_SIDE = 0;
	inline constexpr static size_t OUTPUT_SIDE = 1;

	inline constexpr static size_t ADDITION_SIDE = 0;
	inline constexpr static size_t SUBTRACTION_SIDE = 1;

	struct SerialParam
	{
		bool is_need_apply_tonic_inputs;
		uint64_t rand_seed;
		step_time_t time_keep;
		elec_t* v;
		elec_t* u;
		elec_t* current_I;
		synapse_t* post_conn_next_neuron;
		elec_t* conn_weight;
		synapse_t* prev_conn_serial_index;
		synapse_t* pre_conn_start_index;
	};

	struct LogParam
	{
		bool is_record_weights;
	};
}

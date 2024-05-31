#pragma once

#include <array>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Common.h"
#include "TimeRecorder.hpp"

namespace Juerka::CommonNet
{
	using std::array;
	using std::assume_aligned;
	using std::cout;
	using std::deque;
	using std::move;
	using std::multimap;
	using std::mt19937_64;
	using std::ofstream;
	using std::ostream;
	using std::ref;
	using std::set;
	using std::to_string;
	using std::uniform_int_distribution;
	using std::unique_ptr;
	using std::vector;

	using Juerka::Utility::AbstractTimeRecorder;

	//Spike neural network constants.
	enum snn_t
	{
		CONST_SUB_BLOCK_Ne = 800,
		CONST_SUB_BLOCK_Ni = 200,
		CONST_SUB_BLOCK_N = 1000,
		SNN_T_NUM
	};

	class SerialNet
	{

	//Constructor/Destructor.
	public:
		SerialNet
		(
			const uint_fast32_t arg_id,
			const SerialParam serial_param,
			unique_ptr<AbstractTimeRecorder>&& arg_time_recorder,
			const LogParam log_param
		) noexcept :
		  id(arg_id),
		  time_keep(serial_param.time_keep),
		  v(assume_aligned<ALIGNMENT>(serial_param.v)),
		  u(assume_aligned<ALIGNMENT>(serial_param.u)),
		  current_I(assume_aligned<ALIGNMENT>(serial_param.current_I)),
		  post_conn_next_neuron(assume_aligned<ALIGNMENT>(serial_param.post_conn_next_neuron)),
		  conn_weight(assume_aligned<ALIGNMENT>(serial_param.conn_weight)),
		  prev_conn_serial_index(assume_aligned<ALIGNMENT>(serial_param.prev_conn_serial_index)),
		  pre_conn_start_index(serial_param.pre_conn_start_index),
		  last_fire_time_list(N, deque<step_time_t>()),
		  exc_fire_reservation(vector(D, vector<synapse_t>())),
		  pre_connection(N, vector<synapse_t>()),
		  pre_connection_list(vector(N, vector<synapse_t>())),
		  is_need_apply_tonic_inputs(serial_param.is_need_apply_tonic_inputs),
		  rand_seed(serial_param.rand_seed),
		  engine_drive(serial_param.rand_seed),
		  dist_drive(EXTERNAL_DRIVE_INPUT_START_INDEX, EXTERNAL_DRIVE_INPUT_END_CENTINEL-1),
		  time_recorder(move(arg_time_recorder)),
		  is_record_weights(log_param.is_record_weights)
		{
			for(neuron_t i=0; i<N; i+=1)
			{
				pre_connection[i].reserve(2*C);
				pre_connection_list[i].reserve(2*C);
			}
			for(step_time_t i=0; i<D; i+=1)
			{
				exc_fire_reservation[i].reserve(100);
			}
			inh_fire_reservation.reserve(100);
		  	weight_list.reserve(30000);
			post_neuron_list.reserve(30000);
			j_update_list.reserve(100);
			j_diff_time_list.reserve(100);
			j_weight_list.reserve(100);
		}

		SerialNet(SerialNet&&) noexcept = default;

		virtual ~SerialNet(void) noexcept
		{ }

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Function(s).

		//public interface.
	public:
		void serial_run
		(
			step_time_t arg_time_keep,
			array<vector<neuron_t>, 2>& target_neuron_list,
			array<vector<elec_t>, 2>& synaptic_current_list,
			array<multimap<neuron_t, neuron_t>, 2>& strong_edge_list
		) noexcept;

		//routines.
	private:
		void setup_connections(void) noexcept;

		void apply_tonic_inputs(void) noexcept;

		void apply_synaptic_currents(void) noexcept;

		void apply_external_inputs
		(
			vector<neuron_t>& target_neuron_list,
			vector<elec_t>& synaptic_current_list
		) noexcept;

		void update_neurons(void) noexcept;

		void reserve_fires(void) noexcept;

		void update_pre_neuron_weights(void) noexcept;

		void update_post_neuron_weights(void) noexcept;

		//sub function(s).
	private:
		void init_neurons(void) noexcept;

		void do_update_neurons(neuron_t i) noexcept;

		void do_update_weight
		(
			const synapse_t serial_index,
			const elec_t diff_time,
			const elec_t weight
		) noexcept;

		//for network analysis.
	private:
		void extract_network_graph_edges(array<multimap<neuron_t, neuron_t>, 2>& strong_edge_list);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Constant(s).

		//array size.
	public:
		//number of sub-blocked excitatory neurons.
		inline constexpr static neuron_t Ne = CONST_SUB_BLOCK_Ne;

		//number of sub-blocked inhibitory neurons.
		inline constexpr static neuron_t Ni = CONST_SUB_BLOCK_Ni;

		//number of sub-blocked neurons.
		inline constexpr static neuron_t N = CONST_SUB_BLOCK_N;

		//number of synaptic connections.
		inline constexpr static synapse_count_t C = 100;

		//max conductional delay.
		inline constexpr static step_time_t D = 20 + 1;

		//neuron parameter(s).
	private:
		//excitatory neuron parameter.
		inline constexpr static elec_t A_e = 0.02;
		inline constexpr static elec_t B_e = 0.2;
		inline constexpr static elec_t C_e = -65.0;
		inline constexpr static elec_t D_e = 8.0;

		//inhibitory neuron parameter.
		inline constexpr static elec_t A_i = 0.1;
		inline constexpr static elec_t B_i = 0.2;
		inline constexpr static elec_t C_i = -65.0;
		inline constexpr static elec_t D_i = 2.0;

		//initial values for neurons.
		inline constexpr static elec_t v_init = -65.0;
		inline constexpr static elec_t u_init = -13.0;
		inline constexpr static elec_t i_init = 0.0;

		//synaptic parameter(s).
	private:
		inline constexpr static elec_t conn_weight_init_e = 6.0;
		inline constexpr static elec_t conn_weight_init_i = -5.0;
		inline constexpr static elec_t conn_weight_to_external_e = 6.0;
		inline constexpr static elec_t conn_weight_to_external_i = -5.0;
		inline constexpr static elec_t conn_weight_network_generate_threshold_e = 50.0;

		//STDP parameter(s).
	private:
		inline constexpr static elec_t ETA = 1.1;
		//inline constexpr static elec_t ETA = 1.05;
		//inline constexpr static elec_t ETA = 0.75;
		//	inline constexpr static elec_t ETA = 0.01;
	//	inline constexpr static elec_t J0 = 0.025;
	//	inline constexpr static elec_t J0 = 0.5;
		inline constexpr static elec_t J0 = 1.5;
		//	inline constexpr static elec_t J0 = 0.05;
	//	inline constexpr static elec_t ALPHA = 15;
//		inline constexpr static elec_t ALPHA = 50.0;
//		inline constexpr static elec_t BETA = 100.0;
		inline constexpr static elec_t ALPHA = 10.0;
		inline constexpr static elec_t BETA = 20.0;
//		inline constexpr static elec_t C_PLUS = 20.0;
		inline constexpr static elec_t C_PLUS = 28.0;
//		inline constexpr static elec_t C_MINUS = 80.0;
		inline constexpr static elec_t C_MINUS = 50.0;
//		inline constexpr static elec_t TAU_PLUS = 10.0;
		inline constexpr static elec_t TAU_PLUS = 30.0;
		inline constexpr static elec_t TAU_MINUS = 80.0;

		inline constexpr static elec_t INV_J0 = 1.0 / J0;
		inline constexpr static elec_t INV_J0_BETA = 1.0 / (J0 * BETA);
		inline constexpr static elec_t INV_TAU_PLUS = 1.0 / TAU_PLUS;
		inline constexpr static elec_t INV_TAU_MINUS = 1.0 / TAU_MINUS;
		inline constexpr static elec_t INV_ALPHA = 1.0 / ALPHA;

		//other parameter(s).
	private:
		//range of drive input.
		inline constexpr static neuron_t EXTERNAL_DRIVE_INPUT_START_INDEX = 0;
		inline constexpr static neuron_t EXTERNAL_DRIVE_INPUT_END_CENTINEL = CONST_SUB_BLOCK_N;

		//neuron number to apply one drive input.
		inline constexpr static neuron_t DRIVE_UNIT = 1000;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Variable(s).

		//for network management.
	private:
		uint_fast32_t id;

		//time management.
	private:
		step_time_t time_keep;

		//for spike neural network simulation(neuron(s) & synapse(s) related members).
	private:
		elec_t* v;
		elec_t* u;
		elec_t* current_I;
		synapse_t* post_conn_next_neuron;
		elec_t* conn_weight;
		synapse_t* prev_conn_serial_index;
		synapse_t* pre_conn_start_index;

		deque<neuron_t> is_fired_list;

		vector< deque<step_time_t> > last_fire_time_list;
		vector< vector<synapse_t> > exc_fire_reservation;
		vector<synapse_t> inh_fire_reservation;
		vector< vector<synapse_t> > pre_connection;
		vector< vector<synapse_t> > pre_connection_list;

		vector<elec_t> weight_list;
		vector<neuron_t> post_neuron_list;

		vector<synapse_t> j_update_list;
		vector<elec_t> j_diff_time_list;
		vector<elec_t> j_weight_list;

		//for generating random tonic inputs.
	private:
		bool is_need_apply_tonic_inputs;
		uint64_t rand_seed;
		mt19937_64 engine_drive;
		uniform_int_distribution<> dist_drive;

		vector<neuron_t> neuron_indices;

		//utilities for monitoring purpose.
	private:
		bool is_record_weights;
		set<synapse_t> extracted;

		unique_ptr<AbstractTimeRecorder> time_recorder;
	};
}

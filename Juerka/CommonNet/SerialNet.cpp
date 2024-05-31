/*
 * SerialNet.cpp
 *
 *  Created on: Dec 10, 2020
 *      Author: jun
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>

#include "Common.h"
#include "SerialNet.h"

namespace Juerka::CommonNet
{
	using std::array;
	using std::cerr;
	using std::copy_if;
	using std::cout;
	using std::endl;
	using std::execution::unseq;
	using std::inserter;
	using std::int_fast64_t;
	using std::mt19937_64;
	using std::multimap;
	using std::set;
	using std::set_difference;
	using std::size_t;
	using std::sort;
	using std::uniform_int_distribution;
	using std::vector;

	void SerialNet::serial_run
	(
		step_time_t arg_time_keep,
		array<vector<neuron_t>, 2>& target_neuron_list,
		array<vector<elec_t>, 2>& synaptic_current_list,
		array<multimap<neuron_t, neuron_t>, 2>& strong_edge_list
	) noexcept
	{
		time_keep = arg_time_keep;

		time_recorder->refresh_records();

		time_recorder->start_record();

		setup_connections();

		time_recorder->end_record();

		time_recorder->start_record();

		{
			is_fired_list.clear();
		}

		time_recorder->end_record();

		time_recorder->start_record();

		if (is_need_apply_tonic_inputs)
		{
			apply_tonic_inputs();
		}

		time_recorder->end_record();

		time_recorder->start_record();

		apply_synaptic_currents();

		time_recorder->end_record();

		time_recorder->start_record();

		apply_external_inputs(target_neuron_list[INPUT_SIDE], synaptic_current_list[INPUT_SIDE]);

		time_recorder->end_record();

		time_recorder->start_record();

		update_neurons();

		time_recorder->end_record();

		time_recorder->start_record();

		{
			target_neuron_list[OUTPUT_SIDE].clear();
			synaptic_current_list[OUTPUT_SIDE].clear();

			const auto& target_vector(is_fired_list);

			for(auto it=target_vector.begin(); it!=target_vector.end(); it++)
			{
				neuron_t neuron_index(*it);

				target_neuron_list[OUTPUT_SIDE].emplace_back(neuron_index);
				synaptic_current_list[OUTPUT_SIDE].emplace_back
				(
					(neuron_index<Ne)
					?
					(conn_weight_to_external_e)
					:
					(conn_weight_to_external_i)
				);
			}
		}

		time_recorder->end_record();

		time_recorder->start_record();

		reserve_fires();

		time_recorder->end_record();

		time_recorder->start_record();

		j_update_list.clear();
		j_diff_time_list.clear();
		j_weight_list.clear();

		time_recorder->end_record();

		time_recorder->start_record();

		update_pre_neuron_weights();

		time_recorder->end_record();

		time_recorder->start_record();

		update_post_neuron_weights();

		time_recorder->end_record();

		time_recorder->start_record();

		{
			size_t size(j_update_list.size());

			for(size_t i=0; i<size; i+=1)
			{
				const synapse_t serial_index(j_update_list[i]);
				const elec_t diff_time(j_diff_time_list[i]);
				const elec_t weight(j_weight_list[i]);

				do_update_weight(serial_index, diff_time, weight);
			}
		}

		time_recorder->end_record();

		time_recorder->start_record();

		//serial
		{
			const step_time_t index_to_clear(time_keep % D);

			exc_fire_reservation[index_to_clear].clear();
		}

		time_recorder->end_record();

		time_recorder->start_record();

		if (is_record_weights)
		{
			strong_edge_list[ADDITION_SIDE].clear();
			strong_edge_list[SUBTRACTION_SIDE].clear();

			extract_network_graph_edges(strong_edge_list);
		}

		time_recorder->end_record();

		time_recorder->dump_records();
	}

	void SerialNet::setup_connections(void) noexcept
	{
		//anonymous block Lv.1
		if(0 == time_keep)
		{
			init_neurons();

			//serial
			{
				for(neuron_t i=0; i<N; i+=1)
				{
					auto& target_vector(pre_connection_list[i]);
					std::sort(unseq, target_vector.begin(), target_vector.end());
				}
			}

			//serial
			//anonymous block Lv.2
			{
				neuron_t l(0);
				synapse_t accum_count(0);
				synapse_t serial_index(0);

				for(; l<N; l+=1)
				{
					const auto& target_vector(pre_connection_list[l]);

					const size_t count(target_vector.size());
					synapse_t block_count(0);
					for(size_t k=0; k<count; k+=1)
					{
						const synapse_t pre_conn(target_vector[k]);
						const neuron_t pre_neuron_index(pre_conn/C);
						if(Ne > pre_neuron_index)
						{
							prev_conn_serial_index[serial_index] = pre_conn;
							serial_index += 1;
							block_count += 1;
						}
					}
					pre_conn_start_index[l] = accum_count;

					accum_count += block_count;
				}
				pre_conn_start_index[l] = accum_count;

			}//end anonymous block Lv.2

		}//end anonymous block Lv.1
	}

	void SerialNet::init_neurons(void) noexcept
	{
		vector<synapse_count_t> post_connection;
		post_connection.reserve(C);

		for(neuron_t i=0; i<N; i+=1)
		{
			mt19937_64 engine(rand_seed+i);
			uniform_int_distribution<> dist(0, (i<Ne)?(N-1):(Ne-1));

			v[i] = v_init;
			u[i] = u_init;

			{
				post_connection.clear();

				for(synapse_count_t j=0; j<C; j+=1) {

					const synapse_t serial_index(i*C+j);

					conn_weight[serial_index] = (i<Ne)?(conn_weight_init_e):(conn_weight_init_i);

					const neuron_t pre(i);
					const neuron_t post(dist(engine));

					auto beg_it(post_connection.begin());
					auto end_it(post_connection.end());

					if
					(
						(pre==post)
						||
						(end_it != find(beg_it, end_it, post))
					)
					{j--; continue;}


					post_connection.emplace_back(post);

					post_conn_next_neuron[serial_index] = post;

					pre_connection_list[post].emplace_back(serial_index);
				}
			}
		}//end for
	}


	void SerialNet::apply_tonic_inputs(void) noexcept
	{
		neuron_indices.clear();

		for(uint_fast32_t i=0; i<N/DRIVE_UNIT; i+=1)
		{
			neuron_indices.emplace_back(dist_drive(engine_drive));
		}

		for(auto it=neuron_indices.begin(); it!=neuron_indices.end(); it++)
		{
			current_I[*it] += 20.0;
		}
	}

	void SerialNet::apply_synaptic_currents(void) noexcept
	{
		const step_time_t local_current_time(time_keep % D);

		auto& target_vector_exc(exc_fire_reservation[local_current_time]);
		const size_t size_exc(target_vector_exc.size());

		//sort(unseq, target_vector_exc.begin(), target_vector_exc.end());

		weight_list.clear();
		post_neuron_list.clear();

		for(size_t i=0; i<size_exc; i+=1)
		{
			const synapse_t serial_index(target_vector_exc[i]);

			const elec_t weight(conn_weight[serial_index]);

			if(weight == 0.0) continue;

			const neuron_t post_neuron_index(post_conn_next_neuron[serial_index]);

			weight_list.emplace_back(weight);

			post_neuron_list.emplace_back(post_neuron_index);
		}

		auto& target_vector_inh(inh_fire_reservation);
		const size_t size_inh(target_vector_inh.size());

		for(size_t k=0; k<size_inh; k+=1)
		{
			neuron_t i(target_vector_inh[k]);

			i*=C;

			for(synapse_count_t j=0; j<C; j+=1)
			{
				const synapse_t serial_index(i);

				const elec_t weight(conn_weight[serial_index]);

				const neuron_t post_neuron_index(post_conn_next_neuron[serial_index]);

				weight_list.emplace_back(weight);

				post_neuron_list.emplace_back(post_neuron_index);

				i+=1;
			}
		}

		const size_t size_extracted(weight_list.size());

		for(size_t i=0; i<size_extracted; i+=1)
		{
			neuron_t post_neuron_index(post_neuron_list[i]);

			elec_t weight(weight_list[i]);

			current_I[post_neuron_index] += weight;
		}
	}

	void SerialNet::apply_external_inputs
	(
		vector<neuron_t>& target_neuron_list,
		vector<elec_t>& synaptic_current_list
	) noexcept
	{
		assert(target_neuron_list.size() == synaptic_current_list.size());

		size_t size_0(target_neuron_list.size());
		size_t size_1(synaptic_current_list.size());
		size_t size((size_0 < size_1) ? size_0 : size_1);

		for(size_t i=0; i<size; i+=1)
		{
			current_I[target_neuron_list[i]] += synaptic_current_list[i];
		}
	}

	void SerialNet::update_neurons(void) noexcept
	{
		for(neuron_t i=0; i<N; i+=1)
		{
			do_update_neurons(i);
		}
	}

	inline void SerialNet::do_update_neurons(const neuron_t i) noexcept
	{
		if(v[i]>=30.0)
		{
			v[i] = (i<Ne)?(C_e):(C_i);
			u[i] += (i<Ne)?(D_e):(D_i);
			is_fired_list.emplace_back(i);
			last_fire_time_list[i].emplace_back(time_keep);
		}

		v[i]+=0.5*((0.04*v[i]+5.0)*v[i]+140.0-u[i]+current_I[i]);
		v[i]+=0.5*((0.04*v[i]+5.0)*v[i]+140.0-u[i]+current_I[i]);
		u[i]+=((i<Ne)?(A_e):(A_i))*(((i<Ne)?(B_e):(B_i))*v[i]-u[i]);

		//clear current_I
		current_I[i] = 0.0;
	}


	void SerialNet::reserve_fires(void) noexcept
	{
		//serial.
		{
			inh_fire_reservation.clear();

			const auto& target_vector_0(is_fired_list);
			const size_t M(target_vector_0.size());

			for(size_t k=0; k<M; k+=1)
			{
				const neuron_t i(target_vector_0[k]);

				step_time_t release_item_candidate(time_keep);

				auto& target_vector_1(last_fire_time_list[i]);

				if(target_vector_1.size() > 0)
				{
					release_item_candidate = target_vector_1.front();
				}

				const step_time_t d(static_cast<step_time_t>(D));

				if(time_keep > d)
				{
					const step_time_t release_threshold = time_keep - d;

					if(release_item_candidate < release_threshold)
					{
						target_vector_1.pop_front();
					}
				}

				if(Ne <= i)
				{
					//case inhibitory neuron fires.
					inh_fire_reservation.emplace_back(i);
				}
				else
				{
					//case excitatory neuron fires.
					for(synapse_count_t j=0; j<C; j+=1)
					{
						const synapse_t serial_index(i*C+j);

						const step_time_t current_delay((i<Ne)?((serial_index % D) + 1):(1));

						const step_time_t fire_time = (time_keep + current_delay) % D;

						exc_fire_reservation[fire_time].emplace_back(serial_index);
					}
				}
			}
		}
	}

	void SerialNet::update_pre_neuron_weights(void) noexcept
	{
		const auto& target_is_fired_list(is_fired_list);

		const size_t M(target_is_fired_list.size());

		for(size_t j=0; j<M; j+=1)
		{
			const neuron_t i(target_is_fired_list[j]);

			const synapse_t start_prev_conn_serial_index(pre_conn_start_index[i]);
			const synapse_t end_prev_conn_serial_index(pre_conn_start_index[i+1]);

			for
			(
				synapse_t ind = start_prev_conn_serial_index;
				ind < end_prev_conn_serial_index;
				ind+=1
			)
			{
				const synapse_t serial_index = prev_conn_serial_index[ind];

				const neuron_t pre_neuron_index(serial_index/C);

				const auto& target_last_fire_time_list(last_fire_time_list[pre_neuron_index]);

				if(0 == target_last_fire_time_list.size())
				{
					continue;
				}

				const step_time_t pre_conn_delay((pre_neuron_index<Ne)?((serial_index % D) + 1):(1));

				const step_time_t base_time(time_keep - pre_conn_delay);

				bool is_value_found(false);

				step_time_t pre_fire_time;

				for(int_fast64_t p=target_last_fire_time_list.size()-1; p>=0; p--)
				{
					pre_fire_time = target_last_fire_time_list[p];
					if(pre_fire_time < base_time)
					{
						is_value_found = true;
						break;
					}
				}

				if(is_value_found)
				{
					const step_time_t post_fire_time(time_keep);
					const step_time_t diff_time(pre_fire_time + pre_conn_delay - post_fire_time);

					if(0 != diff_time)
					{
						j_update_list.emplace_back(serial_index);
						j_diff_time_list.emplace_back(static_cast<elec_t>(diff_time));
						j_weight_list.emplace_back(conn_weight[serial_index]);
					}
				}//if
			}//for
		}//for
	}

	void SerialNet::update_post_neuron_weights(void) noexcept
	{
		const step_time_t local_current_time(time_keep % D);

		step_time_t post_fire_time;

		const auto& target_vector(exc_fire_reservation[local_current_time]);

		const size_t M(target_vector.size());

		for(size_t i=0; i<M; i+=1)
		{
			const synapse_t serial_index = target_vector[i];

			const neuron_t post_neuron_index(post_conn_next_neuron[serial_index]);

			const step_time_t pre_fire_time(time_keep);
			const step_time_t base_time(time_keep);

			const auto& target_deque(last_fire_time_list[post_neuron_index]);

			bool is_value_found(false);

			for(int_fast64_t p=target_deque.size()-1; p>=0; p--)
			{
				post_fire_time = target_deque[p];
				if(post_fire_time < base_time)
				{
					is_value_found = true;
					break;
				}
			}

			if(is_value_found)
			{
				const step_time_t diff_time(pre_fire_time - post_fire_time);

				if(0 != diff_time)
				{
					j_update_list.emplace_back(serial_index);
					j_diff_time_list.emplace_back(static_cast<elec_t>(diff_time));
					j_weight_list.emplace_back(conn_weight[serial_index]);
				}//if
			}//if
		}//for
	}

	inline void SerialNet::do_update_weight
	(
		const synapse_t serial_index,
		const elec_t diff_time,
		const elec_t weight
	) noexcept
	{
		elec_t w;

		const elec_t u(diff_time);
		const elec_t j(weight);

		if(u < 0.0)
		{
			const elec_t f_plus(C_PLUS * exp(-j * INV_J0_BETA));
			w = f_plus * exp(u * INV_TAU_PLUS);
		}
		else // u > 0
		{
			elec_t f_minus(-C_MINUS);

			if(j <= J0)
			{
				f_minus = f_minus * j * INV_J0;
			}
			else // j > J0
			{
				f_minus = f_minus * ( 1.0 + log1p( ALPHA * (j * INV_J0 - 1.0)) * INV_ALPHA );
			}
			w = f_minus * exp(-u * INV_TAU_MINUS);
		}

		const elec_t dj = ETA * w;

		elec_t new_j_candidate(j+dj);

		if(new_j_candidate <= 0.0)
		{
			new_j_candidate = 0.0; // clipping
		}

		conn_weight[serial_index] = new_j_candidate;
	}

	void SerialNet::extract_network_graph_edges(array<multimap<neuron_t,neuron_t>, 2>& strong_edge_list)
	{
		set<synapse_t> new_extracted;
		array<set<synapse_t>, 2> network_difference;

		for(synapse_t i=0; i<N*C; i++)
		{
			elec_t weight(conn_weight[i]);
			if (weight >= conn_weight_network_generate_threshold_e)
			{
				new_extracted.emplace(i);
			}
		}

		set_difference
		(
			new_extracted.begin(), new_extracted.end(),
			extracted.begin(), extracted.end(),
			inserter(network_difference[ADDITION_SIDE], network_difference[ADDITION_SIDE].end())
		);
	
		set_difference
		(
			extracted.begin(), extracted.end(),
			new_extracted.begin(), new_extracted.end(),
			inserter(network_difference[SUBTRACTION_SIDE], network_difference[SUBTRACTION_SIDE].end())
		);

		for
		(
			auto it = network_difference[ADDITION_SIDE].begin();
			it != network_difference[ADDITION_SIDE].end();
			it++
		)
		{
			synapse_t serial_index(*it);
			neuron_t pre((serial_index)/C);
			neuron_t post(post_conn_next_neuron[serial_index]);
			strong_edge_list[ADDITION_SIDE].emplace(pre, post);
		}

		for
		(
			auto it = network_difference[SUBTRACTION_SIDE].begin();
			it != network_difference[SUBTRACTION_SIDE].end();
			it++
		)
		{
			synapse_t serial_index(*it);
			neuron_t pre((serial_index) / C);
			neuron_t post(post_conn_next_neuron[serial_index]);
			strong_edge_list[SUBTRACTION_SIDE].emplace(pre, post);
		}

		extracted.swap(new_extracted);
	}
}

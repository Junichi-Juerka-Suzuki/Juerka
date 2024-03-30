#include <cassert>
#include <functional>

#include "NetworkGroup.h"

namespace Juerka::CommonNet
{
	using std::bind;

	void NetworkGroup::work
	(
		stop_token stoken,
		uint_fast32_t thread_serial_number
	) noexcept
	{
		while(not stoken.stop_requested())
		{
			task_signal_list[thread_serial_number].semaphore.acquire();

			if(not task_signal_list[thread_serial_number].tasks.empty())
			{
				auto task(task_signal_list[thread_serial_number].tasks.front());

				if(task)
				{
					task();
				}

				task_signal_list[thread_serial_number].tasks.pop();

			}

			task_signal_list[thread_serial_number].semaphore.release();
		}
	}

	void NetworkGroup::wait_all_asyncs(void) noexcept
	{
		for(uint_fast32_t i=0; i<n_asyncs; i+=1)
		{
			while(true)
			{
				bool end_flag = false;
				task_signal_list[i].semaphore.acquire();
				if(task_signal_list[i].tasks.empty())
				{
					end_flag = true;
				}
				task_signal_list[i].semaphore.release();

				if(end_flag)
				{
					break;
				}
			}
		}
	}

	void NetworkGroup::prepare_and_run_asyncs
	(
		const function<void(void)>&& func,
		const uint_fast32_t async_serial_number
	) noexcept
	{
		task_signal_list[async_serial_number].semaphore.acquire();
		task_signal_list[async_serial_number].tasks.push(move(func));
		task_signal_list[async_serial_number].semaphore.release();
	}

	void NetworkGroup::complete_stop_threads(void) noexcept
	{
		for(uint_fast32_t i=0; i<n_asyncs; i+=1)
		{
			thread_list[i].request_stop();
		}

		for(uint_fast32_t i=0; i<n_asyncs; i+=1)
		{
			task_signal_list[i].semaphore.release();
		}

		for(uint_fast32_t i=0; i<n_asyncs; i+=1)
		{
			thread_list[i].join();
		}
	}

	void NetworkGroup::run
	(
		step_time_t arg_time_keep,
		vector< array<vector<neuron_t>, 2> >& neuron_list,
		vector< array<vector<elec_t>, 2> >& synaptic_current_list,
		vector<	array<set<synapse_t>, 2> >& strong_edge_list
	) noexcept
	{
		if(is_run_parallel)
		{
			parallel_run(arg_time_keep, neuron_list, synaptic_current_list, strong_edge_list);
		}
		else
		{
			serial_run(arg_time_keep, neuron_list, synaptic_current_list, strong_edge_list);
		}
	}

	void NetworkGroup::serial_run
	(
		step_time_t arg_time_keep,
		vector< array<vector<neuron_t>, 2> >& neuron_list,
		vector< array<vector<elec_t>, 2> >& synaptic_current_list,
		vector<	array<set<synapse_t>, 2> >& strong_edge_list
	) noexcept
	{
		exchange_internetwork_signals(neuron_list, synaptic_current_list);

		for(uint_fast32_t i=0; i<Ng; i+=1)
		{
			network_list[i].serial_run(arg_time_keep, neuron_list[i], synaptic_current_list[i], strong_edge_list[i]);
		}
	}

	void NetworkGroup::parallel_run
	(
		step_time_t arg_time_keep,
		vector< array<vector<neuron_t>, 2> >& neuron_list,
		vector< array<vector<elec_t>, 2> >& synaptic_current_list,
		vector< array<set<synapse_t>, 2> >& strong_edge_list
	) noexcept
	{
		do_parallel_progress.a.store(0);

		exchange_internetwork_signals(neuron_list, synaptic_current_list);

		for(uint_fast32_t i=0; i<n_asyncs; i+=1)
		{
			prepare_and_run_asyncs(bind(&NetworkGroup::do_parallel_run, this, arg_time_keep, ref(neuron_list), ref(synaptic_current_list), ref(strong_edge_list), i), i);
		}

		wait_all_asyncs();
	}

	void NetworkGroup::do_parallel_run
	(
		step_time_t arg_time_keep,
		vector< array<vector<neuron_t>, 2> >& neuron_list,
		vector< array<vector<elec_t>, 2> >& synaptic_current_list,
		vector< array<set<synapse_t>, 2> >& strong_edge_list,
		uint_fast32_t thread_serial_number
	) noexcept
	{
		(void) thread_serial_number;

		auto& target_a(do_parallel_progress.a);

		size_t m;

		while((m=target_a.fetch_add(SUB_BLOCK_SIZE)) < Ng)
		{
			for(size_t l=0; l<SUB_BLOCK_SIZE; l+=1)
			{
				size_t i(m+l);
				if(i>=Ng)
				{
					break;
				}

				network_list[i].serial_run(arg_time_keep, neuron_list[i], synaptic_current_list[i], strong_edge_list[i]);
			}//for
		}//while
	}//void NetworkGroup::parallel_run

	void NetworkGroup::exchange_internetwork_signals
	(
		vector< array<vector<neuron_t>, 2> >& arg_neuron_list_list,
		vector< array<vector<elec_t>, 2> >& arg_synaptic_current_list_list
	) noexcept
	{
		assert(arg_neuron_list_list.size() == arg_synaptic_current_list_list.size());

		const size_t network_size(network_list.size());
		const size_t signal_unit(Juerka::CommonNet::SerialNet::N/network_size);

		const size_t size_i(arg_neuron_list_list.size());

		for(size_t i=0; i<size_i; i+=1)
		{
			arg_neuron_list_list[i][Juerka::CommonNet::INPUT_SIDE].clear();
			arg_synaptic_current_list_list[i][Juerka::CommonNet::INPUT_SIDE].clear();
		}

		for(size_t i=0; i<size_i; i+=1)
		{
			const auto& fired_neuron_list
			(
				arg_neuron_list_list[i][Juerka::CommonNet::OUTPUT_SIDE]
			);

			const auto& synaptic_current_list
			(
				arg_synaptic_current_list_list[i][Juerka::CommonNet::OUTPUT_SIDE]
			);

			assert
			(
				fired_neuron_list.size() == synaptic_current_list.size()
			);

			const size_t size_j(fired_neuron_list.size());

			for(size_t j=0; j<size_j; j+=1)
			{
				const neuron_t fired_neuron_index(fired_neuron_list[j]);
				const elec_t synaptic_current(synaptic_current_list[j]);

				if (fired_neuron_index >= Juerka::CommonNet::SerialNet::Ne)
				{
					continue;
				}

//				for (size_t k=0; k<signal_unit; k++)
//				{
					const size_t target_neuron_list_index(((fired_neuron_index / signal_unit)));

					// connection from/to the same network is inhibited here.
					// it is handled in the SerialNet.
					if (target_neuron_list_index != i)
					{
						const neuron_t N(Juerka::CommonNet::SerialNet::N);
						const neuron_t target_neuron_index(fired_neuron_index);
						
						arg_neuron_list_list
							[target_neuron_list_index]
							[Juerka::CommonNet::INPUT_SIDE]
							.emplace_back(target_neuron_index);

						arg_synaptic_current_list_list
							[target_neuron_list_index]
							[Juerka::CommonNet::INPUT_SIDE]
							.emplace_back(synaptic_current);
					}//if
//				}//for k
			}//for j
		}//for i
	}//void NetworkGroup::exchange_internetwork_signals
}//namespace Juerka::CommonNet

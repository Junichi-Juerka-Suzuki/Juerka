#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <semaphore>
#include <thread>
#include <utility>
#include <vector>

#include "Common.h"
#include "SerialNet.h"
#include "TimeRecorder.hpp"

namespace Juerka::CommonNet
{
	using std::align_val_t;
	using std::array;
	using std::atomic;
	using std::binary_semaphore;
	using std::cerr;
	using std::cout;
	using std::endl;
	using std::function;
	using std::jthread;
	using std::move;
	using std::multimap;
	using std::queue;
	using std::ref;
	using std::stop_token;
	using std::thread;
	using std::uint_fast32_t;
	using std::unique_ptr;
	using std::vector;

	using Juerka::Utility::AbstractTimeRecorder;
	using Juerka::Utility::TimeRecorder;

	class NetworkGroup
	{
		//Constants.
		inline constexpr static uint_fast32_t SUB_BLOCK_SIZE = 1;

		//Variables.
		const uint_fast32_t Ng;

		//Buffers.
		alignas(ALIGNMENT) elec_t* v;
		alignas(ALIGNMENT) elec_t* u;
		alignas(ALIGNMENT) elec_t* current_I;
		alignas(ALIGNMENT) synapse_t* post_conn_next_neuron;
		alignas(ALIGNMENT) elec_t* conn_weight;
		alignas(ALIGNMENT) synapse_t* prev_conn_serial_index;
		alignas(ALIGNMENT) synapse_t* pre_conn_start_index;

		//Network itself.
		vector<SerialNet> network_list;

		//for parallelism.
		bool is_run_parallel;
		uint_fast32_t n_asyncs;
		vector<jthread> thread_list;

		template <typename T>
		struct atom_wrapper
		{
			atomic<T> a;

			atom_wrapper(void) noexcept : a() {}

			atom_wrapper(const atomic<T> &arg_a) :a(arg_a.load()) {}

			atom_wrapper(const atom_wrapper& other) :a(other.a.load()) {}

			atom_wrapper& operator=(const atom_wrapper &other)
			{
				a.store(other.a.load());
			}
		};

		atom_wrapper<size_t> do_parallel_progress;

		using task_t = function<void(void)>;
		using semaphore_t = binary_semaphore;
		struct task_pair
		{
			queue<task_t> tasks;
			semaphore_t semaphore;

			task_pair(void) noexcept : tasks(), semaphore(1) {}
		};

		vector<task_pair> task_signal_list;

		//Constructor/Destructor.
	public:

		NetworkGroup
		(
			uint_fast32_t arg_Ng,
			bool arg_is_run_parallel=false,
			bool arg_is_monitor_performance = false,
			bool arg_is_record_weights = false
		) noexcept :
		  Ng(arg_Ng),
		  v(static_cast<elec_t*>(::operator new(sizeof(elec_t) * Ng * (SerialNet::N), align_val_t{ ALIGNMENT }))),
		  u(static_cast<elec_t*>(::operator new(sizeof(elec_t) * Ng * (SerialNet::N), align_val_t{ ALIGNMENT }))),
		  current_I(static_cast<elec_t*>(::operator new(sizeof(elec_t) * Ng * (SerialNet::N), align_val_t{ ALIGNMENT }))),
		  post_conn_next_neuron(static_cast<synapse_t*>(::operator new(sizeof(synapse_t) * Ng * (SerialNet::N * SerialNet::C), align_val_t{ ALIGNMENT }))),
		  conn_weight(static_cast<elec_t*>(::operator new(sizeof(elec_t) * Ng * (SerialNet::N * SerialNet::C), align_val_t{ ALIGNMENT }))),
		  prev_conn_serial_index(static_cast<synapse_t*>(::operator new(sizeof(synapse_t) * Ng * (SerialNet::N * SerialNet::C), align_val_t{ ALIGNMENT }))),
		  pre_conn_start_index(static_cast<synapse_t*>(::operator new(sizeof(synapse_t) * Ng * (SerialNet::N+1), align_val_t{ ALIGNMENT }))),
		  is_run_parallel(arg_is_run_parallel),
		  n_asyncs((thread::hardware_concurrency()==1)?(1):(thread::hardware_concurrency()-1)),
		  do_parallel_progress(0),
	  	  task_signal_list(n_asyncs)
		{
			const LogParam log_param
			{
				.is_record_weights = arg_is_record_weights,
			};

			const size_t N(SerialNet::N);
			const size_t C(SerialNet::C);

			for(uint_fast32_t i=0; i<Ng; i+=1)
			{
				//TODO: care alignment. static_assert?
				SerialParam serial_param
				{
					.rand_seed = i, //TODO: consider.
					.time_keep = 0,
					.v = &(v[N*i]),
					.u = &(u[N*i]),
					.current_I = &(current_I[N*i]),
					.post_conn_next_neuron = &(post_conn_next_neuron[N*C*i]),
					.conn_weight = &(conn_weight[N*C*i]),
					.prev_conn_serial_index = &(prev_conn_serial_index[N*C*i]),
					.pre_conn_start_index = &(pre_conn_start_index[(N+1)*i]),
				};

				if(arg_is_monitor_performance)
				{
					network_list.emplace_back(i, move(serial_param), ((i==0)?(move(unique_ptr<AbstractTimeRecorder>(new TimeRecorder()))):(move(unique_ptr<AbstractTimeRecorder>(new AbstractTimeRecorder())))), log_param);
				}
				else
				{
					network_list.emplace_back(i, move(serial_param), move(unique_ptr<AbstractTimeRecorder>(new AbstractTimeRecorder())), log_param);
				}
			}

			if(is_run_parallel)
			{
				for(uint_fast32_t i=0; i<n_asyncs; i+=1)
				{
					thread_list.emplace_back(jthread([this, i](stop_token st) { this->work(st, i); }));
				}
			}
		}

		virtual ~NetworkGroup(void) noexcept
		{
			if(is_run_parallel)
			{
				complete_stop_threads();
			}

			::operator delete(v, align_val_t{ ALIGNMENT });
			::operator delete(u, align_val_t{ ALIGNMENT });
			::operator delete(current_I, align_val_t{ ALIGNMENT });
			::operator delete(post_conn_next_neuron, align_val_t{ ALIGNMENT });
			::operator delete(conn_weight, align_val_t{ ALIGNMENT });
			::operator delete(prev_conn_serial_index, align_val_t{ ALIGNMENT });
			::operator delete(pre_conn_start_index, align_val_t{ ALIGNMENT });
		}

		//public functions.
	public:

		void run
		(
			step_time_t arg_time_keep,
			vector< array<vector<neuron_t>, 2> >& neuron_list,
			vector< array<vector<elec_t>, 2> >& synaptic_current_list,
			vector<	array<multimap<neuron_t, neuron_t>, 2> >& strong_edge_list
		) noexcept;

		void serial_run
		(
			step_time_t arg_time_keep,
			vector< array<vector<neuron_t>, 2> >& neuron_list,
			vector< array<vector<elec_t>, 2> >& synaptic_current_list,
			vector<	array<multimap<neuron_t, neuron_t>, 2> >& strong_edge_list
		) noexcept;

		void parallel_run
		(
			step_time_t arg_time_keep,
			vector< array<vector<neuron_t>, 2> >& neuron_list,
			vector< array<vector<elec_t>, 2> >& synaptic_current_list,
			vector<	array<multimap<neuron_t, neuron_t>, 2> >& strong_edge_list
		) noexcept;

		void do_parallel_run
		(
			step_time_t arg_time_keep,
			vector< array< vector<neuron_t>, 2> >& neuron_list,
			vector< array< vector<elec_t>, 2> >& synaptic_current_list,
			vector< array<multimap<neuron_t, neuron_t>, 2> >& strong_edge_list,
			uint_fast32_t thread_serial_number
		) noexcept;

		void exchange_internetwork_signals
		(
			vector< array<vector<neuron_t>, 2> >& arg_neuron_list_list,
			vector< array<vector<elec_t>, 2> >& arg_synaptic_current_list_list
		) noexcept;

		//for thread management.
	private:
		void work(stop_token stoken, uint_fast32_t thread_serial_number) noexcept;

		void prepare_and_run_asyncs(const function<void(void)>&& func, const uint_fast32_t async_serial_number) noexcept;

		void wait_all_asyncs(void) noexcept;

		void complete_stop_threads(void) noexcept;
	};
}

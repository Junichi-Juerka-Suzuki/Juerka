#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

#include "NetworkGroup.h"
#include "Logger.h"
#include "WeightLogger.h"

namespace Juerka::Main
{
	void run(bool, bool, bool) noexcept;
};


int main(void) noexcept
{
	bool is_run_parallel(true);
	bool is_monitor_performance(false);
	bool is_record_weights(true);


	Juerka::Main::run
	(
		is_run_parallel,
		is_monitor_performance,
		is_record_weights
	);

	return EXIT_SUCCESS;
}

namespace Juerka::Main
{
	void run
	(
		bool is_run_parallel,
		bool is_monitor_performance,
		bool is_record_weights
	) noexcept
	{
		std::uint_fast32_t network_size(100);

		std::vector< std::array<std::vector<Juerka::CommonNet::neuron_t>, 2> > target_neuron_list(network_size);
		std::vector< std::array<std::vector<Juerka::CommonNet::elec_t>, 2> > synaptic_current_list(network_size);
		std::vector< std::array<std::multimap<Juerka::CommonNet::neuron_t, Juerka::CommonNet::neuron_t>, 2> > strong_edge_list(network_size);

		Juerka::CommonNet::NetworkGroup ng(network_size, is_run_parallel, is_monitor_performance, is_record_weights);
		Juerka::Utility::Logger logger(network_size);
		Juerka::Utility::WeightLogger weight_logger(network_size);

		for(Juerka::CommonNet::step_time_t i=0; i<Juerka::CommonNet::TIME_END; i+=1)
		{
			ng.run(i, target_neuron_list, synaptic_current_list, strong_edge_list);
			logger.log(i, target_neuron_list, synaptic_current_list);
			weight_logger.log(i, strong_edge_list);
			std::cout << i << std::endl;
		}
	}
}

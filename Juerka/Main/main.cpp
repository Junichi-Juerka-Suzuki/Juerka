#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "NetworkGroup.h"
#include "Logger.h"

namespace Juerka::Main
{
	void run(bool) noexcept;
};


int main(void) noexcept
{
	bool is_run_parallel(true);

	Juerka::Main::run(is_run_parallel);

	return EXIT_SUCCESS;
}

namespace Juerka::Main
{
	void run(bool is_run_parallel) noexcept
	{
		std::uint_fast32_t network_size(100);

		std::vector< std::array<std::vector<Juerka::CommonNet::neuron_t>, 2> > target_neuron_list(network_size);
		std::vector< std::array<std::vector<Juerka::CommonNet::elec_t>, 2> > synaptic_current_list(network_size);

		Juerka::CommonNet::NetworkGroup ng(network_size, is_run_parallel);
		Juerka::Utility::Logger logger(network_size);

		for(Juerka::CommonNet::step_time_t i=0; i<Juerka::CommonNet::TIME_END; i+=1)
		{
			ng.run(i, target_neuron_list, synaptic_current_list);
			logger.log(i, target_neuron_list, synaptic_current_list);

		}
	}
}

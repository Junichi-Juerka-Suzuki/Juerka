#include <array>
#include <vector>

#include "Common.h"
#include "Logger.h"

namespace Juerka::Utility
{
	using std::array;
	using std::vector;

	void Logger::log
	(
		Juerka::CommonNet::step_time_t time_keep,
		vector< array<vector<Juerka::CommonNet::neuron_t>, 2> >& target_neuron_list,
		vector< array<vector<Juerka::CommonNet::elec_t>, 2> >& synaptic_current_list
	)
	{
		(void) synaptic_current_list;

		size_t size_k(network_size);
		for(size_t k=0; k<size_k; k+=1)
		{
			auto& inner_vector(target_neuron_list[k][Juerka::CommonNet::OUTPUT_SIDE]);

			size_t size_j(inner_vector.size());

			for(size_t j=0; j<size_j; j+=1)
			{
				out_list[k] << time_keep << ' ' << inner_vector[j] << '\n';
			}
		}
	}
}

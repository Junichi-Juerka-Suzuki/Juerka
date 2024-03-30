#include <array>
#include <set>
#include <vector>

#include "Common.h"
#include "WeightLogger.h"

namespace Juerka::Utility
{
	using std::array;
	using std::set;
	using std::vector;

	void WeightLogger::log
	(
		Juerka::CommonNet::step_time_t time_keep,
		vector< array<set<Juerka::CommonNet::synapse_t>, 2> >& strong_edge_list
	)
	{
		size_t size_k(network_size);
		for(size_t k=0; k<size_k; k+=1)
		{
			auto& inner_set_0(strong_edge_list[k][Juerka::CommonNet::ADDITION_SIDE]);
			auto& inner_set_1(strong_edge_list[k][Juerka::CommonNet::SUBTRACTION_SIDE]);

			out_list[k] << time_keep << " added: ";
			for(auto it=inner_set_0.begin(); it!=inner_set_0.end(); it++)
			{
				out_list[k] << ' ' << *it;
			}
			out_list[k] << '\n';

			out_list[k] << time_keep << " deleted: ";
			for (auto it = inner_set_1.begin(); it != inner_set_1.end(); it++)
			{
				out_list[k] << ' ' << *it;
			}
			out_list[k] << '\n';
		}
	}
}

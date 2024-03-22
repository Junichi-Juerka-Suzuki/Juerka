#pragma once

#include <iostream>

#include <array>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Common.h"

namespace Juerka::Utility
{
	using std::array;
	using std::filesystem::create_directories;
	using std::move;
	using std::ofstream;
	using std::ostringstream;
	using std::put_time;
	using std::string;
	using std::time;
	using std::time_t;
	using std::to_string;
	using std::uint_fast32_t;
	using std::vector;

	class Logger
	{
		uint_fast32_t network_size;
		vector<ofstream> out_list;

		inline static constexpr char LOGFILE_PREFIX[] = "JUNLOG_";
		inline static constexpr char LOGFILE_SUFFIX[] = ".txt";

		inline static constexpr char LOGFILE_DIRECTORY[] = "log";

	public:

		Logger(uint_fast32_t arg_network_size) : network_size(arg_network_size)
		{
  			time_t now(time(nullptr));
			ostringstream oss;
			oss << LOGFILE_DIRECTORY << '/' << put_time(localtime(&now), "%Y%m%d%H%M%S");
			string time_directory(oss.str());

  			bool result(create_directories(time_directory));
  			(void) result;

			for(size_t i=0; i<network_size; i+=1)
			{
				out_list.emplace_back
				(
					move
					(
						ofstream
						(
							time_directory
							+
							string("/")
							+
							string(LOGFILE_PREFIX)
							+
							to_string(i)
							+
							string(LOGFILE_SUFFIX)
						)
					)
				);
			}
		}

		virtual ~Logger(void) { }

		void log
		(
			Juerka::CommonNet::step_time_t time_keep,
			vector< array<vector<Juerka::CommonNet::neuron_t>, 2> >& target_neuron_list,
			vector< array<vector<Juerka::CommonNet::elec_t>, 2> >& synaptic_current_list
		);
	};
}

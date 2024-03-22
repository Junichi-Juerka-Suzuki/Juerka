#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

namespace Juerka::Utility
{
	using std::cerr;
	using std::flush;
	using std::ostream;
	using std::vector;
	using std::scientific;
	using std::setprecision;

	class AbstractTimeRecorder
	{
	private:

	public:
		AbstractTimeRecorder(void)
		{
			//do nothing.
		}

		virtual ~AbstractTimeRecorder(void)
		{
			//do nothing.
		}

		AbstractTimeRecorder(AbstractTimeRecorder&&) = default;

		virtual void start_record(void)
		{
			//do nothing.
		}

		virtual void end_record(void)
		{
			//do nothing.
		}

		virtual void refresh_records(size_t arg_loop_index_to_restart=0, size_t arg_serial_index_to_restart=0)
		{
			(void)arg_loop_index_to_restart;
			(void)arg_serial_index_to_restart;
			//do nothing.
		}

		virtual void dump_records(void)
		{
			//do nothing.
		}

	};

	class TimeRecorder : public AbstractTimeRecorder
	{
	private:
		size_t serial_counter;

		vector<double> time_records;

		std::chrono::system_clock::time_point start, end;

		ostream& out;

	private:
		inline static constexpr size_t PRECISION = 6;

	public:
		TimeRecorder(ostream& arg_out=cerr)
		: AbstractTimeRecorder(),
		  serial_counter(0),
		  start(std::chrono::system_clock::now()),
		  end(std::chrono::system_clock::now()),
		  out(arg_out)
		{
			out << scientific << setprecision(PRECISION);
		}

		virtual ~TimeRecorder(void)
		{
			flush(out);
		}

		TimeRecorder(TimeRecorder&&) = default;

		void start_record(void) override
		{
			start = std::chrono::system_clock::now();
		}

		void end_record(void) override
		{
			end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;

			time_records.emplace_back(elapsed_seconds.count());
		}

		void refresh_records(size_t arg_loop_index_to_restart=0, size_t arg_serial_index_to_restart=0) override
		{
			time_records.clear();

			for(size_t i=0; i<arg_loop_index_to_restart; i+=1)
			{
				time_records.emplace_back(0.0);
			}

			serial_counter = arg_serial_index_to_restart;
		}

		void dump_records(void) override
		{
			out << "time_records: " << serial_counter;

			for(size_t i=0; i<time_records.size(); i+=1)
			{
				out << ' ' << time_records[i];
			}

			out << '\n';

			serial_counter += 1;
		}
	};
}

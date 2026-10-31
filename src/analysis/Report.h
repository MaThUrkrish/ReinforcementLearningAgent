#ifndef REMAP2D_REPORT_H
#define REMAP2D_REPORT_H

#include <string>
#include "RunLoader.h"

namespace remap2d {

// Generate a complete HTML report for a run and write it to
// <run_dir>/report.html. Returns the path to the report.
std::string generate_report(const RunData& data);

} // namespace remap2d

#endif // REMAP2D_REPORT_H

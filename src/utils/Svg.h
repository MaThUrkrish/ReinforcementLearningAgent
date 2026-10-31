#ifndef REMAP2D_SVG_H
#define REMAP2D_SVG_H

#include <string>
#include <vector>

namespace remap2d {

// ── Color helpers ───────────────────────────────────────────────────────────

struct Color { int r, g, b; };

// Convert color to CSS rgb string
std::string rgb_str(const Color& c);

// Jet-like colormap: blue → cyan → green → yellow → red
// t in [0, 1]
Color jet_colormap(float t);

// Diverging colormap for correlations: blue (−1) → white (0) → red (+1)
Color diverging_colormap(float v);

// Heat colormap: black → blue → green → yellow → red → white
Color heat_colormap(float t);

// ── SVG generation helpers ──────────────────────────────────────────────────

// Escape special XML characters
std::string xml_escape(const std::string& s);

// Generate an SVG heatmap from a 2D grid of values.
// width/height are in SVG units. Returns a complete <svg> element.
std::string svg_heatmap(const std::vector<float>& data,
                        int rows, int cols,
                        float min_val, float max_val,
                        int cell_size, bool use_jet = true);

// Generate an SVG line chart.
// series: one line per series. Each series is a vector of (x, y) points.
// Returns a complete <svg> element.
std::string svg_line_chart(const std::vector<std::vector<float>>& series,
                           const std::vector<std::string>& series_names,
                           const std::vector<std::string>& series_colors,
                           float x_min, float x_max,
                           float y_min, float y_max,
                           int width, int height,
                           const std::string& x_label = "",
                           const std::string& y_label = "");

// Generate an SVG bar chart.
// labels: bar labels, values: bar heights, color: bar color.
std::string svg_bar_chart(const std::vector<std::string>& labels,
                          const std::vector<float>& values,
                          float max_val,
                          int width, int height,
                          const std::string& color = "#4a90d9");

// Generate an SVG correlation matrix heatmap.
// labels: row/column labels. matrix: NxN values in [-1, 1].
std::string svg_correlation_matrix(const std::vector<std::string>& labels,
                                    const std::vector<std::vector<float>>& matrix,
                                    int cell_size);

} // namespace remap2d

#endif // REMAP2D_SVG_H

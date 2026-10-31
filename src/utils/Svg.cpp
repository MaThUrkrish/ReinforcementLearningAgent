#include "Svg.h"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace remap2d {

// ── Color helpers ───────────────────────────────────────────────────────────

std::string rgb_str(const Color& c) {
    std::ostringstream oss;
    oss << "rgb(" << c.r << "," << c.g << "," << c.b << ")";
    return oss.str();
}

Color jet_colormap(float t) {
    // Clamp to [0, 1]
    t = std::max(0.0f, std::min(1.0f, t));
    // 5-stop jet: blue(0) → cyan(0.25) → green(0.5) → yellow(0.75) → red(1.0)
    float r, g, b;
    if (t < 0.25f) {
        float s = t / 0.25f;
        r = 0.0f; g = s; b = 1.0f;            // blue → cyan
    } else if (t < 0.5f) {
        float s = (t - 0.25f) / 0.25f;
        r = 0.0f; g = 1.0f; b = 1.0f - s;     // cyan → green
    } else if (t < 0.75f) {
        float s = (t - 0.5f) / 0.25f;
        r = s; g = 1.0f; b = 0.0f;            // green → yellow
    } else {
        float s = (t - 0.75f) / 0.25f;
        r = 1.0f; g = 1.0f - s; b = 0.0f;     // yellow → red
    }
    return {static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255)};
}

Color diverging_colormap(float v) {
    // v in [-1, 1]: blue (−1) → white (0) → red (+1)
    v = std::max(-1.0f, std::min(1.0f, v));
    float intensity = std::abs(v);
    if (v >= 0) {
        // white → red
        return {255, static_cast<int>(255 * (1 - intensity)), static_cast<int>(255 * (1 - intensity))};
    } else {
        // white → blue
        return {static_cast<int>(255 * (1 - intensity)), static_cast<int>(255 * (1 - intensity)), 255};
    }
}

Color heat_colormap(float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    // black → blue → green → yellow → red → white
    float r, g, b;
    if (t < 0.2f) {
        float s = t / 0.2f;
        r = 0; g = 0; b = s;                   // black → blue
    } else if (t < 0.4f) {
        float s = (t - 0.2f) / 0.2f;
        r = 0; g = s; b = 1;                   // blue → cyan
    } else if (t < 0.6f) {
        float s = (t - 0.4f) / 0.2f;
        r = 0; g = 1; b = 1 - s;               // cyan → green
    } else if (t < 0.8f) {
        float s = (t - 0.6f) / 0.2f;
        r = s; g = 1; b = 0;                   // green → yellow
    } else {
        float s = (t - 0.8f) / 0.2f;
        r = 1; g = 1 - s * 0.5f; b = 0;       // yellow → red
    }
    return {static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255)};
}

// ── XML escape ──────────────────────────────────────────────────────────────

std::string xml_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '<':  out += "&lt;"; break;
            case '>':  out += "&gt;"; break;
            case '&':  out += "&amp;"; break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default:   out += c; break;
        }
    }
    return out;
}

// ── SVG Heatmap ─────────────────────────────────────────────────────────────

std::string svg_heatmap(const std::vector<float>& data,
                        int rows, int cols,
                        float min_val, float max_val,
                        int cell_size, bool use_jet)
{
    std::ostringstream svg;
    int w = cols * cell_size;
    int h = rows * cell_size;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w
        << "\" height=\"" << h << "\" viewBox=\"0 0 " << w << " " << h << "\">\n";

    float range = max_val - min_val;
    if (range < 1e-12f) range = 1.0f;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float val = data[r * cols + c];
            float t = (val - min_val) / range;
            Color col = use_jet ? jet_colormap(t) : heat_colormap(t);
            svg << "  <rect x=\"" << c * cell_size << "\" y=\"" << r * cell_size
                << "\" width=\"" << cell_size << "\" height=\"" << cell_size
                << "\" fill=\"" << rgb_str(col) << "\" stroke=\"#333\" stroke-width=\"0.5\"/>\n";
        }
    }
    svg << "</svg>";
    return svg.str();
}

// ── SVG Line Chart ──────────────────────────────────────────────────────────

std::string svg_line_chart(const std::vector<std::vector<float>>& series,
                           const std::vector<std::string>& series_names,
                           const std::vector<std::string>& series_colors,
                           float x_min, float x_max,
                           float y_min, float y_max,
                           int width, int height,
                           const std::string& x_label,
                           const std::string& y_label)
{
    std::ostringstream svg;
    int margin_l = 60, margin_r = 140, margin_t = 20, margin_b = 50;
    int plot_w = width - margin_l - margin_r;
    int plot_h = height - margin_t - margin_b;

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\">\n";
    svg << "  <rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\""
        << height << "\" fill=\"white\"/>\n";

    // Plot border
    svg << "  <rect x=\"" << margin_l << "\" y=\"" << margin_t
        << "\" width=\"" << plot_w << "\" height=\"" << plot_h
        << "\" fill=\"none\" stroke=\"#333\" stroke-width=\"1\"/>\n";

    // Y-axis grid lines and labels
    int n_yticks = 5;
    for (int i = 0; i <= n_yticks; ++i) {
        float y_val = y_min + (y_max - y_min) * i / n_yticks;
        int y = margin_t + plot_h - static_cast<int>(plot_h * i / n_yticks);
        svg << "  <line x1=\"" << margin_l << "\" y1=\"" << y
            << "\" x2=\"" << margin_l + plot_w << "\" y2=\"" << y
            << "\" stroke=\"#ddd\" stroke-width=\"0.5\"/>\n";
        svg << "  <text x=\"" << margin_l - 5 << "\" y=\"" << y + 4
            << "\" text-anchor=\"end\" font-size=\"10\" fill=\"#333\">"
            << std::fixed << std::setprecision(2) << y_val << "</text>\n";
    }

    // X-axis labels
    int n_xticks = 5;
    for (int i = 0; i <= n_xticks; ++i) {
        float x_val = x_min + (x_max - x_min) * i / n_xticks;
        int x = margin_l + static_cast<int>(plot_w * i / n_xticks);
        svg << "  <text x=\"" << x << "\" y=\"" << margin_t + plot_h + 15
            << "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#333\">"
            << std::fixed << std::setprecision(0) << x_val << "</text>\n";
    }

    // Axis labels
    if (!x_label.empty())
        svg << "  <text x=\"" << margin_l + plot_w / 2 << "\" y=\""
            << height - 5 << "\" text-anchor=\"middle\" font-size=\"12\" fill=\"#333\">"
            << xml_escape(x_label) << "</text>\n";
    if (!y_label.empty())
        svg << "  <text x=\"15\" y=\"" << margin_t + plot_h / 2
            << "\" text-anchor=\"middle\" font-size=\"12\" fill=\"#333\" transform=\"rotate(-90 15 "
            << margin_t + plot_h / 2 << ")\">" << xml_escape(y_label) << "</text>\n";

    // Plot series
    float x_range = x_max - x_min;
    float y_range = y_max - y_min;
    if (x_range < 1e-12f) x_range = 1.0f;
    if (y_range < 1e-12f) y_range = 1.0f;

    for (size_t s = 0; s < series.size(); ++s) {
        const auto& pts = series[s];
        if (pts.empty()) continue;
        std::string color = (s < series_colors.size()) ? series_colors[s] : "#333";
        svg << "  <polyline fill=\"none\" stroke=\"" << color
            << "\" stroke-width=\"1.5\" points=\"";
        for (size_t i = 0; i < pts.size(); ++i) {
            float x_val = static_cast<float>(i);
            float y_val = pts[i];
            int px = margin_l + static_cast<int>(plot_w * (x_val - x_min) / x_range);
            int py = margin_t + plot_h - static_cast<int>(plot_h * (y_val - y_min) / y_range);
            svg << px << "," << py;
            if (i + 1 < pts.size()) svg << " ";
        }
        svg << "\"/>\n";

        // Legend entry
        int ly = margin_t + 20 + static_cast<int>(s) * 18;
        svg << "  <line x1=\"" << margin_l + plot_w + 10
            << "\" y1=\"" << ly << "\" x2=\"" << margin_l + plot_w + 30
            << "\" y2=\"" << ly << "\" stroke=\"" << color << "\" stroke-width=\"2\"/>\n";
        std::string name = (s < series_names.size()) ? series_names[s] : ("series " + std::to_string(s));
        svg << "  <text x=\"" << margin_l + plot_w + 35 << "\" y=\""
            << ly + 4 << "\" font-size=\"11\" fill=\"#333\">"
            << xml_escape(name) << "</text>\n";
    }

    svg << "</svg>";
    return svg.str();
}

// ── SVG Bar Chart ───────────────────────────────────────────────────────────

std::string svg_bar_chart(const std::vector<std::string>& labels,
                          const std::vector<float>& values,
                          float max_val,
                          int width, int height,
                          const std::string& color)
{
    std::ostringstream svg;
    int margin_l = 80, margin_r = 20, margin_t = 20, margin_b = 60;
    int plot_w = width - margin_l - margin_r;
    int plot_h = height - margin_t - margin_b;

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\">\n";
    svg << "  <rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\""
        << height << "\" fill=\"white\"/>\n";

    // Y-axis
    int n_yticks = 5;
    for (int i = 0; i <= n_yticks; ++i) {
        float y_val = max_val * i / n_yticks;
        int y = margin_t + plot_h - static_cast<int>(plot_h * i / n_yticks);
        svg << "  <line x1=\"" << margin_l << "\" y1=\"" << y
            << "\" x2=\"" << margin_l + plot_w << "\" y2=\"" << y
            << "\" stroke=\"#ddd\" stroke-width=\"0.5\"/>\n";
        svg << "  <text x=\"" << margin_l - 5 << "\" y=\"" << y + 4
            << "\" text-anchor=\"end\" font-size=\"10\" fill=\"#333\">"
            << std::fixed << std::setprecision(2) << y_val << "</text>\n";
    }

    // Bars
    int n = static_cast<int>(labels.size());
    if (n == 0) { svg << "</svg>"; return svg.str(); }
    int bar_w = plot_w / n - 10;
    if (bar_w < 20) bar_w = 20;
    int total_bar_w = n * (bar_w + 10) - 10;
    int start_x = margin_l + (plot_w - total_bar_w) / 2;

    for (int i = 0; i < n; ++i) {
        float val = values[i];
        int bar_h = static_cast<int>(plot_h * val / max_val);
        int bx = start_x + i * (bar_w + 10);
        int by = margin_t + plot_h - bar_h;
        svg << "  <rect x=\"" << bx << "\" y=\"" << by
            << "\" width=\"" << bar_w << "\" height=\"" << bar_h
            << "\" fill=\"" << color << "\" stroke=\"#333\" stroke-width=\"0.5\"/>\n";
        // Value label
        svg << "  <text x=\"" << bx + bar_w / 2 << "\" y=\"" << by - 5
            << "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#333\">"
            << std::fixed << std::setprecision(3) << val << "</text>\n";
        // X label
        svg << "  <text x=\"" << bx + bar_w / 2 << "\" y=\""
            << margin_t + plot_h + 15 << "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#333\">"
            << xml_escape(labels[i]) << "</text>\n";
    }

    svg << "</svg>";
    return svg.str();
}

// ── SVG Correlation Matrix ──────────────────────────────────────────────────

std::string svg_correlation_matrix(const std::vector<std::string>& labels,
                                    const std::vector<std::vector<float>>& matrix,
                                    int cell_size)
{
    int n = static_cast<int>(labels.size());
    int margin = 80;
    int w = margin + n * cell_size + 20;
    int h = margin + n * cell_size + 60;  // extra for color bar

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w
        << "\" height=\"" << h << "\">\n";
    svg << "  <rect x=\"0\" y=\"0\" width=\"" << w << "\" height=\""
        << h << "\" fill=\"white\"/>\n";

    // Matrix cells
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float v = matrix[i][j];
            Color col = diverging_colormap(v);
            int cx = margin + j * cell_size;
            int cy = margin + i * cell_size;
            svg << "  <rect x=\"" << cx << "\" y=\"" << cy
                << "\" width=\"" << cell_size << "\" height=\"" << cell_size
                << "\" fill=\"" << rgb_str(col) << "\" stroke=\"#333\" stroke-width=\"0.5\"/>\n";
            // Value text
            svg << "  <text x=\"" << cx + cell_size / 2 << "\" y=\""
                << cy + cell_size / 2 + 4 << "\" text-anchor=\"middle\" font-size=\"9\" fill=\""
                << (std::abs(v) > 0.5 ? "white" : "#333") << "\">"
                << std::fixed << std::setprecision(2) << v << "</text>\n";
        }
    }

    // Column labels (top)
    for (int j = 0; j < n; ++j) {
        int cx = margin + j * cell_size + cell_size / 2;
        svg << "  <text x=\"" << cx << "\" y=\"" << margin - 5
            << "\" text-anchor=\"middle\" font-size=\"10\" fill=\"#333\">"
            << xml_escape(labels[j]) << "</text>\n";
    }
    // Row labels (left)
    for (int i = 0; i < n; ++i) {
        int cy = margin + i * cell_size + cell_size / 2 + 3;
        svg << "  <text x=\"" << margin - 5 << "\" y=\"" << cy
            << "\" text-anchor=\"end\" font-size=\"10\" fill=\"#333\">"
            << xml_escape(labels[i]) << "</text>\n";
    }

    // Color bar legend
    int bar_w = 150, bar_h = 12;
    int bar_x = margin;
    int bar_y = margin + n * cell_size + 20;
    for (int i = 0; i < bar_w; ++i) {
        float v = -1.0f + 2.0f * i / bar_w;
        Color col = diverging_colormap(v);
        svg << "  <rect x=\"" << bar_x + i << "\" y=\"" << bar_y
            << "\" width=\"1\" height=\"" << bar_h << "\" fill=\""
            << rgb_str(col) << "\"/>\n";
    }
    svg << "  <text x=\"" << bar_x << "\" y=\"" << bar_y + bar_h + 12
        << "\" text-anchor=\"middle\" font-size=\"9\" fill=\"#333\">-1</text>\n";
    svg << "  <text x=\"" << bar_x + bar_w / 2 << "\" y=\"" << bar_y + bar_h + 12
        << "\" text-anchor=\"middle\" font-size=\"9\" fill=\"#333\">0</text>\n";
    svg << "  <text x=\"" << bar_x + bar_w << "\" y=\"" << bar_y + bar_h + 12
        << "\" text-anchor=\"middle\" font-size=\"9\" fill=\"#333\">+1</text>\n";

    svg << "</svg>";
    return svg.str();
}

} // namespace remap2d

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <string>

using namespace std;

// Parabolic kernel K(u) = 0.75*(1-u^2), |u|<=1
double parabolic_kernel(double u) {
    if (fabs(u) <= 1.0) return 0.75 * (1.0 - u * u);
    return 0.0;
}

// Nadaraya-Watson estimate at x_query.
// skip_idx >= 0 — exclude point with this index (for leave-one-out)
double nadaraya_watson(const vector<double>& x, const vector<double>& y,
                       double x_query, double c, int skip_idx = -1) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        if ((int)i == skip_idx) continue;
        double k = parabolic_kernel((x_query - x[i]) / c);
        num += k * y[i];
        den += k;
    }
    if (den < 1e-15) return 0.0;   // no neighbors in window
    return num / den;
}

// LOO-MSE: sum E_j / n, where E_j = (y*(x_j) - y_j)^2 (point j excluded)
double loo_mse(const vector<double>& x, const vector<double>& y, double c) {
    double total = 0.0;
    int n = (int)x.size();
    for (int j = 0; j < n; ++j) {
        double y_pred = nadaraya_watson(x, y, x[j], c, j);
        double e = y_pred - y[j];
        total += e * e;
    }
    return total / n;
}

int main() {
    cout << fixed << setprecision(6);

    // =============== PART 1: synthetic data ===============
    cout << "==================== PART 1 ====================\n";
    const int N = 1000;
    vector<double> x1(N), y1(N);

    mt19937 gen(42);                       // fixed seed for reproducibility
    normal_distribution<double> norm(1.0, 0.05);  // NormRand(1, 0.05)

    for (int i = 0; i < N; ++i) {
        x1[i] = (i + 1) * 0.1;
        double noise = norm(gen);
        y1[i] = sin(x1[i]) + x1[i] / 10.0
                + 50.0 / log(x1[i] + 10.0) * noise;
    }

    // Search for smoothing parameter c
    vector<double> c_vals1 = {0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3,
                              0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0};
    cout << "\nExperiments with smoothing parameter c:\n";
    cout << setw(10) << "c" << setw(18) << "LOO-MSE\n";
    double best_c1 = 0.0, best_mse1 = 1e18;
    for (double c : c_vals1) {
        double mse = loo_mse(x1, y1, c);
        cout << setw(10) << c << setw(18) << mse << "\n";
        if (mse < best_mse1) { best_mse1 = mse; best_c1 = c; }
    }
    cout << "\n>>> Best c = " << best_c1
         << "   LOO-MSE = " << best_mse1 << "\n\n";

    // Save results for plotting
    ofstream f1("part1_recovery.csv");
    f1 << "x,y_clean,y_noisy,y_pred\n";
    for (int i = 0; i < N; ++i) {
        double y_pred  = nadaraya_watson(x1, y1, x1[i], best_c1);
        double y_clean = sin(x1[i]) + x1[i] / 10.0
                         + 50.0 / log(x1[i] + 10.0);
        f1 << x1[i] << "," << y_clean << "," << y1[i] << "," << y_pred << "\n";
    }
    f1.close();
    cout << "Part 1 results saved to part1_recovery.csv\n\n";

    // =============== PART 2: real data ===============
    cout << "==================== PART 2 ====================\n";
    // File part2_data.csv: 50 rows like "x,y" (e.g. area,price)
    vector<double> x2, y2;
    ifstream fin("part2_data.csv");
    if (!fin.is_open()) {
        cerr << "ERROR: file part2_data.csv not found\n"
             << "Create it with 50 rows like\n"
             << "  45.5,7800000\n"
             << "  62.0,9500000\n"
             << "  ...\n";
        return 1;
    }
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        // accept ',' or ';' as separator
        for (char& ch : line) if (ch == ';') ch = ',';
        size_t pos = line.find(',');
        if (pos == string::npos) continue;
        try {
            x2.push_back(stod(line.substr(0, pos)));
            y2.push_back(stod(line.substr(pos + 1)));
        } catch (...) {}
    }
    fin.close();

    cout << "Loaded points: " << x2.size() << "\n";
    if (x2.size() < 5) {
        cerr << "Not enough data to build regression.\n";
        return 1;
    }

    // Scale c range proportionally to the spread of x
    double xmin = *min_element(x2.begin(), x2.end());
    double xmax = *max_element(x2.begin(), x2.end());
    double span = xmax - xmin;
    cout << "x range: [" << xmin << ", " << xmax
         << "], span = " << span << "\n";

    vector<double> c_vals2;
    for (double f = 0.02; f <= 1.001; f += 0.02)
        c_vals2.push_back(f * span);

    cout << "\nExperiments with smoothing parameter c:\n";
    cout << setw(14) << "c" << setw(20) << "LOO-MSE\n";
    double best_c2 = 0.0, best_mse2 = 1e18;
    for (double c : c_vals2) {
        if (c <= 0) continue;
        double mse = loo_mse(x2, y2, c);
        cout << setw(14) << c << setw(20) << mse << "\n";
        if (mse < best_mse2) { best_mse2 = mse; best_c2 = c; }
    }
    cout << "\n>>> Best c = " << best_c2
         << "   LOO-MSE = " << best_mse2 << "\n\n";

    // Save results (sorted by x for plotting convenience)
    ofstream f2("part2_recovery.csv");
    f2 << "x,y,y_pred\n";
    vector<int> idx(x2.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    sort(idx.begin(), idx.end(), [&](int a, int b){ return x2[a] < x2[b]; });
    for (int i : idx) {
        double y_pred = nadaraya_watson(x2, y2, x2[i], best_c2);
        f2 << x2[i] << "," << y2[i] << "," << y_pred << "\n";
    }
    f2.close();
    cout << "Part 2 results saved to part2_recovery.csv\n";

    return 0;
}
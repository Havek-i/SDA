#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>

using namespace std;

// ==================== HELPER FUNCTIONS ====================

double get_noise(mt19937& gen, double sigma) {
    normal_distribution<double> dist(0.0, sigma);
    return dist(gen);
}

double true_y(const vector<double>& x, int s) {
    double y = 0.0;
    for (double xi : x) {
        double ui = s + xi;
        y += ui * ui;
    }
    return y;
}

vector<double> solve_gauss(vector<vector<double>> A, vector<double> b) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        int maxRow = i;
        for (int k = i + 1; k < n; ++k)
            if (fabs(A[k][i]) > fabs(A[maxRow][i])) maxRow = k;
        swap(A[i], A[maxRow]);
        swap(b[i], b[maxRow]);
        for (int k = i + 1; k < n; ++k) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) sum -= A[i][j] * x[j];
        x[i] = sum / A[i][i];
    }
    return x;
}

vector<double> least_squares(const vector<vector<double>>& X, const vector<double>& y) {
    int n = X.size();
    int p = X[0].size();
    vector<vector<double>> XtX(p, vector<double>(p, 0.0));
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j)
            for (int k = 0; k < n; ++k)
                XtX[i][j] += X[k][i] * X[k][j];
    vector<double> Xty(p, 0.0);
    for (int i = 0; i < p; ++i)
        for (int k = 0; k < n; ++k)
            Xty[i] += X[k][i] * y[k];
    return solve_gauss(XtX, Xty);
}

// ==================== DESIGN MATRIX PRINTER ====================

void print_design_matrix(const vector<vector<double>>& x, const vector<double>& y, int s, 
                         bool with_quadratic, bool with_interactions) {
    int N = x.size();
    int n = x[0].size();

    vector<string> headers;
    vector<vector<double>> cols;

    // x0
    headers.push_back("x0");
    cols.push_back(vector<double>(N, 1.0));

    // x1..xn
    for (int j = 0; j < n; ++j) {
        headers.push_back("x" + to_string(j + 1));
        vector<double> col(N);
        for (int i = 0; i < N; ++i) col[i] = x[i][j];
        cols.push_back(col);
    }

    // Квадраты
    if (with_quadratic) {
        for (int j = 0; j < n; ++j) {
            headers.push_back("x" + to_string(j + 1) + "^2");
            vector<double> col(N);
            for (int i = 0; i < N; ++i) col[i] = x[i][j] * x[i][j];
            cols.push_back(col);
        }
    }

    // Парные взаимодействия
    if (with_interactions) {
        for (int j1 = 0; j1 < n; ++j1) {
            for (int j2 = j1 + 1; j2 < n; ++j2) {
                headers.push_back("x" + to_string(j1 + 1) + "x" + to_string(j2 + 1));
                vector<double> col(N);
                for (int i = 0; i < N; ++i) col[i] = x[i][j1] * x[i][j2];
                cols.push_back(col);
            }
        }
    }

    cout << "\n--- Design Matrix X ---\n";
    cout << setw(5) << "Run";
    for (const string& h : headers) cout << setw(8) << h;
    for (int j = 0; j < n; ++j) cout << setw(8) << ("u" + to_string(j + 1));
    cout << setw(10) << "y" << "\n";
    cout << string(5 + 8 * (headers.size() + n) + 10, '-') << "\n";

    cout << fixed << setprecision(3);
    for (int i = 0; i < N; ++i) {
        cout << setw(5) << i + 1;
        for (const auto& col : cols) cout << setw(8) << col[i];
        for (int j = 0; j < n; ++j) cout << setw(8) << (s + x[i][j]);
        cout << setw(10) << y[i] << "\n";
    }
}

// ==================== PART 1: FULL FACTORIAL ====================

void full_factorial(int n, int s, double sigma, mt19937& gen) {
    int N = 1 << n;
    cout << "\n========== PART 1: FULL FACTORIAL, n = " << n << " ==========\n";
    cout << "Number of runs: " << N << "\n";

    vector<vector<double>> x(N, vector<double>(n));
    vector<double> y(N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n; ++j)
            x[i][j] = ((i >> j) & 1) ? 1.0 : -1.0;
        y[i] = true_y(x[i], s) + get_noise(gen, sigma);
    }

    // Для n=6 таблица со всеми парными будет слишком широкой, 
    // поэтому выводим взаимодействия только для n=2 и n=4.
    bool show_inter = (n <= 4);
    print_design_matrix(x, y, s, false, show_inter);

    // Расчет beta через МНК (все главные + парные, для n=2,4)
    // Для n=6 оставляем только главные, чтобы матрица не была вырожденной по размеру вывода
    int p = 1 + n;
    if (n <= 4) p += n * (n - 1) / 2;

    vector<vector<double>> X(N, vector<double>(p, 0.0));
    for (int i = 0; i < N; ++i) {
        X[i][0] = 1.0;
        for (int j = 0; j < n; ++j) X[i][1 + j] = x[i][j];
        if (n <= 4) {
            int idx = 1 + n;
            for (int j1 = 0; j1 < n; ++j1)
                for (int j2 = j1 + 1; j2 < n; ++j2)
                    X[i][idx++] = x[i][j1] * x[i][j2];
        }
    }
    vector<double> beta = least_squares(X, y);

    cout << "\nCoefficients (beta):\n";
    cout << fixed << setprecision(3);
    cout << "  beta_0 = " << setw(8) << beta[0] << " | Theory: " << 5 * n << "\n";
    for (int j = 0; j < n; ++j)
        cout << "  beta_" << j + 1 << " = " << setw(8) << beta[1 + j] << " | Theory: 4.000\n";

    if (n <= 4) {
        cout << "Pairwise interactions (should be ~0):\n";
        int idx = 1 + n;
        for (int j1 = 0; j1 < n; ++j1)
            for (int j2 = j1 + 1; j2 < n; ++j2)
                cout << "  beta_" << j1 + 1 << j2 + 1 << " = " << setw(8) << beta[idx++] << "\n";
    }

    // Доказательство, что тройное взаимодействие ~0 (для n=4)
    if (n == 4) {
        double b123 = 0.0;
        for (int i = 0; i < N; ++i) b123 += x[i][0] * x[i][1] * x[i][2] * y[i];
        b123 /= N;
        cout << "Check triple interaction beta_123 = " << b123 << " (should be ~0)\n";
    }

    cout << "alpha_i = beta_i / delta_u = beta_i (since delta_u = 1)\n";
}

// ==================== PART 2: FRACTIONAL REPLICAS ====================

void fractional_replica(int n, int k, int s, double sigma, mt19937& gen) {
    int base_n = n - k;
    int N = 1 << base_n;
    cout << "\n========== PART 2: FRACTIONAL REPLICA 2^(" << n << "-" << k << "), n = " << n << " ==========\n";
    cout << "Number of runs: " << N << "\n";

    vector<vector<double>> x(N, vector<double>(n));
    vector<double> y(N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < base_n; ++j)
            x[i][j] = ((i >> j) & 1) ? 1.0 : -1.0;

        if (n == 4 && k == 1) {
            x[i][3] = x[i][0] * x[i][1] * x[i][2];
        } else if (n == 6 && k == 2) {
            x[i][4] = x[i][0] * x[i][1] * x[i][2];
            x[i][5] = x[i][0] * x[i][1] * x[i][3];
        }

        y[i] = true_y(x[i], s) + get_noise(gen, sigma);
    }

    // Выводим матрицу. Для n=4 покажем, что x4 == x1x2x3
    print_design_matrix(x, y, s, false, false);
    if (n == 4) {
        cout << "\nNote: column x4 equals x1*x2*x3 (generating relation):\n";
        for (int i = 0; i < N; ++i) {
            double x123 = x[i][0] * x[i][1] * x[i][2];
            cout << "  Run " << i+1 << ": x4 = " << x[i][3] << ", x1x2x3 = " << x123;
            if (fabs(x[i][3] - x123) < 1e-9) cout << "  -> match\n";
            else cout << "  -> MISMATCH!\n";
        }
    }

    double beta0 = 0.0;
    for (double v : y) beta0 += v;
    beta0 /= N;

    vector<double> beta(n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < N; ++i) beta[j] += x[i][j] * y[i];
        beta[j] /= N;
    }

    cout << "\nCoefficients (beta):\n";
    cout << fixed << setprecision(3);
    cout << "  beta_0 = " << setw(8) << beta0 << " | Theory: " << 5 * n << "\n";
    for (int j = 0; j < n; ++j)
        cout << "  beta_" << j + 1 << " = " << setw(8) << beta[j] << " | Theory: 4.000\n";
}

// ==================== PART 3: CENTRAL COMPOSITE DESIGN ====================

void central_composite(int s, double sigma, mt19937& gen) {
    cout << "\n========== PART 3: CENTRAL COMPOSITE DESIGN (n = 2) ==========\n";

    double a = sqrt(2.0);
    int n_center = 5;
    int N = 4 + 4 + n_center;

    cout << "Star arm a = " << a << "\n";
    cout << "Number of runs: " << N << " (4 factorial + 4 star + " << n_center << " center)\n";

    vector<vector<double>> X(N, vector<double>(6));
    vector<double> y(N);
    vector<vector<double>> x_plan(N, vector<double>(2));

    int row = 0;
    for (double x1 : {-1.0, 1.0}) {
        for (double x2 : {-1.0, 1.0}) {
            x_plan[row] = {x1, x2};
            X[row][0] = 1; X[row][1] = x1; X[row][2] = x2;
            X[row][3] = x1 * x1; X[row][4] = x2 * x2; X[row][5] = x1 * x2;
            y[row] = true_y(x_plan[row], s) + get_noise(gen, sigma);
            row++;
        }
    }

    vector<pair<double, double>> star = { {-a, 0}, {a, 0}, {0, -a}, {0, a} };
    for (auto [x1, x2] : star) {
        x_plan[row] = {x1, x2};
        X[row][0] = 1; X[row][1] = x1; X[row][2] = x2;
        X[row][3] = x1 * x1; X[row][4] = x2 * x2; X[row][5] = x1 * x2;
        y[row] = true_y(x_plan[row], s) + get_noise(gen, sigma);
        row++;
    }

    vector<double> y_center(n_center);
    for (int i = 0; i < n_center; ++i) {
        x_plan[row] = {0.0, 0.0};
        X[row][0] = 1; X[row][1] = 0; X[row][2] = 0;
        X[row][3] = 0; X[row][4] = 0; X[row][5] = 0;
        y[row] = true_y(x_plan[row], s) + get_noise(gen, sigma);
        y_center[i] = y[row];
        row++;
    }

    print_design_matrix(x_plan, y, s, true, true);

    vector<double> beta = least_squares(X, y);

    double delta_u = 1.0;
    cout << "\nQuadratic model coefficients (beta):\n";
    cout << fixed << setprecision(4);
    cout << "  beta_0  = " << setw(9) << beta[0] << "\n";
    cout << "  beta_1  = " << setw(9) << beta[1] << " (theory: 4.0)\n";
    cout << "  beta_2  = " << setw(9) << beta[2] << " (theory: 4.0)\n";
    cout << "  beta_11 = " << setw(9) << beta[3] << " (theory: 1.0)\n";
    cout << "  beta_22 = " << setw(9) << beta[4] << " (theory: 1.0)\n";
    cout << "  beta_12 = " << setw(9) << beta[5] << " (theory: 0.0)\n";

    cout << "\nCheck at center (u1 = u2 = " << s << "):\n";
    cout << "  True y = " << 2 * s * s << "\n";
    cout << "  Model beta_0 = " << beta[0] << "\n";

    double yc_mean = 0;
    for (double v : y_center) yc_mean += v;
    yc_mean /= n_center;

    double sigma2 = 0;
    for (double v : y_center) sigma2 += (v - yc_mean) * (v - yc_mean);
    sigma2 /= (n_center - 1);

    cout << "\nEstimated variance from center points: sigma^2 = " << sigma2 << "\n";
    cout << "Estimated sigma = " << sqrt(sigma2) << " (true 0.5)\n";

    cout << "\nalpha for quadratic model (delta_u = 1):\n";
    cout << "  alpha_0  = " << beta[0] << "\n";
    cout << "  alpha_1  = " << beta[1] / delta_u << "\n";
    cout << "  alpha_2  = " << beta[2] / delta_u << "\n";
    cout << "  alpha_11 = " << beta[3] / (delta_u * delta_u) << "\n";
    cout << "  alpha_22 = " << beta[4] / (delta_u * delta_u) << "\n";
    cout << "  alpha_12 = " << beta[5] / (delta_u * delta_u) << "\n";
}

// ==================== MAIN ====================

int main() {
    int s = 2;
    double sigma = 0.5;
    mt19937 gen(42);

    cout << "==================================================\n";
    cout << " Lab: Design of Experiments\n";
    cout << " Variant s = " << s << ", sigma = " << sigma << "\n";
    cout << "==================================================\n";

    full_factorial(2, s, sigma, gen);
    full_factorial(4, s, sigma, gen);
    full_factorial(6, s, sigma, gen);

    fractional_replica(4, 1, s, sigma, gen);
    fractional_replica(6, 2, s, sigma, gen);

    central_composite(s, sigma, gen);

    return 0;
}
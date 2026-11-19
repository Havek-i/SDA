#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>
// #include <locale.h>
#include <windows.h>

using namespace std;

// Функция генерации нормального шума
double get_noise(mt19937& gen, double sigma) {
    normal_distribution<double> dist(0.0, sigma);
    return dist(gen);
}

// Функция вычисления значения объекта y(u)
double compute_y(const vector<int>& x, int s) {
    double y = 0.0;
    for (int xi : x) {
        double ui = s + xi; // так как delta_u = 1
        y += ui * ui;
    }
    return y;
}

// Функция решения задачи для заданного n
void solve_experiment(int n, int s, double sigma) {
    int N = 1 << n; // 2^n опытов
    cout << "====================================================\n";
    cout << " ПФЭ для n = " << n << " (Опытов: " << N << ")\n";
    cout << " Центр плана u0 = " << s << ", Delta u = 1\n";
    cout << "====================================================\n";

    // Матрица плана (x) и вектор откликов (y)
    vector<vector<int>> plan(N, vector<int>(n));
    vector<double> y(N);
    
    mt19937 gen(42); // Фиксированный seed для воспроизводимости

    // 1. Составление плана и проведение эксперимента
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < n; ++j) {
            // Формируем комбинации -1 и +1 через биты
            plan[i][j] = ((i >> j) & 1) ? 1 : -1;
        }
        double y_true = compute_y(plan[i], s);
        double h = get_noise(gen, sigma);
        y[i] = y_true + h;
    }

    // 2. Расчет коэффициентов beta
    // beta_0
    double beta_0 = 0.0;
    for (double val : y) beta_0 += val;
    beta_0 /= N;

    cout << fixed << setprecision(3);
    cout << "Свободный член:\n";
    cout << "  beta_0 = " << setw(8) << beta_0 << " | Теоретически: " << 5 * n << "\n";

    cout << "Линейные эффекты (главные эффекты):\n";
    for (int j = 0; j < n; ++j) {
        double beta_j = 0.0;
        for (int i = 0; i < N; ++i) {
            beta_j += plan[i][j] * y[i];
        }
        beta_j /= N;
        cout << "  beta_" << j + 1 << " = " << setw(8) << beta_j << " | Теоретически: 4.000\n";
    }

    // Расчет взаимодействий (покажем для парных, чтобы доказать, что их нет)
    if (n >= 2) {
        cout << "Парные взаимодействия (должны быть ~0 из-за шума):\n";
        for (int j1 = 0; j1 < n; ++j1) {
            for (int j2 = j1 + 1; j2 < n; ++j2) {
                double beta_j1j2 = 0.0;
                for (int i = 0; i < N; ++i) {
                    beta_j1j2 += plan[i][j1] * plan[i][j2] * y[i];
                }
                beta_j1j2 /= N;
                cout << "  beta_" << j1 + 1 << j2 + 1 << " = " << setw(8) << beta_j1j2 << " | Теоретически: 0.000\n";
            }
        }
    }
    cout << "\n";
}

int main() {
        SetConsoleCP(65001);       // Установка кодировки UTF-8 на ввод в консоль
    SetConsoleOutputCP(65001);
    // setlocale(LC_ALL,"Russian");
    int s = 2;             // Номер варианта
    double sigma = 0.5;    // Среднеквадратическое отклонение помехи
    
    // Решаем для n = 2, 4, 6 как требуется в задании
    solve_experiment(2, s, sigma);
    solve_experiment(4, s, sigma);
    solve_experiment(6, s, sigma);

    return 0;
}
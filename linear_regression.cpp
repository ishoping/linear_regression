#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <string>
#include <cmath>
#include <ctime>
#include <sstream>

using namespace std;


double hypothesis(double *theta, double *x, int num_cols) {
	double z = 0;
	for (int i = 0; i < num_cols; i++) {
		z += x[i] * theta[i];
		//cout << "x[i]" << x[i] << endl;
	}
	//cout << "z: " << z << endl;
	return z;
}

double cost_function(double **x, double *y, double *theta, int num_rows, int num_cols, double lda) {
	double cost = 0;
	for (int i = 0; i < num_rows; i++) {
		double h = hypothesis(theta, x[i], num_cols);
		cost += (h - y[i]) * (h - y[i]);
		// if (fabs(h) < 0.0000001 || fabs(1 - h) < 0.0000001) {
		// 	cout << "h: " << h << endl;
		// }
	}

	return sqrt(cost / num_rows);

	// double reg_sum = 0;

	// for (int j = 0; j < num_cols; j++) {
	// 	reg_sum += theta[j] * theta[j];
	// }
	// return (cost + lda * reg_sum) / (2 * num_rows);
}

double cost_function_derivative(double **x, double *y, double *theta, int j, int num_rows, int num_cols) {
	double sum = 0;

	for (int i = 0; i < num_rows; i++) {
		double h = hypothesis(theta, x[i], num_cols);
		sum += (h - y[i]) * x[i][j];
	}

	return sum;
}

void linear_regression(double **x, double *y, double* &theta, double alpha, double lda, int num_rows, int num_cols, int num_iters) {
	double *tmp_theta = (double*)malloc(num_cols * sizeof(double));
	for (int i = 0; i < num_iters; i++) {

		for (int j = 0; j < num_cols; j++) {
			tmp_theta[j] = theta[j] * (1 - alpha * lda / num_rows) - alpha / num_rows * cost_function_derivative(x, y, theta, j, num_rows, num_cols);
		}

		double *tmp = theta;
		theta = tmp_theta;
		tmp_theta = tmp;
		tmp = NULL;

		if (i % 1000 == 0) {
			// for (int j = 0; j < num_cols; j++) {
			// 	cout << "theta: "  << theta[j] << " ";
			// }
			// cout << endl;
			cout << "cost_func: " << cost_function(x, y, theta, num_rows, num_cols, lda) << endl;
		}
	}
	//cout << "cost_func: " << cost_function(x, y, theta, num_rows, num_cols, lda) << endl;
	free(tmp_theta);
}



vector<string> split(const string& str, const string& delim) {
	vector<string> res;
	if("" == str) return res;
	char * strs = new char[str.length() + 1];
	strcpy(strs, str.c_str()); 

	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char *p = strtok(strs, d);

	while(p) {
		string s = p;
		res.push_back(s);
		p = strtok(NULL, d);
	}

	free(strs);
	free(d);
	free(p);
	return res;
}

void read_data(string filename, double **x, double *y, int flag = 1) {
	ifstream in;
	in.open(filename, ios::in);
	string line;
	if (in.is_open()) {
		int cur_row = 0;
		vector<string> sub_str;
		vector<string> tmp;

		getline(in, line);
		while (getline(in, line)) {
			sub_str = split(line, ",");
			y[cur_row] = stod(sub_str[sub_str.size() - 1]);
			x[cur_row][0] = 1;
			for (int i = 1; i < sub_str.size() - flag; i++) {
				x[cur_row][i] = stod(sub_str[i]);
			}
			cur_row++;
		}
	}

	in.close();
}

void predict(double **x, double *y, double *theta, int num_rows, int num_cols) {
	for (int i = 0; i < num_rows; i++) {
		y[i] = hypothesis(theta, x[i], num_cols);
	}
}


int main() {
	time_t start;
	time_t end;

	int num_rows = 25000;
	int num_cols = 384 + 1;

	double **x;
	double *y;

	x = (double**)malloc(num_rows * sizeof(double*));
	for (int i = 0; i < num_rows; i++) {
		x[i] = (double*)malloc(num_cols * sizeof(double));
		memset(x[i], 0, num_cols * sizeof(double));
	}

	y = (double*)malloc(num_rows * sizeof(double));
	memset(y, 0, num_rows * sizeof(double));


	read_data("train.csv", x, y);


    double *theta = (double*)malloc(num_cols * sizeof(double));
	memset(theta, 0, num_cols * sizeof(double));

	double alpha = 5;
	double lda = 2;
	int num_iters = 100001;

	start = clock();
	linear_regression(x, y, theta, alpha, lda, num_rows, num_cols, num_iters);
	end = clock();

	cout << "time: " << (end - start) / CLOCKS_PER_SEC << endl;

	// for (int i = 0; i < num_cols; i++) {
	// 	cout << "theta: " << theta[i] << endl;
	// }

	for (int i = 0; i < num_rows; i++) {
		free(x[i]);
	}
	free(x);
	free(y);


	double **test_x;
	double *test_y;
	int test_rows = 25000;
	int test_cols = num_cols + 1;
	
	test_x = (double**)malloc(test_rows * sizeof(double*));
	for (int i = 0; i < test_rows; i++) {
		test_x[i] = (double*)malloc(test_cols * sizeof(double));
	}

	test_y = (double*)malloc(test_rows * sizeof(double));

	read_data("test.csv", test_x, test_y, 0);
	predict(test_x, test_y, theta, test_rows, test_cols);

	// for (int i = 0; i < test_cols; i++) {
	// 	cout << "theta: " << theta[i] << endl;
	// }

	ofstream out;
	out.open("result.csv");
	out << "id,reference" << endl;
	for (int i = 0; i < test_rows; i++) {
		out << i << "," << test_y[i] << endl;
	}
	out.close();
	for (int i = 0; i < test_rows; i++) {
		free(test_x[i]);
	}
	free(test_x);
	free(test_y);
	free(theta);


    return 0;
}
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include "mpi.h"
#include "defs.h"

using namespace std;

double ic(const double x, const double y) /*initial conditions*/
{
	if ((x < accuracy) || (y < accuracy) || (fabs(x - 1) < accuracy) || (fabs(y - 1) < accuracy)) return 0; /*border conditions*/
	else
		return 0; /*inside conditions*/
}
double f(const double x, const double y) /*right side*/
{
	double spiy = sin(pi * y);
	return (2.0 * spiy + (l2 + pi2) * ((1.0 - x) * x * spiy));

}
double f_a(const double x, const double y)/*analitical solution*/
{
	return ((1 - x) * x * sin(pi * y));
}

void print_info(const int id, const int N, const int np, const double MAX_G, const double MAX_L, const int counter, const double t1, const double t2){
	if (id == 0)
	{
		cout << endl;

	if(type == 1){
		cout << "Send - Recv" << std::endl;
		}
	if(type == 2){
		cout << "SendRecv" << std::endl;
		}
	if(type == 3){
		cout << "I_Send - I_Recv" << std::endl;
		}

		cout << "N = " << N << ", number of processors = " << np << endl;
		cout << "Local error = " << MAX_G << endl;
		cout << "Global error = " << MAX_L << endl;
		cout << "Number of iterations: " << counter << endl;
		cout << "Parallel algorithm took : " << (t2 - t1) << " seconds" << endl;
		cout << endl;
	}
}
/*difference between numerical and analitical solutions in point*/
double analitical_difference(const vector<double>& c_M, const int n_loc, const int m_len, const int id, const int np)
{
	double max = 0.;
	double temp = 0.;
	for (int i = 0; i < m_len; ++i)
		for (int j = 0; j < N; ++j){
			temp = fabs(c_M[i * N + j] - f_a((id * n_loc + i) * h, j * h));
			if (temp > max) max = temp;
		}
	return max;

}

/*difference between two nearest layers in point*/
double local_difference(const vector<double>& M, const vector<double>& M1, const int n, const int m)/*local error*/
{
	double max = 0.;
	double temp = 0.;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j){
			temp = fabs(M1[i * m + j] - M[i * m + j]);
			if (max < temp) max = temp;
		}
	return max;
}

/*Send - Recv communication between processes
*@param current matrix, upper vector, down vector, lenght, process ID, number of processors
*/
void f_send_recv(vector<double>& c_M, vector<double>& d_M, vector<double>& u_M, const int n_loc, const int id, const int np)
{ 
	MPI_Status mps;
	if (id != np - 1) /*if its not the last process, send last line to the next(lower) process*/
	{
		MPI_Send(c_M.data() + (n_loc - 1) * N, N, MPI_DOUBLE, id + 1, 1, MPI_COMM_WORLD);
		MPI_Recv(d_M.data(), N, MPI_DOUBLE, id + 1, 2, MPI_COMM_WORLD, &mps);
	}

	if (id != 0) /*if its not the firts process, send first line to the previous(upper) process**/
	{
		MPI_Recv(u_M.data(), N, MPI_DOUBLE, id - 1, 1, MPI_COMM_WORLD, &mps);
		MPI_Send(c_M.data(), N, MPI_DOUBLE, id - 1, 2, MPI_COMM_WORLD);
	}
}

/*Sendrecv communication between processes
*@param current matrix, upper vector, down vector, lenght, process ID, number of processors
*/
void f_sendrecv(vector<double>& c_M, vector<double>& d_M, vector<double>& u_M, const int n_loc, const int id, const int np)
{
	MPI_Status mps;

	int t_disp = 50; /*just a big number for ID's*/
	int sen, rec, up_N, d_N; /*sender, reciever, size of vectors to be send up and down*/

	if (id == 0) /*if its first process we send zero bytes to the last process*/
	{
		d_N = 0;
		sen = np - 1;
	}
	else /*else we send line to previous(higher) process*/
	{
		d_N = N;
		sen = id - 1;
	}

	if (id == np - 1) /*if its last process we send zero bytes to the first process*/
	{
		up_N = 0;
		rec = 0;
	}
	else /*else we send line to the next(lower) process*/
	{
		up_N = N;
		rec = id + 1;
	}

	MPI_Sendrecv(c_M.data() + (n_loc - 1) * N, up_N, MPI_DOUBLE, rec, 0 + rec, u_M.data(), \
		d_N, MPI_DOUBLE, sen, 0 + id, MPI_COMM_WORLD, &mps); /*rec process recieves last(lower) line from previous process(sen)*/

	MPI_Sendrecv(c_M.data(), d_N, MPI_DOUBLE, sen, t_disp + sen, d_M.data(), \
		up_N, MPI_DOUBLE, rec, t_disp + id, MPI_COMM_WORLD, &mps);/*sen process recieves first(higher) line from next process(rec)*/
}
/*Starts sending*/
void f_isend_irecv(MPI_Request* request)
{ 
	MPI_Start(&request[0]);
	MPI_Start(&request[1]);
	MPI_Start(&request[2]);
	MPI_Start(&request[3]);
}

/*Iterative process itself*/
void zeydel_isendrecv(const int id, const int np)
{
	int m_len, n_loc = N / np; /*lenght of middle blocks*/

	vector<int> len(np);

	if (id == 0)
	{
		int n = 0;
		for (int i = 0; i < np - 1; ++i)
			len[i] = N / np;
		len[np - 1] = N - len[0] * (np - 1); /*last process lenght*/
	}

	MPI_Scatter(len.data(), 1, MPI_INT, &m_len, 1, MPI_INT, 0, MPI_COMM_WORLD); /*send vector with sizes to all processes*/

	vector<double> c_M; /*current local vec*/
	vector<double> n_M; /*new local vec*/
	vector<double> u_M(N); /*upper line*/
	vector<double> d_M(N); /*lower line*/

	double MAX_L, MAX_G; /*errors*/

	double t1 = MPI_Wtime();

	int sign;
	(n_loc % 2 != 0) ?
		((id % 2 == 0) ? sign = 1 : sign = 0) : sign = 0; /*here we count if we do right or red iterations*/

	int ds = N - (N / np) * np; /*thats what we add to last line*/
	(id == (np - 1) ? ds = ds : ds = 0); /*ds = 0 for all exept last*/

	if (id == 0)
		for (int i = 0; i < n_loc + 1; ++i)
			for (int j = 0; j < N; ++j)
				c_M.push_back(ic(i * h, j * h)); /*initilize first line*/
	else
		if (id == (np - 1))
			for (int i = id * n_loc; i < ((np - 1) * n_loc) + m_len; ++i)
				for (int j = 0; j < N; ++j)
					c_M.push_back(ic(i * h, j * h)); /*initilize last line*/
		else
			for (int i = id * n_loc; i < (id + 1) * n_loc; ++i)
				for (int j = 0; j < N; ++j)
					c_M.push_back(ic(i * h, j * h)); /*initilize cental line*/

	n_M = c_M; 
	/*create requests*/
	MPI_Request* request1 = new MPI_Request[4];
	MPI_Request* request2 = new MPI_Request[4];

	MPI_Status* status = new MPI_Status[4];

	int sen, rec, up_N, d_N;

		if (id == 0)
		{
			d_N = 0;
			sen = np - 1;
		}
		else
		{
			d_N = N;
			sen = id - 1;
		}

		if (id == np - 1)
		{
			up_N = 0;
			rec = 0;
		}
		else
		{
			up_N = N;
			rec = id + 1;
		}
		/*create map*/
		MPI_Send_init(c_M.data() + (n_loc - 1) * N, up_N, MPI_DOUBLE, rec, 0, MPI_COMM_WORLD, &request1[0]);
		MPI_Send_init(c_M.data(), d_N, MPI_DOUBLE, sen, 0, MPI_COMM_WORLD, &request1[1]);

		MPI_Recv_init(u_M.data(), d_N, MPI_DOUBLE, sen, 0, MPI_COMM_WORLD, &request1[2]);
		MPI_Recv_init(d_M.data(), up_N, MPI_DOUBLE, rec, 0, MPI_COMM_WORLD, &request1[3]);


		MPI_Send_init(n_M.data() + (n_loc - 1) * N, up_N, MPI_DOUBLE, rec, 0, MPI_COMM_WORLD, &request2[0]);
		MPI_Send_init(n_M.data(), d_N, MPI_DOUBLE, sen, 0, MPI_COMM_WORLD, &request2[1]);

		MPI_Recv_init(u_M.data(), d_N, MPI_DOUBLE, sen, 0, MPI_COMM_WORLD, &request2[2]);
		MPI_Recv_init(d_M.data(), up_N, MPI_DOUBLE, rec, 0, MPI_COMM_WORLD, &request2[3]);

	int counter = 0;

	do
	{
		f_isend_irecv((counter % 2 == 0) ? request1 : request2); /*send border lines*/
		
		for (int i = 1; i < m_len - 1; ++i) /*do even iterations*/
			for (int j = 1 + (i + sign) % 2; j < N - 1; j += 2)
				n_M[i * N + j] = ((c_M[(i + 1) * N + j] + c_M[(i - 1) * N + j] + c_M[i * N + j + 1] + c_M[i * N + j - 1]) \
					+ f((id * n_loc + i) * h, j * h) * H) * znam; 

		MPI_Waitall(4, (counter % 2 == 0) ? request1 : request2, status); /*wait till all processes ends sending*/

		if (id != 0)
			for (int j = 1 + (sign % 2); j < N - 1; j += 2) /*calc highest line*/ 
				n_M[j] = (c_M[N + j] + u_M[j] + c_M[j + 1] + c_M[j - 1] + f(id * n_loc * h, j * h) * H) * znam;

		if (id != np - 1) /*calc lowest line*/ 
			for (int j = 1 + (n_loc - 1 + sign) % 2; j < N - 1; j += 2)
				n_M[(n_loc - 1) * N + j] = (d_M[j] + c_M[(n_loc - 2) * N + j] + c_M[(n_loc - 1) * N + j + 1] + c_M[(n_loc - 1) * N + j - 1] \
					+ f((id * n_loc + n_loc - 1) * h, j * h) * H) * znam;

		f_isend_irecv((counter % 2 == 0) ? request2 : request1); /*send border lines again*/
		

		for (int i = 1; i < m_len - 1; ++i) /*do odd iterations*/
			for (int j = 1 + (i + sign + 1) % 2; j < N - 1; j += 2)
				n_M[i * N + j] = ((n_M[(i + 1) * N + j] + n_M[(i - 1) * N + j] + n_M[i * N + j + 1] + n_M[i * N + j - 1]) \
					+ f((id * n_loc + i) * h, j * h) * H) * znam;

		MPI_Waitall(4, (counter % 2 == 0) ? request2 : request1, status);/*wait till all processes ends sending*/


		if (id != 0)
			for (int j = 1 + (sign + 1) % 2; j < N - 1; j += 2) /*calc highest line*/ 
				n_M[j] = (n_M[N + j] + u_M[j] + n_M[j + 1] + n_M[j - 1] + f(id * n_loc * h, j * h) * H) * znam;

		if (id != np - 1) /*calc lowest line*/ 
			for (int j = 1 + (n_loc - 1 + sign + 1) % 2; j < N - 1; j += 2)
				n_M[(n_loc - 1) * N + j] = (d_M[j] + n_M[(n_loc - 2) * N + j] + n_M[(n_loc - 1) * N + j + 1] + n_M[(n_loc - 1) * N + j - 1] \
					+ f((id * n_loc + m_len - 1) * h, j * h) * H) * znam;

		counter++; 

		MAX_L = local_difference(c_M, n_M, m_len, N); /*calculate local diff*/
		MPI_Reduce(&MAX_L, &MAX_G, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /*calc max diff and send it to root process*/
		MPI_Bcast(&MAX_G, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); /*send this max value to all processes*/

		c_M.swap(n_M); /*swap old and new solutions*/

	} while (MAX_G > eps);

	double t2 = MPI_Wtime();


	double norm;
	norm  = analitical_difference(c_M, n_loc, m_len, id, np); /*calc diff between numerical and analitical solutions*/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&norm, &MAX_L, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /*calc max diff and send it to root process*/

	print_info(id, N, np,  MAX_G,  MAX_L,  counter,  t1,  t2);
}

void zeydel_Send_Recv(const int id, const int np)
{
	int m_len, n_loc = N / np; /*lenght of middle blocks*/

	vector<int> len(np);

	if (id == 0)
	{
		int n = 0;
		for (int i = 0; i < np - 1; ++i)
			len[i] = N / np;
		len[np - 1] = N - len[0] * (np - 1); /*last process lenght*/
	}

	MPI_Scatter(len.data(), 1, MPI_INT, &m_len, 1, MPI_INT, 0, MPI_COMM_WORLD); /*send vector with sizes to all processes*/

	vector<double> c_M; /*current local vec*/
	vector<double> n_M; /*new local vec*/
	vector<double> u_M(N); /*upper line*/
	vector<double> d_M(N); /*lower line*/

	double MAX_L, MAX_G; /*errors*/

	double t1 = MPI_Wtime();

	int sign;
	(n_loc % 2 != 0) ?
		((id % 2 == 0) ? sign = 1 : sign = 0) : sign = 0; /*here we count if we do right or red iterations*/

	int ds = N - (N / np) * np; /*thats what we add to last line*/
	(id == (np - 1) ? ds = ds : ds = 0); /*ds = 0 for all exept last*/

	if (id == 0)
		for (int i = 0; i < n_loc + 1; ++i)
			for (int j = 0; j < N; ++j)
				c_M.push_back(ic(i * h, j * h)); /*initilize first line*/
	else
		if (id == (np - 1))
			for (int i = id * n_loc; i < ((np - 1) * n_loc) + m_len; ++i)
				for (int j = 0; j < N; ++j)
					c_M.push_back(ic(i * h, j * h)); /*initilize last line*/
		else
			for (int i = id * n_loc; i < (id + 1) * n_loc; ++i)
				for (int j = 0; j < N; ++j)
					c_M.push_back(ic(i * h, j * h)); /*initilize cental line*/

	n_M = c_M; 

	int counter = 0;

	do
	{
		f_send_recv(c_M, d_M, u_M, n_loc, id, np);  /*send border lines*/
		
		for (int i = 1; i < m_len - 1; ++i) /*do even iterations*/
			for (int j = 1 + (i + sign) % 2; j < N - 1; j += 2)
				n_M[i * N + j] = ((c_M[(i + 1) * N + j] + c_M[(i - 1) * N + j] + c_M[i * N + j + 1] + c_M[i * N + j - 1]) \
					+ f((id * n_loc + i) * h, j * h) * H) * znam; 

		if (id != 0)
			for (int j = 1 + (sign % 2); j < N - 1; j += 2) /*calc highest line*/ 
				n_M[j] = (c_M[N + j] + u_M[j] + c_M[j + 1] + c_M[j - 1] + f(id * n_loc * h, j * h) * H) * znam;

		if (id != np - 1) /*calc lowest line*/ 
			for (int j = 1 + (n_loc - 1 + sign) % 2; j < N - 1; j += 2)
				n_M[(n_loc - 1) * N + j] = (d_M[j] + c_M[(n_loc - 2) * N + j] + c_M[(n_loc - 1) * N + j + 1] + c_M[(n_loc - 1) * N + j - 1] \
					+ f((id * n_loc + n_loc - 1) * h, j * h) * H) * znam;

		f_send_recv(n_M, d_M, u_M, n_loc, id, np); /*send border lines again*/
		

		for (int i = 1; i < m_len - 1; ++i) /*do odd iterations*/
			for (int j = 1 + (i + sign + 1) % 2; j < N - 1; j += 2)
				n_M[i * N + j] = ((n_M[(i + 1) * N + j] + n_M[(i - 1) * N + j] + n_M[i * N + j + 1] + n_M[i * N + j - 1]) \
					+ f((id * n_loc + i) * h, j * h) * H) * znam;

		if (id != 0)
			for (int j = 1 + (sign + 1) % 2; j < N - 1; j += 2) /*calc highest line*/ 
				n_M[j] = (n_M[N + j] + u_M[j] + n_M[j + 1] + n_M[j - 1] + f(id * n_loc * h, j * h) * H) * znam;

		if (id != np - 1) /*calc lowest line*/ 
			for (int j = 1 + (n_loc - 1 + sign + 1) % 2; j < N - 1; j += 2)
				n_M[(n_loc - 1) * N + j] = (d_M[j] + n_M[(n_loc - 2) * N + j] + n_M[(n_loc - 1) * N + j + 1] + n_M[(n_loc - 1) * N + j - 1] \
					+ f((id * n_loc + m_len - 1) * h, j * h) * H) * znam;

		counter++; 

		MAX_L = local_difference(c_M, n_M, m_len, N); /*calculate local diff*/
		MPI_Reduce(&MAX_L, &MAX_G, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /*calc max diff and send it to root process*/
		MPI_Bcast(&MAX_G, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); /*send this max value to all processes*/

		c_M.swap(n_M); /*swap old and new solutions*/

	} while (MAX_G > eps);

	double t2 = MPI_Wtime();


	double norm;
	norm  = analitical_difference(c_M, n_loc, m_len, id, np); /*calc diff between numerical and analitical solutions*/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&norm, &MAX_L, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /*calc max diff and send it to root process*/

	print_info(id, N, np,  MAX_G,  MAX_L,  counter,  t1,  t2);
}

void zeydel_SendRecv(const int id, const int np)
{
	int m_len, n_loc = N / np; /*lenght of middle blocks*/

	vector<int> len(np);

	if (id == 0)
	{
		int n = 0;
		for (int i = 0; i < np - 1; ++i)
			len[i] = N / np;
		len[np - 1] = N - len[0] * (np - 1); /*last process lenght*/
	}

	MPI_Scatter(len.data(), 1, MPI_INT, &m_len, 1, MPI_INT, 0, MPI_COMM_WORLD); /*send vector with sizes to all processes*/

	vector<double> c_M; /*current local vec*/
	vector<double> n_M; /*new local vec*/
	vector<double> u_M(N); /*upper line*/
	vector<double> d_M(N); /*lower line*/

	double MAX_L, MAX_G; /*errors*/

	double t1 = MPI_Wtime();

	int sign;
	(n_loc % 2 != 0) ?
		((id % 2 == 0) ? sign = 1 : sign = 0) : sign = 0; /*here we count if we do right or red iterations*/

	int ds = N - (N / np) * np; /*thats what we add to last line*/
	(id == (np - 1) ? ds = ds : ds = 0); /*ds = 0 for all exept last*/

	if (id == 0)
		for (int i = 0; i < n_loc + 1; ++i)
			for (int j = 0; j < N; ++j)
				c_M.push_back(ic(i * h, j * h)); /*initilize first line*/
	else
		if (id == (np - 1))
			for (int i = id * n_loc; i < ((np - 1) * n_loc) + m_len; ++i)
				for (int j = 0; j < N; ++j)
					c_M.push_back(ic(i * h, j * h)); /*initilize last line*/
		else
			for (int i = id * n_loc; i < (id + 1) * n_loc; ++i)
				for (int j = 0; j < N; ++j)
					c_M.push_back(ic(i * h, j * h)); /*initilize cental line*/

	n_M = c_M; 

	int counter = 0;

	do
	{
		f_sendrecv(c_M, d_M, u_M, n_loc, id, np);  /*send border lines*/
		
		for (int i = 1; i < m_len - 1; ++i) /*do even iterations*/
			for (int j = 1 + (i + sign) % 2; j < N - 1; j += 2)
				n_M[i * N + j] = ((c_M[(i + 1) * N + j] + c_M[(i - 1) * N + j] + c_M[i * N + j + 1] + c_M[i * N + j - 1]) \
					+ f((id * n_loc + i) * h, j * h) * H) * znam; 

		if (id != 0)
			for (int j = 1 + (sign % 2); j < N - 1; j += 2) /*calc highest line*/ 
				n_M[j] = (c_M[N + j] + u_M[j] + c_M[j + 1] + c_M[j - 1] + f(id * n_loc * h, j * h) * H) * znam;

		if (id != np - 1) /*calc lowest line*/ 
			for (int j = 1 + (n_loc - 1 + sign) % 2; j < N - 1; j += 2)
				n_M[(n_loc - 1) * N + j] = (d_M[j] + c_M[(n_loc - 2) * N + j] + c_M[(n_loc - 1) * N + j + 1] + c_M[(n_loc - 1) * N + j - 1] \
					+ f((id * n_loc + n_loc - 1) * h, j * h) * H) * znam;

		f_sendrecv(n_M, d_M, u_M, n_loc, id, np); /*send border lines again*/
		

		for (int i = 1; i < m_len - 1; ++i) /*do odd iterations*/
			for (int j = 1 + (i + sign + 1) % 2; j < N - 1; j += 2)
				n_M[i * N + j] = ((n_M[(i + 1) * N + j] + n_M[(i - 1) * N + j] + n_M[i * N + j + 1] + n_M[i * N + j - 1]) \
					+ f((id * n_loc + i) * h, j * h) * H) * znam;

		if (id != 0)
			for (int j = 1 + (sign + 1) % 2; j < N - 1; j += 2) /*calc highest line*/ 
				n_M[j] = (n_M[N + j] + u_M[j] + n_M[j + 1] + n_M[j - 1] + f(id * n_loc * h, j * h) * H) * znam;

		if (id != np - 1) /*calc lowest line*/ 
			for (int j = 1 + (n_loc - 1 + sign + 1) % 2; j < N - 1; j += 2)
				n_M[(n_loc - 1) * N + j] = (d_M[j] + n_M[(n_loc - 2) * N + j] + n_M[(n_loc - 1) * N + j + 1] + n_M[(n_loc - 1) * N + j - 1] \
					+ f((id * n_loc + m_len - 1) * h, j * h) * H) * znam;

		counter++; 

		MAX_L = local_difference(c_M, n_M, m_len, N); /*calculate local diff*/
		MPI_Reduce(&MAX_L, &MAX_G, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /*calc max diff and send it to root process*/
		MPI_Bcast(&MAX_G, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); /*send this max value to all processes*/

		c_M.swap(n_M); /*swap old and new solutions*/

	} while (MAX_G > eps);

	double t2 = MPI_Wtime();


	double norm;
	norm  = analitical_difference(c_M, n_loc, m_len, id, np); /*calc diff between numerical and analitical solutions*/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&norm, &MAX_L, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); /*calc max diff and send it to root process*/

	print_info(id, N, np,  MAX_G,  MAX_L,  counter,  t1,  t2);
}

int main(int argc, char** argv)
{
	int np, id;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if(type == 1){zeydel_Send_Recv(id, np);}
	if(type == 2){zeydel_SendRecv(id, np);}
	if(type == 3){zeydel_isendrecv(id, np);}
	
	

	MPI_Finalize();
}

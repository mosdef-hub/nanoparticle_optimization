#include "common.hpp"
#include "mpi.h"

int int_index(int t0, int t1){
	if(t0 == t1)
		return t0;
	return 0;
}

double get_mass(std::string i){
    if(i == "Si")
        return 28.0855;
    else
        return 15.9994;
}

void load_np(const char* filename, gbb& np){
    coordlist_t xyz;
    std::vector<std::string> types;
    std::map<std::string,int> type_map;
    type_map.insert(std::pair<std::string,int>("Si",1));
    type_map.insert(std::pair<std::string,int>("O",2));
    
    load_xyz(filename,xyz,types);
    
    for(int i = 0; i < xyz.size(); i++){
        np.insert_particle(xyz[i], get_mass(types[i]), type_map.find(types[i])->second, 0);
    }
}

bool calc_pe_compass(gbb np1, gbb np2, double *U, int pid, int n_threads)
{
	double U_local = 0;
	
	double sigma_SiSi = 4.405;
	double epsilon_SiSi = 0.198;
	
	double sigma_OO = 3.3;
	double epsilon_OO = 0.08;
    
    double sigma_SiO = 4.0323;
    double epsilon_SiO = 0.08993;
	
	coord_t epsilon;
	coord_t sigma;
	
	epsilon.push_back(epsilon_SiO);
	epsilon.push_back(epsilon_SiSi);
	epsilon.push_back(epsilon_OO);
	
	sigma.push_back(sigma_SiO);
	sigma.push_back(sigma_SiSi);
	sigma.push_back(sigma_OO);
    
	int i = 0;
	double pe_temp = 0;
    
    int breakdown = np1.n_coord / n_threads;
    int n_start = pid * breakdown;
    int n_final = n_start + breakdown;
    
    if(pid == n_threads - 1)
        n_final = np1.n_coord;
    
	for(i = n_start; i < n_final; i++){
		for(int j = 0; j < np2.n_coord; j++){
            
            double r = calc_distance(np1.v_coord[i], np2.v_coord[j]);
            
            double r3 = r*r*r;
			double r6 = r3*r3;
			double r9 = r3*r6;
			
			int pair_type = int_index(np1.v_type[i], np2.v_type[j]);
            
			double sigma_local;
            if(pair_type > 0)
                sigma_local = sigma[pair_type];
            else{
                sigma_local = sigma[0];
            }
            
            
            double sigma_overlap = sigma_local*0.8;
            if(r < sigma_overlap)
                return false;
            
			double sigma6 = sigma_local*sigma_local*sigma_local*sigma_local*sigma_local*sigma_local;
			double sigma9 = sigma6*sigma_local*sigma_local*sigma_local;
			
            double epsilon_local;
            
            if(pair_type > 0)
                epsilon_local = epsilon[pair_type];
            else
                epsilon_local = epsilon[0];
            
			U_local += epsilon_local * (2.0*(sigma9/r9) - 3.0*(sigma6/r6));
		}
	}
    
    *U = U_local;
    
	return true;
}

bool check_flatness(histogram_1D& visited, double threshold){
    double count = 0;
    double sum = 0;
    double average = 0;
    for(int i = 0; i < visited.size(); i++){
        sum += visited.array[i];
        count++;
    }
    
    average = sum / count;
    
    if(average != 0){
        bool is_flat = true;
        double avg_flat = 0.0;
        double min_flat = 1.0;
        double max_flat = 0.0;
        int count = 0;
        
        int gutter = 2;
        
        for(int i = gutter; i < visited.size()-gutter; i++)
        {
            double fraction = visited.array[i]/average;
            if(fraction < min_flat)
                min_flat = fraction;
            if(fraction > max_flat)
                max_flat = fraction;
            
            if(fraction < threshold)
                is_flat = false;
            
            count++;
        }
        std::cout << min_flat << "\t" << max_flat << std::endl;
        
        return is_flat;
    }
    return false;
}



int main(int argc, char *argv[]){

    MPI::Init (argc, argv);
    int n_threads, pid;
    
    n_threads = MPI::COMM_WORLD.Get_size();
    pid = MPI::COMM_WORLD.Get_rank();
    
    int np_size = atoi(argv[1]);
    stringstream filename;
    filename << np_size << "nm_nano_AA.xyz";
	
    gbb nanoparticle;
	load_np(filename.str().c_str(), nanoparticle);
    nanoparticle.shift_com();

    gbb np1;
    gbb np1_accept;
    gbb np2;
    gbb np2_accept;
    
    double maxx = 20.0*np_size + 40.0 + 10.0*(ceil(np_size/2.0) - 1.0);
    double minn = 20.0*np_size;
	double n_points = 100;
	double dr = (maxx-minn)/n_points;

    //this will contain the PE as a function of separation, block averaged
    block_average pe_block;
    pe_block.init(minn, maxx, dr);

    //this will hold the number of times we've visited a site
    //we'll use this to accept and reject moves.
    histogram_1D visited;
    visited.init(minn, maxx, dr);
    
    bool not_flat = true;
    
    double r_start = (maxx-minn)/2.0 + minn;

    //we'll start the particles out far away from each other
    np1 = nanoparticle;
    np1_accept = np1;
    
    np2 = nanoparticle;
    np2.translate_coord(r_start, 0.0, 0.0);
    np2_accept = np2;

    double dr_max = 10.0;
    double dtheta_max = 1.0;
    
    double r_current = r_start;
    double r_last = r_start;
    double theta_last1= 0;
    double theta_current1 = theta_last1;

    double theta_last2 = 0;
    double theta_current2 = theta_last2;
    
    stringstream trajname;
    trajname << np_size << "nm.lammpstrj";
    std::ofstream trajOut(trajname.str().c_str());

    box_info box;
    init_box(box, maxx*2.0);
    
    int count = 0;
    
    double *swap_local, *swap_global;
    int array_size = 2;
    swap_local = (double *)malloc( array_size * sizeof(double) );;
    swap_global = (double *)malloc( array_size * sizeof(double) );;
    
    
    while(not_flat)
    {
        r_current = r_last+dr_max*(2.0*drand48()-1.0);
        if(r_current >= minn && r_current< maxx)
        {
            theta_current1 = theta_last1+dtheta_max*(2.0*drand48()-1.0);
            theta_current2 = theta_last2+dtheta_max*(2.0*drand48()-1.0);
            
            np1 = nanoparticle;
            np1.rotate_coord(theta_current1,theta_current1,theta_current1);
            
            np2 = nanoparticle;
            np2.rotate_coord(theta_current2,theta_current2,theta_current2);
            np2.translate_coord(r_current, 0.0,0.0);
            
            double U;
            bool no_overlap = calc_pe_compass(np1,  np2, &U, pid, n_threads);
            
            swap_local[0] = U;
            if(no_overlap == true)
                swap_local[1] = 0;
            else
                swap_local[1] = 1;

            swap_global[0] = swap_global[1] = 0.0;
            MPI::COMM_WORLD.Allreduce(swap_local, swap_global, array_size, MPI_DOUBLE, MPI_SUM);
            
            U = swap_global[0];
            
            if(swap_global[1] == 0)
            //if(no_overlap == true)
            {
                double v_new = visited.get_hist(r_current);
                double v_old = visited.get_hist(r_last);
                double v_delta = v_old - v_new;
                double rand_value = drand48();

                if(exp(v_delta) > rand_value)
                {
                    visited.insert(r_current);
                    r_last = r_current;
                    theta_last1 = theta_current1;
                    theta_last2 = theta_current2;
                    np2_accept = np2;
                    np1_accept = np1;
                }
                else
                {
                    visited.insert(r_last);
                }
                pe_block.insert(r_current,U);

                count++;
            }
        }
        if(count%10 == 0)
        {
            bool is_flat = check_flatness(visited, 0.9);
            if(is_flat == false)
                std::cout << "not flat " << count << std::endl;
            else
                std::cout << "flat " << count<< std::endl;

            if(is_flat == true)
                not_flat = false;
            
            stringstream Uname;
            Uname << "U_" << np_size << "nm.txt";
            
            stringstream visname;
            visname << "visited_" << np_size << "nm.txt";
            
            std::ofstream dataOut1(Uname.str().c_str());
            std::ofstream dataOut2(visname.str().c_str());

            pe_block.print_mid(dataOut1);
            visited.print(dataOut2);
        }
        if(count%1000 == 0)
        {
            System np_system(box);
            np_system.insert_component(np1_accept);
            np_system.insert_component(np2_accept);
            np_system.parse_from_prototype();
            system_translator_LAMMPS_trajectory(np_system, trajOut, count);
        }
    }
    
	return 0;
}



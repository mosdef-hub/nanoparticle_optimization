#include "common.hpp"
#include "mpi.h"

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

bool calc_pe_compass(gbb np, double *U, int pid, int n_threads)
{
	double U_local = 0;
	
	double sigma_SiCH2 = 3.886;
	double epsilon_SiCH2 = 0.10947;
	
	double sigma_OCH2 = 3.445;
	double epsilon_OCH2 = 0.0855;
	
	coord_t epsilon;
	coord_t sigma;
	
	epsilon.push_back(epsilon_SiCH2);
	epsilon.push_back(epsilon_OCH2);
	
	sigma.push_back(sigma_SiCH2);
	sigma.push_back(sigma_OCH2);
    
    coord_t origin;
    origin.push_back(0.0);
    origin.push_back(0.0);
    origin.push_back(0.0);
    
	int i = 0;
	double pe_temp = 0;
    
    int breakdown = np.n_coord / n_threads;
    int n_start = pid * breakdown;
    int n_final = n_start + breakdown;
    
    if(pid == n_threads - 1)
        n_final = np.n_coord;
    
	for(i = n_start; i < n_final; i++){
        
        double r = calc_distance(np.v_coord[i], origin);
        double r2 = r*r;
        
        double r6 = r2*r2*r2;
        double r12 = r6*r6;
        
        int pair_type = np.v_type[i];
        
        double sigma_local = sigma[pair_type - 1];
        
        double sigma_overlap = sigma_local*0.8;
        if(r < sigma_overlap)
            return false;
        
        double sigma6 = sigma_local*sigma_local*sigma_local*sigma_local*sigma_local*sigma_local;
        double sigma12 = sigma6*sigma6;
        
        double epsilon_local = epsilon[pair_type - 1];
        
        U_local += 4.0 * epsilon_local * ((sigma12/r12) - (sigma6/r6));
	}
    
    *U = U_local;
    
	return true;
}

bool check_flatness(histogram_1D& visited, double threshold, int min_configs){
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
        
        int gutter = 10;
        
        for(int i = gutter; i < visited.size()-2; i++)
        {
            double fraction = visited.array[i]/average;
            if(fraction < min_flat)
                min_flat = fraction;
            if(fraction > max_flat)
                max_flat = fraction;
            
            if(fraction < threshold || visited.array[i] < min_configs)
                is_flat = false;
            
            count++;
        }
        //std::cout << min_flat << "\t" << max_flat << std::endl;
        
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

    gbb np;
    gbb np_accept;
    
    coord_t origin;
    origin.push_back(0.0);
    origin.push_back(0.0);
    origin.push_back(0.0);
    
    double maxx = 10.0*np_size + 30.0;
    double minn = 10.0*np_size + 10.0;
	double n_points = 20;
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
    np = nanoparticle;
    
    if(pid == 0){
        np.calc_com();
        std::cout << "Separation-init: " << np.v_com[0] << std::endl;
    }
    
    np.translate_coord(r_start, 0.0, 0.0);
    
    if(pid == 0){
        np.calc_com();
        std::cout << "Separation-first translation: " << np.v_com[0] << std::endl;
    }
    
    np_accept = np;
    
    double dr_max = 2.0;
    double dtheta_max = 1.0;
    
    double r_current = r_start;
    double r_last = r_start;
    double theta_last = 0;
    double theta_current = theta_last;
    
    stringstream trajname;
    trajname << np_size << "nm_CH2.lammpstrj";
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
            theta_current = theta_last+dtheta_max*(2.0*drand48()-1.0);
            
            np = nanoparticle;
            np.rotate_coord(theta_current,theta_current,theta_current);
            np.translate_coord(r_current, 0.0,0.0);
            
            double U;
            bool no_overlap = calc_pe_compass(np, &U, pid, n_threads);
            
            if(pid == 0){
                np.calc_com();
                std::cout << "Separation: " << np.v_com[0] << std::endl;
                std::cout << "r_current: " << r_current << std::endl;
            }
            
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
                
                /*
                visited.insert(r_current);
                r_last = r_current;
                theta_last = theta_current;
                np_accept = np;
                 */
                
                if(exp(v_delta) > rand_value)
                {
                    visited.insert(r_current);
                    r_last = r_current;
                    theta_last = theta_current;
                    np_accept = np;
                }
                else
                {
                    visited.insert(r_last);
                }
                pe_block.insert(r_current,U);

                count++;
            }
        }
        else{
            r_current = r_last;
        }
        if(count%10 == 0)
        {
            bool is_flat = check_flatness(visited, 0.9, 500);
            /*
            if(is_flat == false)
                std::cout << "not flat " << count << std::endl;
            else
                std::cout << "flat " << count<< std::endl;
             */

            if(is_flat == true)
                not_flat = false;
            
            stringstream Uname;
            Uname << "U_" << np_size << "nm_CH2_well.txt";
            
            stringstream visname;
            visname << "visited_" << np_size << "nm_CH2_well.txt";
            
            std::ofstream dataOut1(Uname.str().c_str());
            std::ofstream dataOut2(visname.str().c_str());

            pe_block.print_mid(dataOut1);
            visited.print(dataOut2);
        }
        if(count%500 == 0)
        {
            System np_system(box);
            np_system.insert_component(np_accept);
            
            gbb cross_particle;
            cross_particle.insert_particle(origin, 14.0268, 3, 0);
            
            np_system.insert_component(cross_particle);
            np_system.parse_from_prototype();
            system_translator_LAMMPS_trajectory(np_system, trajOut, count);
        }
    }
    
	return 0;
}



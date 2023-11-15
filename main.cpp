#include <iostream>
#include <mpi.h>
#include <neso_particles.hpp>
#include <chrono>
#include <csignal>
#include <string>

using namespace cl;
using namespace NESO::Particles;

inline void hybrid_move_driver(const int N_total, 
  const int Nsteps_warmup = 1024,
  const int Nsteps = 2048
){

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 16;
  dims[1] = 16;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;
  
  const int global_cell_count = dims[0] * dims[1] * std::pow(std::pow(2, subdivision_order), ndim);
  const int npart_per_cell = std::round((double) N_total / (double) global_cell_count);

  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order, stencil_width);

  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  A->add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);
  

  const REAL dt = 0.001;
  const int cell_count = domain->mesh->get_cell_count();
  const int N = npart_per_cell * cell_count;
  const int N_total_actual = npart_per_cell * global_cell_count;

  if (rank == 0){
    nprint("Stencil width:", stencil_width);
    nprint("Num Particles requested:", N_total);
    nprint("Num Particles actual:", N_total_actual);
    nprint("Num Warm-up Steps:", Nsteps_warmup);
    nprint("Num Steps:", Nsteps);
  }
  
  if (N > 0){
  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells, rng_pos);

  auto velocities = NESO::Particles::normal_distribution(
      N, ndim, 0.0, 0.5, rng_vel);
  std::uniform_int_distribution<int> uniform_dist(
      0, size - 1);
  ParticleSet initial_distribution(N, A->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  A->add_particles_local(initial_distribution);
  }
  //parallel_advection_initialisation(A, 64);

  if (rank == 0){
    std::cout << "Particles Added..." << std::endl;
  }

  auto pbc = std::make_shared<CartesianPeriodic>(sycl_target, mesh, A->position_dat);
  auto ccb = std::make_shared<CartesianCellBin>(sycl_target, mesh, A->position_dat, A->cell_id_dat);


  ParticleLoop advect_new(
    "Avection",
    A,
    [=](auto V, auto P){
      for(int dx=0 ; dx<ndim ; dx++){
        P[dx] += dt * V[dx];
      }
    },
    Access::read(Sym<REAL>("V")),
    Access::write(Sym<REAL>("P"))
  );


  REAL T = 0.0;
 
  pbc->execute();
  A->hybrid_move();
  ccb->execute();
  A->cell_move();

  MPI_Barrier(sycl_target->comm_pair.comm_parent);
  if (rank == 0){
    std::cout << N_total_actual << " Particles Distributed..." << std::endl;
  }

  //H5Part h5part("traj.h5part", A, Sym<REAL>("P"), Sym<INT>("CELL_ID"));

  for (int stepx = 0; stepx < Nsteps_warmup; stepx++) {

    pbc->execute();

    A->hybrid_move();

    ccb->execute();
    A->cell_move();
    //h5part.write();
    
    advect_new.execute();

    T += dt;
    
    if( (stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }
  }
  //h5part.close();
  sycl_target->profile_map.reset();

  std::chrono::high_resolution_clock::time_point time_start = std::chrono::high_resolution_clock::now();

  for (int stepx = 0; stepx < Nsteps; stepx++) {


    pbc->execute();

    A->hybrid_move();

    ccb->execute();
    A->cell_move();
    advect_new.execute();


    T += dt;   
    if( (stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }
  }

  std::chrono::high_resolution_clock::time_point time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_taken = time_end - time_start;
  const double time_taken_double = (double) time_taken.count();
  
  if (rank == 0){
    std::cout << "TIME TAKEN: " << time_taken_double << " PER STEP: " << time_taken_double / Nsteps << std::endl;
  }


  mesh->free();
  
  if (rank == 0){
    sycl_target->profile_map.print();
  }

}

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }
 
  if (argc > 3){

    std::vector<int> argsi(3);
    for(int ix=0 ; ix<3 ; ix++){
      std::string argv0 = std::string(argv[ix+1]);
      argsi.at(ix) = std::stoi(argv0);
    }
    hybrid_move_driver(argsi.at(0), argsi.at(1), argsi.at(2));
  }


  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}

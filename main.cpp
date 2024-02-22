#include <iostream>
#include <mpi.h>
#include <neso_particles.hpp>
#include <chrono>
#include <cmath>
#include <string>

using namespace cl;
using namespace NESO::Particles;

inline void hybrid_move_driver(
  const bool single_cell_mode,
  const int N_total, 
  const int Ncells = 16,
  const int Nsteps_warmup = 1024,
  const int Nsteps = 2048
){
 
  // Extent of each coarse cell in each dimension.
  const double cell_extent = 1.0;
  // Number of times to subdivide each coarse cell to create the mesh.
  const int subdivision_order = 0;
  const REAL fine_cell_extent = cell_extent / std::pow(2.0, subdivision_order);
  // thermal velocity scaling
  const REAL v_sigma = 0.5;
  // Time step size.
  const REAL dt = 0.25 * fine_cell_extent / v_sigma;
  // Number of spatial dimensions.
  const int ndim = 2;
  std::vector<int> dims(ndim);
  // Number of coarse cells in the mesh in each dimension.
  dims[0] = Ncells;
  dims[1] = Ncells;

  // Halo width for local move.
  const int stencil_width = 2;
  // Create the mesh.
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order, stencil_width);
  mesh->single_cell_mode = single_cell_mode;
  nprint("Num Owned Cells:", mesh->get_owned_cells().size());

  // Create a container that wraps a sycl queue and a MPI communicator.
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());
  // Object to map particle positions into cells.
  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  // Create a domain from a mesh and a rule to map local cells to owning mpi ranks.
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  // The specification of the particle properties.
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};
  
  // Create a ParticleGroup from a domain, particle specification and a compute
  // target.
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  // Spawn particles cells wise to be close to the requested number of
  // particles globally.
  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);
  const int cart_cell_count = mesh->get_cart_cell_count();
  const int global_cell_count = dims[0] * dims[1] * std::pow(std::pow(2, subdivision_order), ndim);
  const int npart_per_cell = std::max((INT)std::round((double) N_total / (double) global_cell_count), (INT)1);
  const int N = npart_per_cell * cart_cell_count;
  const int N_total_actual = npart_per_cell * global_cell_count;

  if (rank == 0){
    sycl_target->print_device_info();
    nprint("dt:", dt);
    nprint("NP   Cell Count:", mesh->get_cell_count());
    nprint("Mesh Cell Count:", cart_cell_count);
    nprint("Global Cell Count:", global_cell_count);
    nprint("Single Cell Mode:", single_cell_mode);
    nprint("Stencil width:", stencil_width);
    nprint("Num MPI ranks:", sycl_target->comm_pair.size_parent);
    nprint("Num Particles requested:", N_total);
    nprint("Num Particles actual:", N_total_actual);
    nprint("Num Warm-up Steps:", Nsteps_warmup);
    nprint("Num Steps:", Nsteps);
    nprint("Num cells per dimension:", Ncells);
  }
  
  if (N > 0){
    std::vector<std::vector<double>> positions;
    std::vector<int> cells;
    // Sample particles randomly in each local cell.
    uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells, rng_pos);
    // Sample some particle velocities.
    auto velocities = NESO::Particles::normal_distribution(
        N, ndim, 0.0, v_sigma, rng_vel);
    std::uniform_int_distribution<int> uniform_dist(
        0, size - 1);
    // Host space to store the created particles.
    ParticleSet initial_distribution(N, A->get_particle_spec());
    // Populate the host space with particle data.
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        initial_distribution[Sym<REAL>("P")][px][dimx] = positions.at(dimx).at(px);
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities.at(dimx).at(px);
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    // Add the particles to the ParticleGroup
    A->add_particles_local(initial_distribution);
  }

  if (rank == 0){
    std::cout << "Particles Added..." << std::endl;
  }
  
  // Create object to apply periodic boundary conditions.
  auto pbc = std::make_shared<CartesianPeriodic>(sycl_target, mesh, A->position_dat);
  // Create object to map particle positions to mesh cells.
  auto ccb = std::make_shared<CartesianCellBin>(sycl_target, mesh, A->position_dat, A->cell_id_dat);

  std::vector<REAL> extents = {dims[0] * cell_extent, dims[1] * cell_extent};
  auto la_extents = std::make_shared<LocalArray<REAL>>(sycl_target, extents);

  auto advect_pbc_loop = particle_loop(
    "AvectionPBC",
    A,
    [=](auto V, auto P, auto EXTENTS){
      for(int dx=0 ; dx<ndim ; dx++){
        const REAL p_old = P.at(dx);
        const REAL p_new = p_old + dt * V.at(dx);
        const REAL tmp_extent = EXTENTS.at(dx);
        const int n_extent_offset_int = abs((int)p_new);
        const REAL n_extent_offset_real = n_extent_offset_int + 2;
        const REAL p_pbc = 
          fmod(p_new + n_extent_offset_real * tmp_extent, tmp_extent);
        P.at(dx) = p_pbc;
      }
    },
    Access::read(Sym<REAL>("V")),
    Access::write(Sym<REAL>("P")),
    Access::read(la_extents)
  );


  // This creates a ParticleLoop to apply the advection.
  auto advect_loop = particle_loop(
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

  MPI_Barrier(sycl_target->comm_pair.comm_parent);
  if (rank == 0){
    std::cout << N_total_actual << " Particles Distributed..." << std::endl;
  }
  
  // Uncomment to write a trajectory.
  //H5Part h5part("traj.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"), Sym<INT>("CELL_ID"), Sym<INT>("ID"));

  REAL T = 0.0;
  for (int stepx = 0; stepx < Nsteps_warmup; stepx++) {
    
    // Apply periodic boundary conditions.
    // pbc->execute();
    
    // Move particles between MPI ranks.
    A->hybrid_move();
    
    if (!single_cell_mode){
      // Bin particles into cells (determine the owning cell).
      ccb->execute();

      // Move particles into owning cells.
      A->cell_move();
    }

    // Uncomment to write a trajectory.
    //h5part.write();
    
    // Execute the advection particle loop.
    // advect_loop->execute();
    advect_pbc_loop->execute();

    T += dt;
    
    if( (stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }
  }

  // Uncomment to write a trajectory.
  //h5part.close();
  MPI_Barrier(sycl_target->comm_pair.comm_parent);
  sycl_target->profile_map.reset();

  std::chrono::high_resolution_clock::time_point time_start = std::chrono::high_resolution_clock::now();
  
  auto comm = sycl_target->comm_pair.comm_parent;

  auto lambda_do_run = [&](){
    for (int stepx = 0; stepx < Nsteps; stepx++) {
      //ProfileRegion r0("main", "pbc");
      //pbc->execute();
      //r0.end();
      //sycl_target->profile_map.add_region(r0);

      MPI_Barrier(comm);
      ProfileRegion r1("main", "hybrid_move");
      A->hybrid_move();
      r1.end();
      sycl_target->profile_map.add_region(r1);

      if (!single_cell_mode){
        // Bin particles into cells (determine the owning cell).
        ccb->execute();
        // Move particles into owning cells.
        A->cell_move();
      }

      //ProfileRegion r2("main", "advect_loop");
      //advect_loop->execute();
      //r2.end();
      //sycl_target->profile_map.add_region(r2);
      
      ProfileRegion r2("main", "advect_pbc_loop");
      advect_pbc_loop->execute();
      r2.end();
      sycl_target->profile_map.add_region(r2);

      T += dt;   
      if( (stepx % 100 == 0) && (rank == 0)) {
        std::cout << stepx << std::endl;
      }
    }
  };

  lambda_do_run();

  std::chrono::high_resolution_clock::time_point time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_taken = time_end - time_start;
  const double time_taken_double = (double) time_taken.count();
  
  if (rank == 0){
    std::cout << "TIME TAKEN: " << time_taken_double << " PER STEP: " << time_taken_double / Nsteps << std::endl;
  }

  A->free();
  mesh->free();
  
  if (rank == 0){
    sycl_target->profile_map.print();
  }

  sycl_target->profile_map.write_events_json("hybrid_move_events", rank);
}

int main(int argc, char **argv) {
  
  int provided_thread_level;
#ifndef NESO_PARTICLES_THREAD_LEVEL
#define THREAD_LEVEL MPI_THREAD_FUNNELED
#else
#define THREADS_LEVEL NESO_PARTICLES_THREAD_LEVEL
#endif

  if (MPI_Init_thread(&argc, &argv, THREAD_LEVEL, &provided_thread_level) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }
  if (provided_thread_level < THREAD_LEVEL) {
    std::cout << "ERROR: provided thread level too low." << std::endl;
    return -1;
  }
  
  const int nargs = 5;
  if (argc > nargs){
    std::vector<int> argsi(nargs);
    for(int ix=0 ; ix<nargs ; ix++){
      std::string argv0 = std::string(argv[ix+1]);
      argsi.at(ix) = std::stoi(argv0);
    }
    hybrid_move_driver((bool) argsi.at(0), argsi.at(1), argsi.at(2), argsi.at(3), argsi.at(4));
  } else {
    nprint("Insufficient number of arguments. Please pass: <mode> <number of particles> <number of cells> <number of warmup steps> <number of timed steps>.");
  }


  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}

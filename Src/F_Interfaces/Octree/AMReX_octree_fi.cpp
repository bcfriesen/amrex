
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_ParmParse.H>
#include <AMReX_AmrCore.H>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

namespace
{
    struct treenode {
        int level, grid;
    };
}

extern "C" {

    void amrex_fi_init_octree ()
    {
        ParmParse pp("amr");
        int cnt = pp.countval("max_grid_size");
        int max_grid_size;
        if (cnt == 0) {
            max_grid_size = 8;
            pp.add("max_grid_size", max_grid_size);
        } else if (cnt == 1) {
            pp.get("max_grid_size", max_grid_size);
        } else {
            amrex::Abort("amrex_fi_init_octree: must use the same max_grid_size for all levels");
        }

        int blocking_factor = 2*max_grid_size;
        pp.add("blocking_factor", blocking_factor);

        pp.add("grid_eff", 1.0);

        int max_level;
        pp.get("max_level", max_level);

        Array<int> ref_ratio(max_level, 2);
        pp.addarr("ref_ratio", ref_ratio);
    }

    void amrex_fi_build_octree_leaves (AmrCore* const amrcore, int* n, Array<treenode>*& leaves)
    {
        leaves = new Array<treenode>;
        const int finest_level = amrcore->finestLevel();
        const int myproc = ParallelDescriptor::MyProc();
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            const BoxArray& ba = amrcore->boxArray(lev);
            const DistributionMapping& dm = amrcore->DistributionMap(lev);
            const int ngrids = ba.size();
            BL_ASSERT(ba.size() < std::numeric_limits<int>::max());
            if (lev == finest_level)
            {
                for (int i = 0; i < ngrids; ++i) {
                    if (dm[i] == myproc) {
                        leaves->push_back({lev, i});
                    }
                }
            }
            else
            {
                const BoxArray& fba = amrcore->boxArray(lev+1);
                const IntVect& rr = amrcore->refRatio(lev);
                for (int i = 0; i < ngrids; ++i) {
                    if (dm[i] == myproc)
                    {  
                        Box bx = ba[i];
                        bx.refine(rr);
                        if (!fba.intersects(bx)) {
                            leaves->push_back({lev, i});
                        }
                    }
                }
            }
        }
        *n = leaves->size();
    }

    void amrex_fi_copy_octree_leaves (Array<treenode>* leaves, treenode a_copy[])
    {
        const int n = leaves->size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; ++i) {
            a_copy[i] = (*leaves)[i];
        }
        
        delete leaves;
    }
}


#include <AMReX_FluxRegister.H>

using namespace amrex;

extern "C"
{
    void amrex_fi_new_fluxregister (FluxRegister*& flux_reg, const BoxArray* ba, 
                                    const DistributionMapping* dm, int rr, int flev, int ncomp)
    {
        flux_reg = new FluxRegister(*ba, *dm, IntVect(D_DECL(rr,rr,rr)), flev, ncomp);
    }

    void amrex_fi_delete_fluxregister (FluxRegister* flux_reg)
    {
        delete flux_reg;
    }

    void amrex_fi_fluxregister_fineadd (FluxRegister* flux_reg, MultiFab* flxs[], Real scale)
    {
        for (int dir = 0; dir < BL_SPACEDIM; ++dir) {
            BL_ASSERT(flux_reg->nComp() == flxs[dir]->nComp());
            flux_reg->FineAdd(*flxs[dir], dir, 0, 0, flux_reg->nComp(), scale);
        }
    }

    void amrex_fi_fluxregister_crseinit (FluxRegister* flux_reg, MultiFab* flxs[], Real scale)
    {
        for (int dir = 0; dir < BL_SPACEDIM; ++dir) {
            BL_ASSERT(flux_reg->nComp() == flxs[dir]->nComp());
            flux_reg->CrseInit(*flxs[dir], dir, 0, 0, flux_reg->nComp(), scale);
        }
    }

    void amrex_fi_fluxregister_crseadd (FluxRegister* flux_reg, MultiFab* flxs[], Real scale,
                                        const Geometry* geom)
    {
        for (int dir = 0; dir < BL_SPACEDIM; ++dir) {
            BL_ASSERT(flux_reg->nComp() == flxs[dir]->nComp());
            flux_reg->CrseAdd(*flxs[dir], dir, 0, 0, flux_reg->nComp(), scale, *geom);
        }
    }

    void amrex_fi_fluxregister_setval (FluxRegister* flux_reg, Real val)
    {
        flux_reg->setVal(val);
    }

    void amrex_fi_fluxregister_reflux (FluxRegister* flux_reg, MultiFab* mf, Real scale, 
                                       const Geometry* geom)
    {
        MultiFab vol;
        geom->GetVolume(vol, mf->boxArray(), mf->DistributionMap(), 0);
        BL_ASSERT(flux_reg->nComp() == mf->nComp());
        flux_reg->Reflux(*mf, vol, scale, 0, 0, flux_reg->nComp(), *geom);
    }
}

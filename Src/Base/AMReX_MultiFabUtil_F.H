#ifndef _MtltiFabUtil_F_H_
#define _MtltiFabUtil_F_H_

#include <AMReX_BLFort.H>

extern "C"
{
    void amrex_fort_avg_nd_to_cc(const int* lo, const int* hi, const int* ncomp,
                                 BL_FORT_FAB_ARG(cc), const BL_FORT_FAB_ARG(nd));
}

BL_FORT_PROC_DECL(BL_AVG_EG_TO_CC, bl_avg_eg_to_cc)
  (const int* lo, const int* hi,
   BL_FORT_FAB_ARG(cc),
   D_DECL(const BL_FORT_FAB_ARG(Ex),
	  const BL_FORT_FAB_ARG(Ey),
	  const BL_FORT_FAB_ARG(Ez)));

BL_FORT_PROC_DECL(BL_AVG_FC_TO_CC, bl_avg_fc_to_cc)
  (const int* lo, const int* hi,
   BL_FORT_FAB_ARG(cc),
   D_DECL(const BL_FORT_FAB_ARG(fx),
	  const BL_FORT_FAB_ARG(fy),
	  const BL_FORT_FAB_ARG(fz)),
   const amrex::Real* dx, const amrex::Real* problo, const int& coord_type);

BL_FORT_PROC_DECL(BL_AVG_CC_TO_FC, bl_avg_cc_to_fc)
  (const int* xlo, const int* xhi,
#if (BL_SPACEDIM > 1)
   const int* ylo, const int* yhi,
#endif
#if (BL_SPACEDIM == 3)
   const int* zlo, const int* zhi,
#endif
   D_DECL(BL_FORT_FAB_ARG(fx),
	  BL_FORT_FAB_ARG(fy),
	  BL_FORT_FAB_ARG(fz)),
   const BL_FORT_FAB_ARG(cc),
   const amrex::Real* dx, const amrex::Real* problo, const int& coord_type);

BL_FORT_PROC_DECL(BL_AVGDOWN_FACES, bl_avgdown_faces)
  (const int* lo, const int* hi,
   const BL_FORT_FAB_ARG(fine),
   BL_FORT_FAB_ARG(crse),
   const int* ratio, const int& n, const int& ncomp); 

BL_FORT_PROC_DECL(BL_AVGDOWN, bl_avgdown)
  (const int* ovlp_lo, const int* ovlp_hi,
   const BL_FORT_FAB_ARG(fine),
   BL_FORT_FAB_ARG(crse),
   const int* ratio, const int* ncomp);

BL_FORT_PROC_DECL(BL_AVGDOWN_WITH_VOL, bl_avgdown_with_vol)
  (const int* ovlp_lo, const int* ovlp_hi,
   const BL_FORT_FAB_ARG(fine),
   BL_FORT_FAB_ARG(crse),
   const BL_FORT_FAB_ARG(fvolume),
   const int* ratio, const int* ncomp);

#endif

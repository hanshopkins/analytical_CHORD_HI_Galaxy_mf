//g++ -fPIC -o amf.so -shared analytical_mf.cpp -fopenmp

#include <iostream>
#include <math.h>       /* sin, cos, fmod, fabs, asin */
#include <cmath>        /* cyl_bessel_j, acos */
#include <omp.h>
#include<limits>
#include <stdio.h>

#define D 6.0 //meters
#define L 7.0 //meters
#define PI 3.14159265358979323846
#define omega 2*PI/86400 //earth angular velocity in rads/second

inline void ang2vec (const double theta, const double phi, double outvec [3])
{
    outvec[0] = sin(theta)*cos(phi);
    outvec[1] = sin(theta)*sin(phi);
    outvec[2] = cos(theta);
}

inline void rotate (const double v [3], double outvec [3], const double alpha)
{
    outvec[0] = cos(alpha)*v[0] - sin(alpha)*v[1];
    outvec[1] = sin(alpha)*v[0] + cos(alpha)*v[1];
    outvec[2] = v[2];
}

inline double dot (const double v1 [3], const double v2 [3])
{
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

inline void cross (const double v1 [3], const double v2 [3], double outvec [3])
{
    outvec[0] = v1[1]*v2[2]-v1[2]*v2[1];
    outvec[1] = v1[2]*v2[0]-v1[0]*v2[2];
    outvec[2] = v1[0]*v2[1]-v1[1]*v2[0]; 
}

inline double crossmag(const double v1 [3], const double v2 [3])
{
    double cv [3];
    cross(v1,v2,cv);
    return sqrt(dot(cv,cv));
}

inline double subtractdot (const double v1_a [3], const double v1_b [3], const double v2 [3])
{
    return (v1_a[0]-v1_b[0])*v2[0]+(v1_a[1]-v1_b[1])*v2[1]+(v1_a[2]-v1_b[2])*v2[2];
}

inline double source_phi (const double source_phi_0, const double t)
{
    return source_phi_0 + omega * t;
}

double B_sq (const double alpha, const double wavelength)
{
    double alphaprime = PI*D*alpha/wavelength;
    if (alphaprime <= 1E-8 && alphaprime >= -1E-8)
        return (std::cyl_bessel_j(0,alphaprime)-std::cyl_bessel_j(2,alphaprime))*(std::cyl_bessel_j(0,alphaprime)-std::cyl_bessel_j(2,alphaprime)); //l'Hopital's
    else
        return (2*std::cyl_bessel_j(1,alphaprime)/alphaprime) * (2*std::cyl_bessel_j(1,alphaprime)/alphaprime);
}

inline double Bsq_from_vecs (const double v1 [3], const double v2 [3], const double wavelength)
{
    double dp = dot(v1,v2);
    //need some clamping logic for the acos
    //if (dp >= 1) return 1;
    //if (dp <= -1) return (2*std::cyl_bessel_j(1,PI*PI*D/wavelength)/(PI*PI*D/wavelength)) * (2*std::cyl_bessel_j(1,PI*PI*D/wavelength)/(PI*PI*D/wavelength));
    //we want to deal with the arccos instiblity by using the cross product formula instead
    double delta_ang;
    if (dp < 0.99 && dp > -0.99) delta_ang = std::acos(dp);    
    else 
    {
        delta_ang = std::asin(crossmag(v1,v2));
        delta_ang = (dp > 0) ? delta_ang : PI-delta_ang;
    }
    //printf("dp: %f, crossmag %f, Delta ang: %f, B_sq %f\n", dp, crossmag(v1,v2), delta_ang, B_sq(delta_ang, wavelength));
    return B_sq(delta_ang, wavelength);
}

double sin_sq_ratio (const unsigned short m, const double x_prime)
{
    double x = fmod(x_prime,PI); // -pi < x < pi
    x = fabs(x); // 0 < x < pi
    x = (x > PI/2) ? PI-x : x; //0 < x < pi/2
    
    //if (x < 1E-9) return m*m +1.0/3 * (m*m-m*m*m*m) * x*x;
    if (fabs(x) < 1E-9) return m*m*cos(m*x)*cos(m*x)/(cos(x)*cos(x)); //asin(std::numeric_limits<double>::epsilon())
    else return sin(m*x)*sin(m*x)/(sin(x)*sin(x));
}

extern "C" {void analytic_matched_filter (const double chord_theta, const double wavelength, const double source_theta, const double source_phi_0, const unsigned short m, 
                                    const double * u, const unsigned int num_u, const double delta_tau, const unsigned int time_samples, double * mf)
{
    double chord_pointing [3];
    double dir1_proj_vec [3]; //north/south chord direction
    double dir2_proj_vec [3]; //east/west chord direction
    ang2vec(chord_theta, 0, chord_pointing);
    ang2vec(chord_theta + PI/2, 0, dir1_proj_vec);
    cross(dir1_proj_vec, chord_pointing, dir2_proj_vec);
    
    //calculating the normalization factor
    double normalization_sum=0;
    for (unsigned int j = 0; j < time_samples; j++)
    {
        double tau = j*delta_tau;
        double source_pointing [3];
        ang2vec(source_theta, source_phi(source_phi_0, tau), source_pointing);
        double Bsq_source = Bsq_from_vecs(source_pointing, chord_pointing, wavelength);
        ///////////
        normalization_sum += Bsq_source*Bsq_source;
    }
    double normalization = sqrt(normalization_sum)*m*m*m*m;
    //calculating rest of matched filter
    #pragma omp parallel for
    for (unsigned int i = 0; i < num_u; i++)
    {
        double numerator_sum = 0;
        double denominator_sum = 0;
        const double * u_i = u + 3*i;
        for (unsigned int j = 0; j < time_samples; j++)
        {
            double tau = j*delta_tau;
            double source_pointing [3];
            ang2vec(source_theta, source_phi(source_phi_0, tau), source_pointing);
            
            double u_rot [3];
            rotate(u_i, u_rot, tau*omega);
            
            double Bsq_u = Bsq_from_vecs(u_rot, chord_pointing, wavelength);
            //if (i == 0 && j == 0) std::cout << Bsq_u << std::endl;
            
            //computing numerator
            double cdir1 = PI*L/wavelength*subtractdot(source_pointing, u_rot, dir1_proj_vec);
            double cdir2 = PI*L/wavelength*subtractdot(source_pointing, u_rot, dir2_proj_vec);
            
            double Bsq_source = Bsq_from_vecs(source_pointing, chord_pointing, wavelength);
            ///////////
            numerator_sum += Bsq_source * Bsq_u * sin_sq_ratio(m,cdir1) * sin_sq_ratio(m,cdir2);
            denominator_sum += Bsq_u*Bsq_u;
        }
        //if (i%3000 == 0) std::cout << "i: " << i << " num sum: " << numerator_sum << " Denom sum:" << denominator_sum << std::endl;
        mf[i] = numerator_sum/sqrt(denominator_sum)/normalization;
    }
}}

extern "C" {double analytic_matched_filter_single_u (const double * chord_theta, const double wavelength, const double source_theta, const double source_phi_0, const unsigned short m, 
                                    const double * u, const double delta_tau, const unsigned int time_samples, const unsigned int ndithers)
{
    double numerator_sum = 0;
    double denominator_sum = 0;
    double normalization_sum = 0;
    for (unsigned int k = 0; k<ndithers; k++)
    {
        double chord_pointing [3];
        double dir1_proj_vec [3]; //north/south chord direction
        double dir2_proj_vec [3]; //east/west chord direction
        ang2vec(chord_theta[k], 0, chord_pointing);
        ang2vec(chord_theta[k] + PI/2, 0, dir1_proj_vec);
        cross(dir1_proj_vec, chord_pointing, dir2_proj_vec);
        
        #pragma omp parallel for reduction(+:numerator_sum,denominator_sum, normalization_sum)
        for (unsigned int j = 0; j < time_samples; j++)
        {
            double tau = j*delta_tau;
            double source_pointing [3];
            ang2vec(source_theta, source_phi(source_phi_0, tau), source_pointing);
            
            double u_rot [3];
            rotate(u, u_rot, tau*omega);
            
            double Bsq_u = Bsq_from_vecs(u_rot, chord_pointing, wavelength);
            
            //computing numerator
            double cdir1 = PI*L/wavelength*subtractdot(source_pointing, u_rot, dir1_proj_vec);
            double cdir2 = PI*L/wavelength*subtractdot(source_pointing, u_rot, dir2_proj_vec);
            
            double Bsq_source = Bsq_from_vecs(source_pointing, chord_pointing, wavelength);
            ///////////
            //printf("numerator: %f, Denom: %f, normalization: %f\n", Bsq_source * Bsq_u * sin_sq_ratio(m,cdir1) * sin_sq_ratio(m,cdir2), Bsq_u*Bsq_u,Bsq_source*Bsq_source);
            numerator_sum += Bsq_source * Bsq_u * sin_sq_ratio(m,cdir1) * sin_sq_ratio(m,cdir2);
            denominator_sum += Bsq_u*Bsq_u;
            normalization_sum += Bsq_source*Bsq_source;
        }
    }
    double normalization = sqrt(normalization_sum)*m*m*m*m;
    //std::cout << "num: " << numerator_sum << " denom: " << sqrt(denominator_sum) << " Normalization: " << normalization << std::endl;
    return numerator_sum/sqrt(denominator_sum)/normalization;
}}

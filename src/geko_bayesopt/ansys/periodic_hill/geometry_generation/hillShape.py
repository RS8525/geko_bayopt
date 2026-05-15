# Modified script, original version "https://github.com/xiaoh/para-database-for-PIML/blob/master/utility/hill-geometry-gereration/hillShape.py"
# This script generates the bottom boundary shape of the periodic hill geometry 
# based on the provided polynomial equations given in "Flows over periodic hills of parameterized geometries: A dataset for
# data-driven turbulence modeling from direct simulations" by Xiao et al.
# The hill length is scaled according to the provided alphaparameter, 
# the hill height is normalized to 1 and the flat portion is determined by the total length.

import numpy as np


# Generates baseline (e.g. non-stretched or in other words alpha = 1) hill geometry
def profile(yy):
    'Calculate the shape of the periodic hill'

    import numpy as np

    y = np.array(yy)
    x = y * 28

    h = np.zeros(len(x))
    for i in range(len(x)):
        
        # hill is periodic
        if x[i] > 126.0 :
            x[i] = 252.0 - x[i]
        
        ## "normal" hill generation
        
        if (x[i]>=0) and (x[i]<9) :
            h[i] = np.minimum(28., 2.8e+01 + 0.0e+00*x[i] + \
                              6.775070969851e-03*x[i]**2 - 2.124527775800e-03*x[i]**3);
        elif (x[i]>=9) and (x[i]<14) :
            h[i] = 2.507355893131E+01 + 9.754803562315E-01*x[i] - \
                   1.016116352781E-01*x[i]**2 + 1.889794677828E-03*x[i]**3;
        elif (x[i]>=14) and (x[i]<20) :
            h[i] = 2.579601052357E+01 + 8.206693007457E-01*x[i] - \
                   9.055370274339E-02*x[i]**2 + 1.626510569859E-03*x[i]**3;
        elif (x[i]>=20) and (x[i]<30) :
            h[i] = 4.046435022819E+01 - 1.379581654948E+00*x[i] + \
                   1.945884504128E-02*x[i]**2 - 2.070318932190E-04*x[i]**3;
        elif (x[i]>=30) and (x[i]<40) :
            h[i] = 1.792461334664E+01 + 8.743920332081E-01*x[i] - \
                   5.567361123058E-02*x[i]**2 + 6.277731764683E-04*x[i]**3;
        elif (x[i]>=40) and (x[i]<=54) :
            h[i] = np.maximum(0., 5.639011190988E+01 - 2.010520359035E+00*x[i] + \
                              1.644919857549E-02*x[i]**2 + 2.674976141766E-05*x[i]**3);
        elif (x[i]>54) and (x[i]<=126) :
            h[i] = 0;

    hout = h/28.0
    return hout

# Seperate numbers by a dot: . 
# e+01 notation works

#Set: x = [t] ;; Scale = 1.00

## Start = 0,00 ;; End = 9,00
# y = Min(28.0, 2.8e+01 + 6.775070969851e-03*Pow([t],2) - 2.124527775800e-03*Pow([t],3))

## Start = 9,00 ;; End = 14,00
# y = 2.507355893131e+01 + 9.754803562315e-01*[t] - 1.016116352781e-01*Pow([t],2) + 1.889794677828e-03*Pow([t],3)

## Start = 14,00 ;; End = 20,00
# y = 2.579601052357e+01 + 8.206693007457e-01*[t] - 9.055370274339e-02*Pow([t],2) + 1.626510569859e-03*Pow([t],3)

## Start = 20,00 ;; End = 30,00
# y = 4.046435022819e+01 - 1.379581654948e+00*[t] + 1.945884504128e-02*Pow([t],2) - 2.070318932190e-04*Pow([t],3)

## Start = 30,00 ;; End = 40,00
# y = 1.792461334664e+01 + 8.743920332081e-01*[t] - 5.567361123058e-02*Pow([t],2) + 6.277731764683e-04*Pow([t],3)

## Start = 40,00 ;; End = 54,00
# y = Max(0.0, 5.639011190988e+01 - 2.010520359035e+00*[t] + 1.644919857549e-02*Pow([t],2) + 2.674976141766e-05*Pow([t],3))

## Start = 54,00 ;; End = 198,00  <- middle part, from now on periodic back
# y = 0.0

## Start = 198,00 ;; End = 212,00
# y = Max(0.0, 5.639011190988e+01 - 2.010520359035e+00*(252-[t]) + 1.644919857549e-02*Pow((252-[t]),2) + 2.674976141766e-05*Pow((252-[t]),3))

## Start = 212,00 ;; End = 222,00
# y = 1.792461334664e+01 + 8.743920332081e-01*(252-[t]) - 5.567361123058e-02*Pow((252-[t]),2) + 6.277731764683e-04*Pow((252-[t]),3)

## Start = 222,00 ;; End = 232,00
# y = 4.046435022819e+01 - 1.379581654948e+00*(252-[t]) + 1.945884504128e-02*Pow((252-[t]),2) - 2.070318932190e-04*Pow((252-[t]),3)

## Start = 232,00 ;; End = 238,00
# y = 2.579601052357e+01 + 8.206693007457e-01*(252-[t]) - 9.055370274339e-02*Pow((252-[t]),2) + 1.626510569859e-03*Pow((252-[t]),3)

## Start = 238,00 ;; End = 243,00
# y = 2.507355893131e+01 + 9.754803562315e-01*(252-[t]) - 1.016116352781e-01*Pow((252-[t]),2) + 1.889794677828e-03*Pow((252-[t]),3)

## Start = 243,00 ;; End = 252,00
# y = Min(28.0, 2.8e+01 + 6.775070969851e-03*Pow((252-[t]),2) - 2.124527775800e-03*Pow((252-[t]),3))

### Hill is done

## Inlet:  Draw by hand; should be 57,008 mm 
# Calculations: Full heigth should be calculated by: 3,036 * 28 = 85,008 (the scaling factor we used the whole time)
# From full height one can substract the hill height which is 28 => 57,008
# Alternatively one can also just multiply by 2,036 instead XD
    
    
## Outlet: See above
    

## Ceiling: Just connect

 
# scales hill according to non-zero alpha and variable domain length
def para_profile(yy_a, a, L):
    'Calculate the shape of the parameterized periodic hill with variable length'
    import numpy as np

    y = np.array(yy_a)
    h = np.zeros(len(y))
    
    # Base canonical width is 54, scaling factor is 28.
    hill_width = (54.0 / 28.0) * a
    
    for i in range(len(y)):
        if y[i] <= hill_width:
            # Map physical y coordinate back to canonical left hill x (0 to 54)
            x = (y[i] / a) * 28.0
        elif y[i] >= L - hill_width:
            # Map physical y coordinate back to canonical right hill x (distance from end)
            x = ((L - y[i]) / a) * 28.0
        else:
            # Flat middle section
            h[i] = 0.0
            continue

        # Evaluate standard polynomials for generated canonical x
        if (x>=0) and (x<9) :
            h[i] = np.minimum(28., 2.8e+01 + 0.0e+00*x + \
                              6.775070969851e-03*x**2 - 2.124527775800e-03*x**3)
        elif (x>=9) and (x<14) :
            h[i] = 2.507355893131E+01 + 9.754803562315E-01*x - \
                   1.016116352781E-01*x**2 + 1.889794677828E-03*x**3
        elif (x>=14) and (x<20) :
            h[i] = 2.579601052357E+01 + 8.206693007457E-01*x - \
                   9.055370274339E-02*x**2 + 1.626510569859E-03*x**3
        elif (x>=20) and (x<30) :
            h[i] = 4.046435022819E+01 - 1.379581654948E+00*x + \
                   1.945884504128E-02*x**2 - 2.070318932190E-04*x**3
        elif (x>=30) and (x<40) :
            h[i] = 1.792461334664E+01 + 8.743920332081E-01*x - \
                   5.567361123058E-02*x**2 + 6.277731764683E-04*x**3
        elif (x>=40) and (x<=54) :
            h[i] = np.maximum(0., 5.639011190988E+01 - 2.010520359035E+00*x + \
                              1.644919857549E-02*x**2 + 2.674976141766E-05*x**3)

    hout = h / 28.0
    return y, hout





if __name__ == "__main__":
        
    import matplotlib.pyplot as plt
    yy=np.arange(0, 9, 0.01)
    print(yy)
    
    h = profile(yy)

    symbols=['k-', 'g-', 'b-', 'm-', 'r-']
    #alphas = np.array([0.5, 0.8, 1, 1.2, 1.5])
    
    alphas = np.array([1.0])
    
    for i, a in enumerate(alphas):
    
        ya, ha = para_profile(yy, a, 9.0)
        plt.plot(ya, ha, symbols[i])
        xend = ya[-1]
        
        # plots: inlet-wall; outlet-wall; ceiling
        # NOTICE: 3.036 float (was originally at 3.06???)
        outline = np.array([[0, 0, xend, xend], [1, 3.06, 3.06, 1]])
        plt.plot(outline[0, :], outline[1, :], symbols[i])

    plt.axis([0, 5, 0, 5])
    plt.axis('equal')
    plt.savefig('para-shapes.pdf')
    plt.show()

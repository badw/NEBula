import matplotlib.pyplot as plt 
import numpy as np 
from ase.mep import NEBTools 

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import PeriodicSite
from pymatgen.core import Structure


class NEBulaPlotter:

    def __init__(self):
        """
        nebulaplotter
        """
        self.plot_kws = {'marker': 'o',
                           'linestyle': 'None',
                           'color': 'grey',
                           'markerfacecolor': 'white',
                           'markersize': 5}
        
        self.subplot_kws = {'figsize':(6,6)}

    def nebanalysis(self,neb):
        nebtools = NEBTools(neb)
        Ef,dE = nebtools.get_barrier()
        fit = nebtools.get_fit()
        self.fit = fit 
        self.Ef = Ef
        self.dE = dE

    def save_neb_as_poscar(self,images,mobile_species='H',filename='poscar.vasp'):

        neb = [AseAtomsAdaptor.get_structure(x) for x in images] # assumes atoms
        sites = []
        sites.append(neb[0][-1])
        sites.append(neb[-1][-1])
        for s in neb[1:-1]:
            sites.append(PeriodicSite(mobile_species,s[-1].frac_coords,s.lattice))
        sites.extend(neb[0].sites[1:])
        Structure.from_sites(sites).to(filename=filename,fmt='poscar')
    
    def plot_neb(self,ax=None,subplot_kws={},plot_kws = {}):
        self.plot_kws.update(plot_kws)
        self.subplot_kws.update(subplot_kws)

    
        if not ax:
            fig,ax = plt.subplots(**self.subplot_kws)

        ax.plot(self.fit[2],self.fit[3])
        ax.plot(self.fit[0],self.fit[1],**self.plot_kws)
        ax.set_ylabel('Energy (eV)')
        ax.set_xlabel('Path Coordinate (arb. units)')

        return(ax)


        





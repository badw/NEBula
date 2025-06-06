
from ase.optimize import MDMin, BFGS, BFGSLineSearch, FIRE
from chgnet.model import CHGNetCalculator
from typing import Union, Optional 
from pymatgen.core import Structure 
from ase.atoms import Atoms 
from pymatgen.io.ase import AseAtomsAdaptor
import copy 
from ase.mep import NEB 


class NEBulaRelax:
    def __init__(self,asecalculator):
        self.relax_models = {
            "MDMin": MDMin,
            "BFGS": BFGS,
            "BFGSLineSearch": BFGSLineSearch,
            "FIRE":FIRE
        }
        self.asecalculator = asecalculator

    def relax(
        self,
        structure: Optional[Union[Structure, Atoms]],
        fmax: float = 0.1,
        steps: int = 1000,
        relax_model: str = "MDMin",
        **relax_kws,
    ):
        """
        relaxes a structure given an ase calculator object 
        """
        self.chosen_relax_model = self.relax_models[relax_model] # this is so that things are consistent between this and NEB

        if isinstance(structure,Structure):
            atoms = AseAtomsAdaptor.get_atoms(
                copy.deepcopy(structure)
            )  # convert pymatgen to ase
        else:
            atoms = copy.deepcopy(structure)

        atoms.calc = self.asecalculator
        dyn = self.relax_models[relax_model](atoms)
        dyn.run(fmax=fmax, steps=steps,**relax_kws) # i believe this doesn't relax the cell 
        energy = atoms.get_potential_energy()
        struct = AseAtomsAdaptor.get_structure(atoms) 
        #fstruct = struct.get_sorted_structure()
        return(energy, struct)
    
    def nebrun(
            self,
            init_structure:Optional[Union[Structure, Atoms]],
            final_structure:Optional[Union[Structure, Atoms]],
            fmax:float=0.1,
            relax_endpoints:bool = True,
            nimages:int=5,
            climbing_image:bool = True,
            maxstep:int=200,
    ):

        if relax_endpoints:
            _,init_structure = self.relax(init_structure)
            _,final_structure = self.relax(final_structure)

        if isinstance(init_structure,Structure):
            init_structure = AseAtomsAdaptor.get_atoms(init_structure)
        else:
            init_structure = init_structure
        if isinstance(final_structure,Structure):
            final_structure = AseAtomsAdaptor.get_atoms(final_structure)
        else:
            final_structure = final_structure
        
        images = [init_structure.copy() for x in range(nimages+1)]
        images.append(final_structure.copy())
        for image in images:
            image.calc = self.asecalculator
        neb = NEB(images,climb=climbing_image,allow_shared_calculator=True) # uses a shared calculator - perhaps a singlecalculatorNEb is more effective?
        neb.interpolate()
        dyn = self.chosen_relax_model(neb)
        dyn.run(fmax=fmax,steps=maxstep)
        return(images)
